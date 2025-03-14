# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/ppo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override

from ...extras import logging
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm, get_rewards_from_rule
import time

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


class CustomPPOTrainer(PPOTrainer, Trainer):
    r"""Inherit PPOTrainer."""

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[list["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
            remove_unused_columns=training_args.remove_unused_columns,
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning_rank0("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.reward_model = reward_model
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info_rank0("max_steps is given, it will override any value given in num_train_epochs")
        # find token id for the \n token
        newline_token_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        newlines_token_id = self.tokenizer.encode("\n\n", add_special_tokens=False)[0]
        periodnewline_token_id = self.tokenizer.encode(".\n", add_special_tokens=False)[0]
        periodnewlines_token_id = self.tokenizer.encode(".\n\n", add_special_tokens=False)[0]
        self.newline_token_id = [newline_token_id, newlines_token_id, periodnewline_token_id, periodnewlines_token_id]

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if finetuning_args.reward_model_type == "full":
            if self.is_deepspeed_enabled:
                if not (
                    getattr(reward_model.pretrained_model, "is_loaded_in_8bit", False)
                    or getattr(reward_model.pretrained_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.reward_model = self._prepare_deepspeed(self.reward_model)
            else:
                self.reward_model = self.accelerator.prepare_model(self.reward_model, evaluation_mode=True)

        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""Implement training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer."""
        if resume_from_checkpoint is not None:
            raise ValueError("`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        logger.info_rank0("***** Running training *****")
        logger.info_rank0(f"  Num examples = {num_examples:,}")
        logger.info_rank0(f"  Num Epochs = {num_train_epochs:,}")
        logger.info_rank0(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        logger.info_rank0(
            f"  Total train batch size (w. parallel, buffer, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info_rank0(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps:,}")
        logger.info_rank0(f"  Num optimization epochs per batch = {self.finetuning_args.ppo_epochs:,}")
        logger.info_rank0(f"  Total training steps = {max_steps:,}")
        logger.info_rank0(f"  Number of trainable parameters = {count_parameters(self.model)[0]:,}")

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            self.model.eval()
            self.tokenizer.padding_side = "right"  # change padding side
            queries, responses, rewards = [], [], []
            for idx in range(0, self.config.batch_size, self.config.mini_batch_size):
                mini_batch_queries, mini_batch_responses = self.get_inputs(
                    batch[idx : idx + self.config.mini_batch_size]
                )
                if self.config.remove_unused_columns!=True:
                    mini_batch_labels = batch["labels"][idx : idx + self.config.mini_batch_size]
                else:
                    mini_batch_labels = None
                mini_batch_rewards = self.get_rewards(mini_batch_queries, mini_batch_responses, mini_batch_labels)
                # print how many trajectories with reward smaller than 1
                print("There are {} trajectories to explore".format(sum(mini_batch_rewards < 1)))
                if self.finetuning_args.gpo_explore_trajectory:
                    for i in range(len(mini_batch_queries)):
                        if mini_batch_rewards[i] < 1:
                            print("Running GPO Exploration")
                            mini_batch_queries[i], mini_batch_responses[i], mini_batch_rewards[i] = self.go_and_explore(mini_batch_queries[i], mini_batch_responses[i], mini_batch_rewards[i], mini_batch_labels[i])
                        
                queries.extend(mini_batch_queries)
                responses.extend(mini_batch_responses)
                rewards.extend(mini_batch_rewards)

            # Run PPO step
            self.model.train()
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning_rank0("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: dict[str, "torch.Tensor"]) -> tuple[list["torch.Tensor"], list["torch.Tensor"]]:
        r"""Generate model's responses given queries."""
        if batch["input_ids"].size(0) == 1:  # handle llama2 ppo with gradient accumulation > 1
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_output: torch.Tensor = unwrapped_model.generate(
                generation_config=self.generation_config, logits_processor=get_logits_processor(), **batch
            )
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1) :].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:  # allow empty response
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:  # include eos token
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])  # remove padding from left
            responses.append(response[i, :response_length])  # remove padding from right

        return queries, responses

    @torch.no_grad()
    def get_rewards(
        self,
        queries: list["torch.Tensor"],
        responses: list["torch.Tensor"],
        labels: Optional["torch.Tensor"] = None,
    ) -> list["torch.Tensor"]:
        r"""Compute scores using given reward model.

        Both inputs and outputs are put on CPU.
        """
        if self.finetuning_args.reward_model_type == "rule":
            if not isinstance(self.reward_model, str):
                raise ValueError("For 'rule' reward type, reward_model should be a string path to the Python function")
            
            decoded_queries = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
            decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
            
            # Pre-process labels to replace -100 values before decoding
            labels_for_decode = labels.clone()
            # Replace -100 with a safe token ID (usually pad token)
            labels_for_decode[labels_for_decode == -100] = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            decoded_labels = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            return get_rewards_from_rule(self.reward_model, decoded_queries, decoded_responses, decoded_labels)
        
        if self.finetuning_args.reward_model_type == "api":
            token_ids = [torch.cat((q, r), dim=-1).tolist() for q, r in zip(queries, responses)]
            messages = self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
            return get_rewards_from_server(self.reward_model, messages)

        batch: dict[str, torch.Tensor] = self.prepare_model_inputs(queries, responses)
        unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="reward")
            reward_model = self.model
        else:
            reward_model = self.reward_model

        with unwrap_model_for_generation(reward_model, self.accelerator), self.amp_context:  # support bf16
            values: torch.Tensor = reward_model(**batch, return_dict=True, use_cache=False)[-1]

        if self.finetuning_args.reward_model_type == "lora":
            replace_model(unwrapped_model, target="default")

        rewards = values.gather(dim=-1, index=(batch["attention_mask"].sum(dim=-1, keepdim=True) - 1))
        return rewards.float().detach()  # use fp32 type
    
    @torch.no_grad()
    def go_and_explore(self, query: torch.Tensor, response: torch.Tensor, reward: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Go and explore the trajectory with parallel step processing."""
        MAX_STEPS = self.finetuning_args.gpo_max_steps
        MIN_STEP_LENGTH = 30
        NUM_EXPLORATIONS = self.finetuning_args.gpo_explore_num
        NUM_STEP_PARALLEL = self.finetuning_args.gpo_step_parallel  # Number of steps to process in parallel

        response_list = response.tolist()
        step_indices = [i for i, token_id in enumerate(response_list) if token_id in self.newline_token_id]

        # Initialize steps
        steps = []
        current_step_start = 0

        # Group by newlines, ensuring minimum step length
        for i, idx in enumerate(step_indices):
            # If we've reached the maximum number of steps, group the rest
            if len(steps) >= MAX_STEPS - 1:
                steps.append((current_step_start, len(response_list) - 1))
                break

            step_length = idx - current_step_start

            # If step is too short, continue to next newline
            if step_length < MIN_STEP_LENGTH and i < len(step_indices) - 1:
                continue

            # Add the step
            steps.append((current_step_start, idx))
            current_step_start = idx + 1

        # Add the final step if needed
        if current_step_start < len(response_list) and len(steps) < MAX_STEPS:
            steps.append((current_step_start, len(response_list) - 1))

        if len(steps) == 0:
            return query, response, reward

        # Create a modified generation config for batch generation
        batch_generation_config = GenerationConfig(
            **self.generation_config.to_dict(),
        )
        batch_generation_config.num_return_sequences = NUM_EXPLORATIONS
        batch_generation_config.do_sample = True  # Ensure diversity in generated sequences

        print(f"There are {len(steps)} steps to explore")
        start_time = time.time()

        # Unwrap model once for all generations
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            # Track best step and response
            best_step_avg_reward = 0.0
            best_step_idx = -1
            best_responses_per_step = {}
            best_rewards_per_step = {}

            # Process steps in parallel batches
            for batch_start in range(0, len(steps), NUM_STEP_PARALLEL):
                batch_end = min(batch_start + NUM_STEP_PARALLEL, len(steps))
                current_batch_steps = steps[batch_start:batch_end]

                # Prepare inputs for all steps in this batch
                batch_inputs = []
                max_length = 0
                for step_start, step_end in current_batch_steps:
                    partial_response = response[:step_start]
                    input_tensor = torch.cat([query, partial_response], dim=0).unsqueeze(0)
                    max_length = max(max_length, input_tensor.size(1))

                # Create padded inputs
                for step_start, step_end in current_batch_steps:
                    partial_response = response[:step_start]
                    input_tensor = torch.cat([query, partial_response], dim=0)

                    # Pad to max_length
                    pad_length = max_length - input_tensor.size(0)
                    if pad_length > 0:
                        padding = torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=input_tensor.dtype)
                        input_tensor = torch.cat([padding, input_tensor], dim=0)

                    # Add to batch (no need to repeat)
                    batch_inputs.append(input_tensor.unsqueeze(0))

                # Concatenate all inputs into a single batch
                batch_size = len(current_batch_steps)
                if batch_size == 0:
                    continue
                
                combined_inputs = torch.cat(batch_inputs, dim=0)
                attention_mask = (combined_inputs != self.tokenizer.pad_token_id).long()

                # Prepare batch for generation
                batch = {
                    "input_ids": combined_inputs,
                    "attention_mask": attention_mask
                }

                # Send the batch to the model's device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}

                # Generate continuations for all steps in this batch
                generate_output = unwrapped_model.generate(
                    generation_config=batch_generation_config,
                    logits_processor=get_logits_processor(),
                    **batch
                )

                # Process the results for each step
                for batch_idx, (step_idx, (step_start, step_end)) in enumerate(zip(range(batch_start, batch_end), current_batch_steps)):
                    step_rewards = []
                    step_best_reward = 0.0
                    step_best_response = None

                    # Process each exploration for this step
                    for exp_idx in range(NUM_EXPLORATIONS):
                        output_idx = batch_idx * NUM_EXPLORATIONS + exp_idx
                        input_length = combined_inputs[batch_idx].size(0)

                        # Extract the generated continuation
                        new_response = generate_output[output_idx, input_length:].detach().cpu()

                        # Evaluate the reward for this continuation
                        new_rewards = self.get_rewards([query], [new_response], label.unsqueeze(0) if label is not None else None)
                        new_reward = new_rewards[0].item()
                        step_rewards.append(new_reward)

                        # Track best response for this step
                        if new_reward > step_best_reward:
                            step_best_reward = new_reward
                            step_best_response = new_response

                    # Calculate average reward for this step
                    step_avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0

                    # Store best response and reward for this step
                    best_responses_per_step[step_idx] = step_best_response
                    best_rewards_per_step[step_idx] = step_best_reward

                    # Track step with highest average reward
                    if step_avg_reward > best_step_avg_reward:
                        best_step_avg_reward = step_avg_reward
                        best_step_idx = step_idx

                # Check time limit
                end_time = time.time()
                if end_time - start_time > self.finetuning_args.gpo_max_time:
                    print("Time limit reached, early terminating")
                    break

                # Check if we found a very good reward
                if best_step_avg_reward >= 1:
                    print(f"Found excellent reward {best_step_avg_reward}, early terminating")
                    break

            # Restore layernorm parameters
            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        print("Exploration finished")

        # If we found a better step with positive reward, return its best response
        if best_step_idx >= 0 and best_rewards_per_step[best_step_idx] > 0:
            # remove the padding from the best response, note that sometimes the pad token is also the eos token
            print(f"Best step index: {best_step_idx}")
            print(f"Best average reward: {best_rewards_per_step[best_step_idx]}")
            for i in range(len(best_responses_per_step[best_step_idx])):
                if best_responses_per_step[best_step_idx][i] == self.tokenizer.pad_token_id:
                    if self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                        remove_pad_response = best_responses_per_step[best_step_idx][:i+1]
                    else:
                        remove_pad_response = best_responses_per_step[best_step_idx][:i]
                    break
            # concat the partial response in the best step
            partial_response = response[:steps[best_step_idx][0]]
            best_response = torch.cat([partial_response, remove_pad_response], dim=0)
            best_reward = best_rewards_per_step[best_step_idx]
            return query, best_response, best_reward
        else:
            return query, response, reward

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        r"""Calculate model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:  # support bf16
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat((torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""Save model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)  # must be called at all ranks
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning_rank0(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)

        elif self.args.should_save:
            unwrapped_model: AutoModelForCausalLMWithValueHead = self.accelerator.unwrap_model(self.model)
            self._save(output_dir, state_dict=unwrapped_model.state_dict())
