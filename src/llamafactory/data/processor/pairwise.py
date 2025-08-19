# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example_multi_turn(
        self,
        prompt: list[dict[str, str]],
        chosen_response: list[dict[str, str]],
        rejected_response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """
        Encode multi-turn conversations for DPO training.
        
        Args:
            prompt: The conversation history (all turns before the final responses)
            chosen_response: The chosen conversation continuation (can be multi-turn)
            rejected_response: The rejected conversation continuation (can be multi-turn)
            
        Returns:
            Tuple of (chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels)
        """
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + chosen_response, images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + rejected_response, images, videos, audios, self.processor
        )
        
        # Use encode_multiturn to properly handle multi-turn conversations
        chosen_encoded_pairs = self.template.encode_multiturn(self.tokenizer, chosen_messages, system, tools)
        rejected_encoded_pairs = self.template.encode_multiturn(self.tokenizer, rejected_messages, system, tools)

        # Calculate how many assistant responses are in the shared prompt
        # Since prompt should end with a user message (odd length), the number of assistant responses is len(prompt) // 2
        prompt_assistant_turns = len(prompt) // 2
        
        # Process the encoded pairs to create input_ids and labels with proper masking
        chosen_input_ids, chosen_labels = self._process_multiturn_pairs(chosen_encoded_pairs, prompt_assistant_turns)
        rejected_input_ids, rejected_labels = self._process_multiturn_pairs(rejected_encoded_pairs, prompt_assistant_turns)
        
        # Apply multimedia processing
        chosen_input_ids, _ = self.template.mm_plugin.process_token_ids(
            chosen_input_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        rejected_input_ids, _ = self.template.mm_plugin.process_token_ids(
            rejected_input_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        
        # Apply sequence length constraints
        max_len = max(len(chosen_input_ids), len(rejected_input_ids))
        source_len, target_len = infer_seqlen(0, max_len, self.data_args.cutoff_len)
        
        chosen_input_ids = chosen_input_ids[:target_len]
        chosen_labels = chosen_labels[:target_len]
        rejected_input_ids = rejected_input_ids[:target_len]
        rejected_labels = rejected_labels[:target_len]
        
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def _process_multiturn_pairs(self, encoded_pairs: list, prompt_turns: int) -> tuple[list[int], list[int]]:
        """
        Process multi-turn encoded pairs to create proper input_ids and labels.
        
        Args:
            encoded_pairs: List of (source_ids, target_ids) pairs from encode_multiturn
            prompt_turns: Number of assistant responses that are part of the shared prompt (to be masked)
            
        Returns:
            Tuple of (input_ids, labels) with proper masking
        """
        input_ids, labels = [], []
        
        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            # Always mask user messages (source_ids)
            if self.template.efficient_eos and turn_idx != 0:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_label = [IGNORE_INDEX] * len(source_ids)
            
            # For assistant responses (target_ids):
            # - Mask responses that are part of the shared prompt
            # - Keep responses that are part of chosen/rejected continuation for training
            if turn_idx < prompt_turns:
                # This is part of the shared conversation history - mask it
                target_label = [IGNORE_INDEX] * len(target_ids)
            else:
                # This is part of the chosen/rejected continuation - train on it
                target_label = target_ids
                
            input_ids.extend(source_ids + target_ids)
            labels.extend(source_label + target_label)
        
        if self.template.efficient_eos:
            input_ids.append(self.tokenizer.eos_token_id)
            labels.append(self.tokenizer.eos_token_id)
            
        return input_ids, labels

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            # Check if this is multi-turn DPO format (chosen/rejected are lists)
            is_multi_turn = (
                len(examples["_response"][i]) >= 2 and
                isinstance(examples["_response"][i][0], list) and
                isinstance(examples["_response"][i][1], list)
            )
            
            if is_multi_turn:
                # Multi-turn DPO: _response contains [chosen_turns, rejected_turns]
                chosen_response = examples["_response"][i][0]
                rejected_response = examples["_response"][i][1]
                
                # Validate format
                if len(examples["_prompt"][i]) % 2 != 1:
                    logger.warning_rank0(
                        "Dropped invalid multi-turn example - prompt must have odd number of turns: {}".format(examples["_prompt"][i])
                    )
                    continue
                    
                chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example_multi_turn(
                    prompt=examples["_prompt"][i],
                    chosen_response=chosen_response,
                    rejected_response=rejected_response,
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )
            else:
                # Original single-turn DPO format
                if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                    logger.warning_rank0(
                        "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                    )
                    continue

                chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                    prompt=examples["_prompt"][i],
                    response=examples["_response"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )
                
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
