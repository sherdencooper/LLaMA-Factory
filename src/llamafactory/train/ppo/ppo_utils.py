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

import json
from contextlib import nullcontext
from typing import TYPE_CHECKING, Literal, Optional, List

import torch
from transformers.integrations import is_deepspeed_zero3_enabled

from ...extras.packages import is_requests_available
import importlib.util
import sys
import os

if is_requests_available():
    import requests


if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from trl import AutoModelForCausalLMWithValueHead


def get_rewards_from_server(server_url: str, messages: list[str]) -> list["torch.Tensor"]:
    r"""Get reward scores from the API server."""
    headers = {"Content-Type": "application/json"}
    payload = {"model": "model", "messages": messages}
    response = requests.post(server_url, json=payload, headers=headers)
    rewards = json.loads(response.text)["scores"]
    return torch.Tensor(rewards)

def get_rewards_from_rule(reward_model: str, queries: List[str], responses: List[str], labels: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Gets reward scores from the rule-based reward model.

    Args:
        reward_model: Path to the Python file containing the reward function, optionally with function name after colon
        queries: List of query strings
        responses: List of response strings
        labels: Optional tensor of labels

    Returns:
        Tensor of reward scores
    """
    # Extract module path and function name if specified
    if ":" in reward_model:
        module_path, func_name = reward_model.split(":", 1)
    else:
        module_path = reward_model
        func_name = "reward"

    # Verify the file exists
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")

    # Extract module name from path
    module_name = os.path.basename(module_path).replace(".py", "")

    try:
        # Load the module from the specified path
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create module spec from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Check if the reward function exists
        if not hasattr(module, func_name):
            raise AttributeError(f"Module {module_name} does not have a {func_name}() function")

        # Call the reward function
        reward_func = getattr(module, func_name)
        if labels is not None:
            rewards = reward_func(queries, responses, labels)
        else:
            rewards = reward_func(queries, responses)

        # Convert to tensor if it isn't already
        if not isinstance(rewards, torch.Tensor):
            if isinstance(rewards, list):
                rewards = torch.tensor([[r] for r in rewards], dtype=torch.float)
            else:
                rewards = torch.tensor([[rewards]], dtype=torch.float)

        # Ensure the tensor has the right shape [batch_size, 1]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)

        return rewards

    except Exception as e:
        # Add more context to the error message
        raise type(e)(f"{str(e)} - Error loading or calling reward function from {module_path}:{func_name}") from e


def replace_model(model: "AutoModelForCausalLMWithValueHead", target: Literal["default", "reward"]) -> None:
    r"""Replace the default/reward modules in the model. The model is already unwrapped."""
    v_head_layer = model.v_head.summary
    if is_deepspeed_zero3_enabled():
        import deepspeed  # type: ignore

        params = [v_head_layer.weight, v_head_layer.bias]
        context_maybe_zero3 = deepspeed.zero.GatheredParameters(params, modifier_rank=0)
    else:
        context_maybe_zero3 = nullcontext()

    model.pretrained_model.set_adapter(target)  # set the LoRA adapter to be active
    with context_maybe_zero3:
        if target == "reward":  # save default head temporarily
            setattr(model, "default_head_weight", v_head_layer.weight.data.detach().clone())
            setattr(model, "default_head_bias", v_head_layer.bias.data.detach().clone())

        device = v_head_layer.weight.device
        v_head_layer.weight.data = model.get_buffer(f"{target}_head_weight").detach().clone().to(device)
        v_head_layer.bias.data = model.get_buffer(f"{target}_head_bias").detach().clone().to(device)


def dump_layernorm(model: "PreTrainedModel") -> dict[str, "torch.Tensor"]:
    r"""Dump the layernorm parameters in the model. The model is already unwrapped (and gathered)."""
    layer_norm_params = {}
    for name, param in model.named_parameters():
        if param.data.dtype == torch.float32:
            layer_norm_params[name] = param.data.detach().clone()
            param.data = param.data.to(model.config.torch_dtype)

    return layer_norm_params


def restore_layernorm(model: "PreTrainedModel", layernorm_params: Optional[dict[str, "torch.Tensor"]] = None) -> None:
    r"""Restore the layernorm parameters in the model. The model is already unwrapped (and gathered)."""
    for name, param in model.named_parameters():
        if name in layernorm_params:
            param.data = layernorm_params[name]
