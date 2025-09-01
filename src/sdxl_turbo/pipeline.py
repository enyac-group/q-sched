# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
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


import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from collections import namedtuple
import math
import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import QConfig, MinMaxObserver, PlaceholderObserver, QuantStub, DeQuantStub

import mixdq_extension._C

from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    IPAdapterMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_invisible_watermark_available,
    is_torch_xla_available,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

import copy
import itertools
import warnings
import functools
import threading

import torch.ao.nn.quantized as nnq
from torch.ao.nn.intrinsic import _FusedModule

from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_static_quant_module_mappings,
    get_default_static_quant_reference_module_mappings,
    get_default_qat_module_mappings,
    get_default_qconfig_propagation_list,
    no_observer_set,
    _has_special_act_post_process,
    _get_special_act_post_process,
)
from torch.ao.quantization.utils import get_qparam_dict, has_no_children_ignoring_parametrizations
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
from torch.ao.quantization.qconfig import (
    _add_module_to_qconfig_obs_ctr,
    default_dynamic_qconfig,
    float16_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    float_qparams_weight_only_qconfig_4bit,
    _activation_is_memoryless)
from torch.nn.utils.parametrize import type_before_parametrizations
from torch.ao.quantization.observer import _is_activation_post_process

# TODO remove this once BC is no longer required to avoid a SEV
from torch.ao.quantization.observer import (   # noqa: F401
    _is_activation_post_process as is_activation_post_process
)

if is_invisible_watermark_available():
    from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

XLA_AVAILABLE = False


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionXLPipeline

        >>> pipe = StableDiffusionXLPipeline.from_pretrained(
        ...     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

######################################################################################################
# quant ops

quantize_per_tensor = mixdq_extension._C.quantize_per_tensor_to_int8


def qconv2d(
    input_int,
    weight_int,
    weight_scale,
    input_scale,
    input_zp,
    scale,
    weight_sum_by_input_channels,
    bias0,
    bias=None,
    stride=1,
    padding=0,
):
    dilation = 1
    return mixdq_extension._C.qconv2d_w8_a8_ohalf(
        input_int, weight_int, weight_scale, input_scale, input_zp,
        scale, weight_sum_by_input_channels, bias0, 
        bias, stride, padding, dilation
    )


qlinear = mixdq_extension._C.qlinear_w8_a8_ohalf

# quant ops
######################################################################################################


######################################################################################################
# the code below is for converting the NN model

__all__ = [
    "get_default_custom_config_dict",
    "propagate_qconfig_",
    "add_quant_dequant",
    "prepare",
    "quantize",
    "quantize_dynamic",
    "prepare_qat",
    "quantize_qat",
    "convert",
    "swap_module",
    'QuantizedLinear',
    'QuantizedConv2d',
]


_DEFAULT_CUSTOM_CONFIG_DICT = {
    'float_to_observed_custom_module_class': {
        nn.LSTM: nn.quantizable.LSTM,
        nn.MultiheadAttention: nn.quantizable.MultiheadAttention,
    },
    'observed_to_quantized_custom_module_class': {
        nn.quantizable.LSTM: nn.quantized.LSTM,
        nn.quantizable.MultiheadAttention: nn.quantized.MultiheadAttention,
    }
}

_SPLIT = [1280, 1280, 1280, 1280, 640, 640, 640, 320, 320]  # For SDXL-Turbo

# global num
_NUM = 0


def get_default_custom_config_dict():
    r"""Defines the default custom config dict.
    """
    return _DEFAULT_CUSTOM_CONFIG_DICT


def _propagate_qconfig_helper(module, qconfig_dict,
                              qconfig_parent=None, prefix='', prepare_custom_config_dict=None):
    r"""This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict
        prepare_custom_config_dict: dictionary for custom handling of modules
                                    see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """

    module_qconfig = qconfig_dict.get(
        type_before_parametrizations(module), qconfig_parent)
    module_qconfig = qconfig_dict.get(prefix, module_qconfig)
    module_qconfig = getattr(module, 'qconfig', module_qconfig)

    torch.ao.quantization.qconfig._assert_valid_qconfig(module_qconfig, module)

    qconfig_with_device_check = _add_module_to_qconfig_obs_ctr(
        module_qconfig, module)
    module.qconfig = qconfig_with_device_check

    for name, child in module.named_children():
        module_prefix = prefix + '.' + name if prefix else name
        #  do no not propagate qconfig to child if child is non traceable
        if prepare_custom_config_dict is None or not (
            name in prepare_custom_config_dict.get(
                "non_traceable_module_name", [])
            or type(child) in prepare_custom_config_dict.get("non_traceable_module_class", [])
        ):
            _propagate_qconfig_helper(
                child, qconfig_dict, qconfig_with_device_check, module_prefix
            )


def propagate_qconfig_(module, qconfig_dict=None, prepare_custom_config_dict=None):
    r"""Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)
        prepare_custom_config_dict: dictionary for custom handling of modules
            see docs for :func:`~torch.ao.quantization.prepare_fx`

    Return:
        None, module is modified inplace with qconfig attached
    """
    if qconfig_dict is None:
        qconfig_dict = {}
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = {}
    _propagate_qconfig_helper(
        module, qconfig_dict, prepare_custom_config_dict=prepare_custom_config_dict)


def _observer_forward_hook(self, input, output):
    r"""Forward hook that calls observer on the output
    """
    return self.activation_post_process(output)


def _observer_forward_pre_hook(self, input):
    r"""Forward pre hook that calls observer on the output
    """
    return self.activation_post_process(input[0])


def _register_activation_post_process_hook(module, pre_hook=False):
    assert hasattr(module, 'activation_post_process'), \
        'Expect activation_post_process attribute already attached to the module'
    if pre_hook:
        handle = module.register_forward_pre_hook(
            _observer_forward_pre_hook, prepend=True
        )
    else:
        handle = module.register_forward_hook(
            _observer_forward_hook, prepend=True
        )


def _add_observer_(module, qconfig_propagation_list=None, non_leaf_module_list=None, device=None, custom_module_class_mapping=None):
    r"""Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules that we want to quantize
        qconfig_propagation_list: a list of quantizable modules that will have observers added to them
            if they are leaf nodes
        device: parent device, if any
        non_leaf_module_list: list of non-leaf modules we want to add observer

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    if qconfig_propagation_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()

    if custom_module_class_mapping is None:
        custom_module_class_mapping = {}

    # respect device affinity when adding observers
    if device is None:
        devices = _get_unique_devices_(module)
        assert len(devices) <= 1, (
            f"_add_observer_ only works with cpu or single-device CUDA modules, but got devices {devices}"
        )
        device = next(iter(devices)) if len(devices) > 0 else None

    def get_activation_post_process(qconfig, device, special_act_post_process=None):
        activation = qconfig.activation(
        ) if special_act_post_process is None else special_act_post_process()
        if device is not None:
            activation.to(device)
        return activation

    def needs_observation(m):
        return hasattr(m, 'qconfig') and m.qconfig is not None

    def insert_activation_post_process(m, special_act_post_process=None):
        """ Adds an activation post process module and register
        a pre or post hook that calls the module
        """
        # We don't insert observer/fake_quantize for DeQuantStub
        if needs_observation(m) and not isinstance(m, DeQuantStub):
            # observer and hook will be gone after we swap the module
            m.add_module('activation_post_process', get_activation_post_process(
                m.qconfig, device, special_act_post_process))
            # Register observer as the first entry in the hook list
            # All post forward hooks are preserved and will be executed after the observer before convert
            _register_activation_post_process_hook(
                m, pre_hook=_activation_is_memoryless(m.qconfig))

    for name, child in module.named_children():
        # TODO remove Dropout special after codebase stable
        if type_before_parametrizations(child) in [nn.Dropout]:
            continue
        elif issubclass(type_before_parametrizations(child), (nnq.FloatFunctional, nnq.QFunctional)):
            if needs_observation(child):
                assert hasattr(child, "activation_post_process"), (
                    f"functional class {type_before_parametrizations(child)} has no pre-defined `activation_post_process`"
                )
                child.activation_post_process = get_activation_post_process(
                    child.qconfig, device)
        elif isinstance(child, _FusedModule):
            # activation_post_process are now added directly to nn.Sequential/_FusedModule
            if needs_observation(child):
                insert_activation_post_process(child)
        elif non_leaf_module_list is not None and type_before_parametrizations(child) in non_leaf_module_list:
            if needs_observation(child):
                insert_activation_post_process(child)
        elif _has_special_act_post_process(child):
            special_act_post_process = _get_special_act_post_process(child)
            insert_activation_post_process(child, special_act_post_process)
        elif needs_observation(child) and type_before_parametrizations(child) in custom_module_class_mapping:
            observed_child = custom_module_class_mapping[type_before_parametrizations(
                child)].from_float(child)
            setattr(module, name, observed_child)
            # TODO: These are the modules that cannot be observed
            #       Once there are more, we should move them to a separate list
            if custom_module_class_mapping[type_before_parametrizations(child)] not in no_observer_set():
                insert_activation_post_process(observed_child)
        else:
            _add_observer_(child, qconfig_propagation_list,
                           non_leaf_module_list, device, custom_module_class_mapping)

    # Insert observers only for leaf nodes, note that this observer is for
    # the output of the module, for input QuantStub will observe them
    if has_no_children_ignoring_parametrizations(module) and not isinstance(module, torch.nn.Sequential) \
       and type_before_parametrizations(module) in qconfig_propagation_list:
        insert_activation_post_process(module)


def _get_unique_devices_(module):
    return {p.device for p in module.parameters()} | \
        {p.device for p in module.buffers()}


def add_quant_dequant(module):
    r"""Wrap the leaf child module in QuantWrapper if it has a valid qconfig
    Note that this function will modify the children of module inplace and it
    can return a new module which wraps the input module as well.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize

    Return:
        Either the inplace modified module with submodules wrapped in
        `QuantWrapper` based on qconfig or a new `QuantWrapper` module which
        wraps the input module, the latter case only happens when the input
        module is a leaf module and we want to quantize it.
    """
    if has_no_children_ignoring_parametrizations(module) and hasattr(module, 'qconfig') and module.qconfig:
        return QuantWrapper(module)

    for name, child in module.named_children():
        module._modules[name] = add_quant_dequant(child)
    return module


def prepare(model, inplace=False, allow_list=None,
            observer_non_leaf_module_list=None,
            prepare_custom_config_dict=None):
    r"""Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        `model`: input model to be modified in-place
        `inplace`: carry out model transformations in-place, the original module is mutated
        `allow_list`: list of quantizable modules
        `observer_non_leaf_module_list`: list of non-leaf modules we want to add observer
        `prepare_custom_config_dict`: customization configuration dictionary for prepare function

    .. code-block:: python

       # Example of prepare_custom_config_dict:
       prepare_custom_config_dict = {
           # user will manually define the corresponding observed
           # module class which has a from_float class method that converts
           # float custom module to observed custom module
           "float_to_observed_custom_module_class": {
               CustomModule: ObservedCustomModule
           }
        }

    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare")
    if prepare_custom_config_dict is None:
        prepare_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = prepare_custom_config_dict.get(
        "float_to_observed_custom_module_class", {})

    if not inplace:
        model = copy.deepcopy(model)

    # TODO: remove allow_list
    qconfig_propagation_list = allow_list
    if allow_list is None:
        qconfig_propagation_list = get_default_qconfig_propagation_list()
    propagate_qconfig_(model, qconfig_dict=None)

    # sanity check common API misusage
    if not any(hasattr(m, 'qconfig') and m.qconfig for m in model.modules()):
        warnings.warn("None of the submodule got qconfig applied. Make sure you "
                      "passed correct configuration through `qconfig_dict` or "
                      "by assigning the `.qconfig` attribute directly on submodules")

    _add_observer_(
        model, qconfig_propagation_list, observer_non_leaf_module_list,
        custom_module_class_mapping=custom_module_class_mapping)
    return model


def _remove_activation_post_process(module):
    # TODO: maybe we should change activation_post_process to _activation_post_process
    # to prevent it from being used by user
    if hasattr(module, 'activation_post_process') and \
       _is_activation_post_process(module.activation_post_process):
        delattr(module, 'activation_post_process')

    # remove activation_post_process pre and post hooks
    def remove_hooks(pre_hook=False):
        hook_map = module._forward_pre_hooks if pre_hook else module._forward_hooks
        observer_hook = _observer_forward_pre_hook if pre_hook else _observer_forward_hook
        handle_ids_to_remove = set()
        for handle_id, hook_fn in hook_map.items():
            if hook_fn is observer_hook:
                handle_ids_to_remove.add(handle_id)
        for handle_id in handle_ids_to_remove:
            hook_map.pop(handle_id)

    remove_hooks(pre_hook=True)
    remove_hooks(pre_hook=False)

# TODO: rename to something more general


def _remove_qconfig(module):
    r"""Clean up the qconfig left in the module so that new qconfig can be
    propagated.

    Args:
        module: module to be cleaned up
    """
    for child in module.children():
        _remove_qconfig(child)

    if hasattr(module, "qconfig"):
        del module.qconfig

    _remove_activation_post_process(module)


def quantize(model, run_fn, run_args, mapping=None, inplace=False):
    r"""Quantize the input float model with post training static quantization.

    First it will prepare the model for calibration, then it calls
    `run_fn` which will run the calibration step, after that we will
    convert the model to a quantized model.

    Args:
        model: input float model
        run_fn: a calibration function for calibrating the prepared model
        run_args: positional arguments for `run_fn`
        inplace: carry out model transformations in-place, the original module is mutated
        mapping: correspondence between original module types and quantized counterparts

    Return:
        Quantized model.
    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize")
    if mapping is None:
        mapping = get_default_static_quant_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    prepare(model, inplace=True)
    run_fn(model, *run_args)
    convert(model, mapping, inplace=True)
    return model


def quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8,
                     mapping=None, inplace=False):
    r"""Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        model: input model
        qconfig_spec: Either:

            - A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfig instances.

            - A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specify the bit-width

        inplace: carry out model transformations in-place, the original module is mutated
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced

    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize_dynamic")
    if qconfig_spec is None:
        if dtype == torch.qint8:
            qconfig_spec = {
                nn.Linear: default_dynamic_qconfig,
                nn.LSTM: default_dynamic_qconfig,
                nn.GRU: default_dynamic_qconfig,
                nn.LSTMCell: default_dynamic_qconfig,
                nn.RNNCell: default_dynamic_qconfig,
                nn.GRUCell: default_dynamic_qconfig,
            }
        elif dtype == torch.float16:
            qconfig_spec = {
                nn.Linear: float16_dynamic_qconfig,
                nn.LSTM: float16_dynamic_qconfig,
                nn.GRU: float16_dynamic_qconfig,
                nn.LSTMCell: float16_dynamic_qconfig,
                nn.RNNCell: float16_dynamic_qconfig,
                nn.GRUCell: float16_dynamic_qconfig,
            }
        elif dtype == torch.quint8:
            qconfig_spec = {
                nn.EmbeddingBag: float_qparams_weight_only_qconfig,
                nn.Embedding: float_qparams_weight_only_qconfig,
            }
        elif dtype == torch.quint4x2:
            qconfig_spec = {
                nn.EmbeddingBag: float_qparams_weight_only_qconfig_4bit,
            }
        else:
            raise ValueError(
                f"Don't know how to quantize with default settings for {dtype}. Provide full qconfig please")
    elif isinstance(qconfig_spec, set):
        if dtype is torch.qint8:
            default_qconfig = default_dynamic_qconfig
        elif dtype is torch.float16:
            default_qconfig = float16_dynamic_qconfig
        elif dtype is torch.quint8:
            default_qconfig = float_qparams_weight_only_qconfig
        elif dtype is torch.quint4x2:
            default_qconfig = float_qparams_weight_only_qconfig_4bit
        else:
            raise RuntimeError(
                'Unknown dtype specified for quantize_dynamic: ', str(dtype))
        qconfig_spec = dict(
            zip(qconfig_spec, itertools.repeat(default_qconfig)))

    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    propagate_qconfig_(model, qconfig_spec)
    convert(model, mapping, inplace=True)
    return model


def prepare_qat(model, mapping=None, inplace=False):
    r"""
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and converts it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    torch._C._log_api_usage_once("quantization_api.quantize.prepare_qat")
    assert model.training, "prepare_qat only works on models in training mode"
    if mapping is None:
        mapping = get_default_qat_module_mappings()

    if not inplace:
        model = copy.deepcopy(model)

    propagate_qconfig_(model, qconfig_dict=None)
    convert(model, mapping=mapping, inplace=True, remove_qconfig=False)
    prepare(model, observer_non_leaf_module_list=set(
        mapping.values()), inplace=True)
    return model


def quantize_qat(model, run_fn, run_args, inplace=False):
    r"""Do quantization aware training and output a quantized model

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
                function that simply runs the prepared model or a training
                loop
        run_args: positional arguments for `run_fn`

    Return:
        Quantized model.
    """
    torch._C._log_api_usage_once("quantization_api.quantize.quantize_qat")
    if not inplace:
        model = copy.deepcopy(model)
    model.train()
    prepare_qat(model, inplace=True)
    run_fn(model, *run_args)
    convert(model, inplace=True)
    return model


def convert(
        module, mapping=None, inplace=False, remove_qconfig=True,
        is_reference=False, convert_custom_config_dict=None, ckpt=None):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class. And remove qconfig at the
    end if remove_qconfig is set to True.

    Args:
        `module`: prepared and calibrated module
        `mapping`: a dictionary that maps from source module type to target
                   module type, can be overwritten to allow swapping user defined
                   Modules
        `inplace`: carry out model transformations in-place, the original module
                   is mutated
        `convert_custom_config_dict`: custom configuration dictionary for convert function

    .. code-block:: python

       # Example of convert_custom_config_dict:
       convert_custom_config_dict = {
           # user will manually define the corresponding quantized
           # module class which has a from_observed class method that converts
           # observed custom module to quantized custom module
           "observed_to_quantized_custom_module_class": {
               ObservedCustomModule: QuantizedCustomModule
           }
       }

    """
    torch._C._log_api_usage_once("quantization_api.quantize.convert")
    if not inplace:
        module = copy.deepcopy(module)
    _convert(
        module, mapping, inplace=True, is_reference=is_reference,
        convert_custom_config_dict=convert_custom_config_dict, ckpt=ckpt)
    if remove_qconfig:
        _remove_qconfig(module)
    return module


def _convert(
        module, mapping=None, inplace=False,
        is_reference=False, convert_custom_config_dict=None, ckpt=None):
    r"""Converts submodules in input module to a different module according to `mapping`
    by calling `from_float` method on the target module class

    Args:
        module: input module
        mapping: a dictionary that maps from source module type to target
                 module type, can be overwritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated
        is_reference: a flag to enable quantized reference module

    """
    if mapping is None:
        mapping = get_default_static_quant_reference_module_mappings() if is_reference \
            else get_default_static_quant_module_mappings()
    if convert_custom_config_dict is None:
        convert_custom_config_dict = get_default_custom_config_dict()
    custom_module_class_mapping = convert_custom_config_dict.get(
        "observed_to_quantized_custom_module_class", {})

    if not inplace:
        module = copy.deepcopy(module)
    reassign = {}
    for name, mod in module.named_children():
        # both fused modules and observed custom modules are
        # swapped as one unit
        if not isinstance(mod, _FusedModule) and \
           type_before_parametrizations(mod) not in custom_module_class_mapping:
            _convert(mod, mapping, True,  # inplace
                     is_reference, convert_custom_config_dict, ckpt=ckpt)
        reassign[name] = swap_module(
            mod, mapping, custom_module_class_mapping, ckpt=ckpt)

    for key, value in reassign.items():
        module._modules[key] = value

    return module


def swap_module(mod, mapping, custom_module_class_mapping, ckpt=None):
    global _NUM
    r"""Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    new_mod = mod
    if hasattr(mod, 'qconfig') and mod.qconfig is not None:
        swapped = False
        if type_before_parametrizations(mod) in custom_module_class_mapping:
            new_mod = custom_module_class_mapping[type_before_parametrizations(
                mod)].from_observed(mod)
            swapped = True
        elif type_before_parametrizations(mod) in mapping:
            qmod = mapping[type_before_parametrizations(mod)]
            if hasattr(qmod, '_IS_REFERENCE') and qmod._IS_REFERENCE:
                assert mod.qconfig is not None
                weight_post_process = mod.qconfig.weight()
                weight_post_process(mod.weight)
                weight_qparams = get_qparam_dict(weight_post_process)
                if 'up_blocks' in mod.module_name and 'conv_shortcut' in mod.module_name:
                    # _NUM = _NUM + 1
                    _split = _SPLIT[_NUM]
                    _NUM = _NUM + 1
                    # num = num + 1
                else:
                    _split = 0
                new_mod = qmod.from_float(mod, weight_qparams, split=_split)
            else:
                if 'up_blocks' in mod.module_name and 'conv_shortcut' in mod.module_name:
                    # _NUM = _NUM + 1
                    _split = _SPLIT[_NUM]
                    _NUM = _NUM + 1
                    # num = num + 1
                    print(f"split at {_split}")
                else:
                    _split = 0
                new_mod = qmod.from_float(mod, split=_split, ckpt=ckpt)
            swapped = True

        if swapped:
            # Preserve module's pre forward hooks. They'll be called on quantized input
            for pre_hook_fn in mod._forward_pre_hooks.values():
                new_mod.register_forward_pre_hook(pre_hook_fn)
            # Preserve module's post forward hooks except _observer_forward_hook
            # After convert they'll work with quantized output
            for hook_fn in mod._forward_hooks.values():
                if hook_fn is not _observer_forward_hook:
                    new_mod.register_forward_hook(hook_fn)

            # respect device affinity when swapping modules
            devices = _get_unique_devices_(mod)
            assert len(devices) <= 1, (
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
            device = next(iter(devices)) if len(devices) > 0 else None
            if device:
                new_mod.to(device)
    return new_mod


def _get_observer_dict(mod, target_dict, prefix=""):
    r"""Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        target_dict: the dictionary used to save all the observers
    """
    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + '.'

    if hasattr(mod, 'activation_post_process'):
        target_dict[get_prefix(
            prefix) + 'activation_post_process'] = mod.activation_post_process
    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_observer_dict(child, target_dict, module_prefix)


# def filter_mod_name_prefix(mod_name):
#     if 'model.' in mod_name:
#         pos = mod_name.index('model.')
#         mod_name = mod_name[pos + 6:]
#     return mod_name


# def register_qconfig_from_input_files(
#     unet,
#     # args,
#     w_bit=8,
#     a_bit=None,
#     bos=True,
#     bos_dict=None
# ):
#     import yaml

#     bw_to_dtype = {
#         8: torch.qint8,
#         4: torch.quint4x2,
#         2: torch.quint4x2,  # !!!TODO: 2 is not supported, treat as 4
#     }

#     # load weight bits
#     # with open(w_config, 'r') as input_file:
#     if w_bit == 8:
#         mod_name_to_weight_width = w8_uniform_config
#     else:
#         raise RuntimeError("we only support int8 quantization")
#     # filter 'model.' from all names
#     mod_name_to_weight_width_copy = {}
#     for mod_name, bit_width in mod_name_to_weight_width.items():
#         new_name = filter_mod_name_prefix(mod_name)
#         mod_name_to_weight_width_copy[new_name] = bit_width
#     mod_name_to_weight_width = mod_name_to_weight_width_copy

#     # add qconfig to all modules whose name are in the yaml
#     mod_name_to_weight_width_copy = mod_name_to_weight_width
#     for name, mod in unet.named_modules():
#         if name in mod_name_to_weight_width:
#             assert not hasattr(mod, 'qconfig')
#             # get the corresponding bit-width of the layer
#             w_bitwidth = mod_name_to_weight_width[name]
#             w_dtype = bw_to_dtype[w_bitwidth]
#             act_preprocess = PlaceholderObserver.with_args(
#                 dtype=torch.float16)  # get the statistic info in the tensor
#             weight_process = PlaceholderObserver.with_args(dtype=w_dtype)
#             mod.qconfig = \
#                 QConfig(activation=act_preprocess, weight=weight_process)

#             # init some parameters for each unquantized module
#             mod.module_name = name  # set module name for each module
#             # record the bit_width of the weight
#             mod.w_bit = mod_name_to_weight_width[name]
#             if 'attn2' in name:
#                 if 'to_k' in name or 'to_v' in name:
#                     mod.bos = bos  # set bos for corss attn layers
#                     mod.bos_pre_computed = bos_dict[name]

#             del mod_name_to_weight_width_copy[name]
#     # check if there is any module not in the unet
#     if len(mod_name_to_weight_width_copy):
#         for name in mod_name_to_weight_width_copy.keys():
#             print(f"{name} not found in UNet!")
#         raise RuntimeError("Not all keys in weight yaml map to a module in "
#                            "UNet.")

#     # load activation bits
#     if a_bit is None:
#         return

#     # with open(a_config, 'r') as input_file:
#     if a_bit == 8:
#         mod_name_to_act_width = a8_mixed_precision_config
#     else:
#         raise RuntimeError("we only support int8 quantization now")
#     # filter 'model.' from all names
#     mod_name_to_act_width_copy = {}
#     for mod_name, bit_width in mod_name_to_act_width.items():
#         new_name = filter_mod_name_prefix(mod_name)
#         mod_name_to_act_width_copy[new_name] = bit_width
#     mod_name_to_act_width = mod_name_to_act_width_copy

#     # add qconfig to all modules whose name are in the yaml
#     mod_name_to_act_width_copy = mod_name_to_act_width
#     for name, mod in unet.named_modules():
#         if name in mod_name_to_act_width:
#             a_bitwidth = mod_name_to_act_width[name]
#             a_dtype = bw_to_dtype[a_bitwidth]
#             act_preprocess = PlaceholderObserver.with_args(dtype=a_dtype)
#             if hasattr(mod, 'qconfig') and mod.qconfig:
#                 assert isinstance(mod.qconfig, QConfig)
#                 mod.qconfig = QConfig(weight=mod.qconfig.weight,
#                                       activation=act_preprocess)
#             else:
#                 weight_process = PlaceholderObserver.with_args(
#                     dtype=torch.float16)
#                 mod.qconfig = QConfig(activation=act_preprocess,
#                                       weight=weight_process)

#             # init some parameters for each unquantized module
#             # record the bit_width of the act
#             mod.a_bit = mod_name_to_act_width[name]

#             del mod_name_to_act_width_copy[name]
#     # check if there is any module not in the unet
#     if len(mod_name_to_act_width_copy):
#         for name in mod_name_to_act_width_copy.keys():
#             print(f"{name} not found in UNet!")
#         raise RuntimeError("Not all keys in act yaml map to a module in "
#                            "UNet.")


# def convert_to_quantized(unet, ckpt):
#     # from quantize import convert
#     convert(unet,
#             mapping={nn.Linear: QuantizedLinear,
#                      nn.Conv2d: QuantizedConv2d,
#                      # QuantStub: Quantizer,
#                      # DeQuantStub: DeQuantizer
#                      },
#             inplace=True,
#             ckpt=ckpt)
#     # print("unet after quantization")
#     # print(unet)

# the code above is for converting the NN model
######################################################################################################


######################################################################################################
# mixdq utils

def quantize_per_tensor_uint4(
    input: torch.Tensor, scale, zero_point,
):

    # reshape the quant parameters for quantizing
    scale = scale.view(-1, *([1] * (len(input.shape) - 1)))
    zero_point = zero_point.view(-1, *([1] * (len(input.shape) - 1)))

    # scale = scale.reshape()
    scale_inv = 1.0 / scale
    int_repr = torch.clamp(torch.round(input * scale_inv) +
                           zero_point, 0, 15).to(torch.uint8)
    if len(input.shape) >= 4:
        assert input.shape[1] % 2 == 0
        return (int_repr[:, ::2, ...] << 4 | int_repr[:, 1::2, ...])
    assert input.shape[-1] % 2 == 0
    return (int_repr[..., ::2] << 4 | int_repr[..., 1::2])


def unpack_uint4(input):
    shape = input.shape
    if len(shape) >= 4:
        packed_dim = 2
        new_shape = (input.shape[0], input.shape[1]*2, *input.shape[2:])
    else:
        packed_dim = -1
        new_shape = (*input.shape[:-1], input.shape[-1]*2)
    first_elements = (input >> 4).to(torch.uint8)
    second_elements = (input & 0b1111).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=packed_dim).view(new_shape)


def dequantize_per_tensor_uint4(
        input, scale, zero_point,
):
    # reshape the quant parameters for dequantizing
    scale = scale.view(-1, *([1] * (len(input.shape) - 1)))
    zero_point = zero_point.view(-1, *([1] * (len(input.shape) - 1)))

    input = unpack_uint4(input)
    return (input.view(torch.uint8).to(torch.float32) - zero_point) * scale


dtype_to_bw = {
    torch.quint8: 8,
    torch.quint4x2: 4,
    torch.quint2x4: 2,
    torch.float16: 16,
}


class QParam(namedtuple("QParam", ["qscheme", "dtype", "scales", "zero_points", "axis"], defaults=[torch.per_tensor_affine, torch.quint8, 1.0, 0.0, 0])):
    @property
    def zp_float(self):
        return self.scales * self.zero_points
    pass


def create_qparams_from_dtype(
    dtype,
    device,
    is_channel_wise=False,
    num_kernels=None,
    ckpt=None,
    module_name=None,
    bit_width=0,
    quant_type=None,
    split=0,
):

    if dtype == torch.float16:
        return None
    elif dtype in [torch.qint8, torch.quint8, torch.quint4x2]:
        if quant_type == 'weight':
            scales, zero_points, scales_0, zero_points_0 = get_quant_para(ckpt,
                                                                          bit_width,
                                                                          module_name,
                                                                          quant_type='weight',
                                                                          split=split,
                                                                          device=device)
        elif quant_type == 'act':
            scales, zero_points, scales_0, zero_points_0 = get_quant_para(ckpt,
                                                                          bit_width,
                                                                          module_name,
                                                                          quant_type='act',
                                                                          split=split,
                                                                          device=device)
    else:
        raise ValueError(f"Unsupported quantize dtype {dtype}")

    if is_channel_wise:
        assert num_kernels is not None
        qparam = QParam(qscheme=torch.per_channel_affine,
                        scales=scales, zero_points=zero_points,
                        dtype=dtype, axis=0)
        if split > 0:
            qparam_0 = QParam(qscheme=torch.per_channel_affine,
                              scales=scales_0, zero_points=zero_points_0,
                              dtype=dtype, axis=0)
        else:
            qparam_0 = None

    else:
        qparam = QParam(qscheme=torch.per_tensor_affine,
                        scales=scales, zero_points=zero_points,
                        dtype=dtype)

        if split > 0:
            qparam_0 = QParam(qscheme=torch.per_tensor_affine,
                              scales=scales_0, zero_points=zero_points_0,
                              dtype=dtype)
        else:
            qparam_0 = None

    return qparam, qparam_0


def quantize_from_qparams(x: torch.Tensor, qparams: QParam):
    if qparams.dtype == torch.quint4x2:
        # TODO: support both per-channel and per-tensor
        # assert qparams.qscheme == torch.per_tensor_affine
        # print(x.shape)
        return quantize_per_tensor_uint4(x, qparams.scales.to(x.device), qparams.zero_points.to(x.device))

    if qparams.qscheme in [torch.per_tensor_affine]:
        scales = qparams.scales
        scales = scales.clone().detach().to(x.device) \
            if isinstance(scales, torch.Tensor) \
            else torch.tensor(scales, dtype=torch.float16, device=x.device)
        zps = qparams.zero_points
        zps = zps.clone().detach().to(x.device) \
            if isinstance(zps, torch.Tensor) \
            else torch.tensor(zps, dtype=torch.float16, device=x.device)

        # Quantize only works on Float Tensor not Half. TODO: custom kernels
        x = x.to(torch.float32)
        x_quant = torch.quantize_per_tensor(x, scales, zps, qparams.dtype)
    elif qparams.qscheme in [torch.per_channel_affine]:
        scales = qparams.scales
        assert isinstance(scales, torch.Tensor)
        scales = scales.clone().detach().to(x.device)
        zps = qparams.zero_points
        assert isinstance(zps, torch.Tensor)
        zps = zps.clone().detach().to(x.device)
        assert qparams.axis < len(x.shape)
        # Quantize only works on Float Tensor not Half TODO: custom kernels
        x = x.to(torch.float32)
        # print(scales.shape)
        # if scales.shape == torch.Size([]):
        #     # torch.quantize_per_channel need the shape of scales and zps to be torch.size([N])
        #     scales = scales.reshape(1)
        #     zps = zps.reshape(1)
        x_quant = torch.quantize_per_channel(x, scales, zps, axis=qparams.axis,
                                             dtype=qparams.dtype)
    else:
        raise ValueError(f"Unknown qscheme {qparams.qscheme}")
    return x_quant


def dequantize_to_float16_linear(x: torch.Tensor, qparams: QParam):
    if x.dtype == torch.float16:
        return x
    if x.dtype in [torch.quint8, torch.qint8]:
        return x.dequantize().to(torch.float32)
    elif x.dtype in [torch.int8]:
        scale = (qparams.scales.view(-1, *
                 ([1] * (len(x.shape) - 1)))).cuda().float()
        zero_points = (qparams.zero_points.view(-1, *
                       ([1] * (len(x.shape) - 1)))).cuda().float()

        x = scale*(x - zero_points)
        return x

    assert x.dtype == torch.uint8  # the current way to support uint4
    return dequantize_per_tensor_uint4(x, qparams.scales.to(x.device), qparams.zero_points.to(x.device)).to(torch.float16)


def dequantize_to_float16(x: torch.Tensor, qparams: QParam):
    if x.dtype == torch.float16:
        return x
    if x.dtype in [torch.quint8, torch.qint8]:
        return x.dequantize().to(torch.float16)
    elif x.dtype in [torch.int8]:
        scale = (qparams.scales.view(-1, *([1] * (len(x.shape) - 1)))).cuda()
        zero_points = (qparams.zero_points.view(-1, *
                       ([1] * (len(x.shape) - 1)))).cuda()

        x = scale*(x - zero_points)
        return x

    assert x.dtype == torch.uint8  # the current way to support uint4
    return dequantize_per_tensor_uint4(x, qparams.scales.to(x.device), qparams.zero_points.to(x.device)).to(torch.float16)


def get_quant_para(ckpt, n_bit, module_name, quant_type, split=0, device=None):

    if split == 0:
        bit_idx = int(math.log2(n_bit)-1)

        if quant_type == 'weight':
            module_name = module_name + '.weight_quantizer'
            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            # sym quantization, zp=0
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx]
            # print(zero_point)

        elif quant_type == 'act':
            module_name = module_name + '.act_quantizer'
            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            # change the data type from uint8 to int8
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx] - 128

        return scales.to(device), zero_point.to(device), None, None

    elif split > 0:
        bit_idx = int(math.log2(n_bit)-1)

        if quant_type == 'weight':
            module_name = module_name + '.weight_quantizer'
            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx]

            module_name = module_name + '_0'
            assert module_name in ckpt.keys()
            scales_0 = ckpt[module_name]['delta_list'][bit_idx]
            zero_point_0 = ckpt[module_name]['zero_point_list'][bit_idx]
            # print(zero_point, zero_point_0)

        elif quant_type == 'act':
            module_name = module_name + '.act_quantizer'

            assert module_name in ckpt.keys()
            scales = ckpt[module_name]['delta_list'][bit_idx]
            zero_point = ckpt[module_name]['zero_point_list'][bit_idx]-128

            module_name = module_name + '_0'
            assert module_name in ckpt.keys()
            scales_0 = ckpt[module_name]['delta_list'][bit_idx]
            zero_point_0 = ckpt[module_name]['zero_point_list'][bit_idx]-128

        return scales.to(device), zero_point.to(device), scales_0.to(device), zero_point_0.to(device)
# mixdq utils
######################################################################################################


######################################################################################################
# mixdq quantized module
class QuantizedConv2d(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, 
                 stride, padding, dilation, groups=1, bias=True,
                 device=None,
                 w_qparams=None, w_qparams_0=None, a_qparams=None, 
                 a_qparams_0 = None, module_name=None, split=0) -> None:
        super().__init__()

        self.module_name = module_name
        self.split = split  # for shortcut layer

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
        self.valid_for_acceleration = (
            w_qparams is not None and \
            a_qparams is not None and \
            w_qparams.dtype in [torch.qint8, torch.quint8] and \
            a_qparams.dtype in [torch.qint8, torch.quint8] and \
            w_qparams.qscheme == torch.per_channel_affine and \
            a_qparams.qscheme == torch.per_tensor_affine and \
            torch.all(w_qparams.zero_points == 0.0).item() and \
            (
                split == 0 or (
                    w_qparams_0 is not None and \
                    a_qparams_0 is not None and \
                    w_qparams_0.dtype in [torch.qint8, torch.quint8] and \
                    a_qparams_0.dtype in [torch.qint8, torch.quint8] and \
                    w_qparams_0.qscheme == torch.per_channel_affine and \
                    a_qparams_0.qscheme == torch.per_tensor_affine and \
                    torch.all(w_qparams_0.zero_points == 0.0).item()
                )
            ) and \
            (
                len(set(self.stride)) == 1 and len(set(self.padding)) == 1 and \
                len(set(self.dilation)) == 1 and self.dilation[0] == 1 and \
                self.groups == 1
            )
        )
        if self.valid_for_acceleration and (
            self.in_channels % 4 != 0 or self.out_channels % 4 != 0):
            logging.warning("Linear layer with in_features = "
                    f"{self.in_channels} and out_features = "
                    f"{self.out_channels} cannot use quantized kernel due to "
                    "misalignment. Falling back to FP kernels")
            self.valid_for_acceleration = False

        if self.valid_for_acceleration:
            self.register_buffer("weight_scales", 
                                 w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", 
                                 w_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales", 
                                 a_qparams.scales.to(device).float())
            self.register_buffer("act_zero_points", 
                                 a_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales_inv", 1 / self.act_scales)
            if self.split != 0:
                self.register_buffer("weight_scales_0", 
                                    w_qparams_0.scales.to(device).float())
                self.register_buffer("weight_zero_points_0", 
                                    w_qparams_0.zero_points.to(device).float())
                self.register_buffer("act_scales_0", 
                                    a_qparams_0.scales.to(device).float())
                self.register_buffer("act_zero_points_0", 
                                    a_qparams_0.zero_points.to(device).float())
                self.register_buffer("act_scales_inv_0", 1 / self.act_scales_0)

    @classmethod
    def from_float(cls, float_mod, split=0, ckpt=None):
        
        assert hasattr(float_mod, 'qconfig') and isinstance(float_mod.qconfig, 
                                                            QConfig)
        weight_process = float_mod.qconfig.weight()
        w_dtype = weight_process.dtype
        num_kernels = float_mod.weight.shape[0]
        device=float_mod.weight.device
        # init the w & a quant parameters
        # split = 0
        # if split == 0:
            # init the quant parameters
        w_qparams, w_qparams_0 = create_qparams_from_dtype(dtype=w_dtype, 
                                                device=device,
                                                is_channel_wise=True,
                                                num_kernels=num_kernels,
                                                ckpt=ckpt,
                                                module_name=float_mod.module_name,
                                                quant_type='weight',
                                                bit_width=float_mod.w_bit,
                                                split=split)


        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype
        
        # if split == 0:
        if hasattr(float_mod, 'a_bit'):
            # if we want to quantized the act
            a_qparams, a_qparams_0 = create_qparams_from_dtype(dtype=act_dtype, 
                                                    device=device,
                                                    is_channel_wise=False,
                                                    num_kernels=num_kernels,
                                                    ckpt=ckpt,
                                                    module_name=float_mod.module_name,
                                                    quant_type='act',
                                                    bit_width=float_mod.a_bit,
                                                    split=split)
        else:
            a_qparams = None
            a_qparams_0 = None
            
        new_mod = cls(float_mod.in_channels,
                      float_mod.out_channels,
                      float_mod.kernel_size,
                      float_mod.stride,
                      float_mod.padding,
                      float_mod.dilation,
                      float_mod.groups,
                      float_mod.bias is not None,
                      device=float_mod.weight.device,

                      w_qparams=w_qparams,
                      w_qparams_0 = w_qparams_0,
                      a_qparams=a_qparams,
                      a_qparams_0 = a_qparams_0,

                      module_name=float_mod.module_name,
                      split = split
                      )

        weight = float_mod.weight.detach()

        if split == 0:
            if new_mod.valid_for_acceleration:
                weight_int = torch.quantize_per_channel(
                    weight.float(), 
                    new_mod.weight_scales, 
                    new_mod.weight_zero_points,
                    axis=w_qparams.axis, 
                    dtype=w_qparams.dtype).int_repr()

                new_mod.register_buffer("weight_int", weight_int)
                # auxiliary structure, used to quickly compute act_zp @ weight
                if float_mod.padding[0] == 0:
                    weight_sum_per_output_channel = \
                        weight_int.float().sum(dim=[1,2,3])
                    new_mod.register_buffer("bias0", 
                        weight_sum_per_output_channel*new_mod.act_zero_points)
                    new_mod.weight_sum_by_input_channels = None
                else:
                    weight_sum_by_input_channels = \
                        weight_int.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels", 
                                            weight_sum_by_input_channels)
                    new_mod.bias0 = None
                new_mod.register_buffer("scale", 
                    new_mod.weight_scales * new_mod.act_scales)
            else:
                new_mod.register_buffer("weight", weight)            
            if float_mod.bias is not None:
                bias = float_mod.bias.detach()
                new_mod.register_buffer("bias", bias)
            else:
                new_mod.bias = None

        # for the weight of the shortcut
        elif split > 0:
            if new_mod.valid_for_acceleration:
                weight_int = torch.quantize_per_channel(
                    weight[:, :split, ...].float(), 
                    new_mod.weight_scales, 
                    new_mod.weight_zero_points,
                    axis=w_qparams.axis, 
                    dtype=w_qparams.dtype).int_repr()

                weight_int_0 = torch.quantize_per_channel(
                    weight[:, split:, ...].float(), 
                    new_mod.weight_scales_0, 
                    new_mod.weight_zero_points_0,
                    axis=w_qparams_0.axis, 
                    dtype=w_qparams_0.dtype).int_repr()

                new_mod.register_buffer("weight_int", weight_int)
                new_mod.register_buffer("weight_int_0", weight_int_0)
                
                # auxiliary structure, used to quickly compute act_zp @ weight
                if float_mod.padding[0] == 0:
                    weight_sum_per_output_channel = \
                        weight_int.float().sum(dim=[1,2,3])
                    new_mod.register_buffer("bias0", 
                        weight_sum_per_output_channel * new_mod.act_zero_points)
                    weight_sum_per_output_channel_0 = \
                        weight_int_0.float().sum(dim=[1,2,3])
                    new_mod.register_buffer("bias0_0", 
                        weight_sum_per_output_channel_0 * new_mod.act_zero_points_0)
                    new_mod.weight_sum_by_input_channels = None
                    new_mod.weight_sum_by_input_channels_0 = None
                else:
                    weight_sum_by_input_channels = \
                        weight_int.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels", 
                                            weight_sum_by_input_channels)
                    weight_sum_by_input_channels_0 = \
                        weight_int_0.float().sum(dim=1, keepdim=True)
                    new_mod.register_buffer("weight_sum_by_input_channels_0", 
                                            weight_sum_by_input_channels_0)
                    new_mod.bias0 = None
                    new_mod.bias0_0 = None
                new_mod.register_buffer("scale", 
                    new_mod.weight_scales * new_mod.act_scales)
                new_mod.register_buffer("scale_0", 
                    new_mod.weight_scales_0 * new_mod.act_scales_0)
            else:
                new_mod.register_buffer("weight", weight)

            if float_mod.bias is not None:
                bias = float_mod.bias.detach()
                new_mod.register_buffer("bias", bias)
            else:
                new_mod.bias = None

        return new_mod
    
    def _get_name(self):
        if self.valid_for_acceleration:
            return "QuantizedConv2dW8A8"
        return "QuantizedConv2dFPFallback"
    
    def forward_fallback(self, x: torch.Tensor):
        weight_recovered = \
            self.weight_int.float() * self.weight_scales[:, None, None, None]
        weight_recovered = weight_recovered.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None

        if self.split == 0:
            return torch.nn.functional.conv2d(x, 
                                            weight_recovered,
                                            bias,
                                            self.stride, 
                                            self.padding,
                                            self.dilation,
                                            self.groups)
        else:
            weight_0_recovered = \
                self.weight_int_0.float() * self.weight_scales_0[:, None, None, None]
            weight_0_recovered = weight_0_recovered.to(x.dtype)
            output = torch.nn.functional.conv2d(x[:, :self.split, :, :], 
                                                weight_recovered,
                                                bias,
                                                self.stride, 
                                                self.padding,
                                                self.dilation,
                                                self.groups)
            output_0 = torch.nn.functional.conv2d(x[:, self.split:, :, :],
                                                weight_0_recovered,
                                                None,
                                                self.stride,
                                                self.padding,
                                                self.dilation,
                                                self.groups)
            output = output + output_0
            return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.valid_for_acceleration:
            return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, 
                            self.dilation, self.groups)

        if not x.dtype == torch.float16:
            return self.forward_fallback(x)

        if self.split == 0:
            x_int = quantize_per_tensor(x, 
                                        self.act_scales_inv, 
                                        self.act_zero_points)
            output = qconv2d(x_int,                             # input_int
                             self.weight_int,                   # weight_int,
                             self.weight_scales,                # weight_scale
                             self.act_scales,                   # input_scale
                             self.act_zero_points,              # input_zp
                             self.scale,                        # scale
                             self.weight_sum_by_input_channels, 
                                                # weight_sum_by_input_channels
                             self.bias0,
                             self.bias,                         # bias
                             self.stride[0],                    # stride
                             self.padding[0],                   # padding
                             )
            return output
        else:
            x_int = quantize_per_tensor(x[:, :self.split, :, :], 
                                        self.act_scales_inv,
                                        self.act_zero_points)
            x_int_0 = quantize_per_tensor(x[:, self.split:, :, :], 
                                          self.act_scales_inv_0,
                                          self.act_zero_points_0)
            output = qconv2d(x_int,                             # input_int
                             self.weight_int,                   # weight_int,
                             self.weight_scales,                # weight_scale
                             self.act_scales,                   # input_scale
                             self.act_zero_points,              # input_zp
                             self.scale,                        # scale
                             self.weight_sum_by_input_channels, 
                                                # weight_sum_by_input_channels
                             self.bias0,
                             self.bias,                         # bias
                             self.stride[0],                    # stride
                             self.padding[0],                   # padding
                             )
            output_0 = qconv2d(x_int_0,                             # input_int
                               self.weight_int_0,                   # weight_int,
                               self.weight_scales_0,                # weight_scale
                               self.act_scales_0,                   # input_scale
                               self.act_zero_points_0,              # input_zp
                               self.scale_0,                        # scale
                               self.weight_sum_by_input_channels_0, 
                                                # weight_sum_by_input_channels
                               self.bias0_0,
                               None,            # bias. Here is none because bias
                                        # need to be applied just once in output
                               self.stride[0],                      # stride
                               self.padding[0],                     # padding
                               )
            output = output + output_0
            return output


class QuantizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
        device=None, w_qparams=None, a_qparams=None, module_name=None) -> None:
        
        super().__init__()
        self.module_name = module_name
        # print(module_name)
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.valid_for_acceleration = (
            w_qparams is not None and \
            a_qparams is not None and \
            w_qparams.dtype in [torch.qint8, torch.quint8] and \
            a_qparams.dtype in [torch.qint8, torch.quint8] and \
            w_qparams.qscheme == torch.per_channel_affine and \
            a_qparams.qscheme == torch.per_tensor_affine and \
            torch.all(w_qparams.zero_points == 0.0).item()
        )
        if self.valid_for_acceleration and (
            self.in_features % 4 != 0 or self.out_features % 4 != 0):
            logging.warning("Linear layer with in_features = "
                    f"{self.in_features} and out_features = "
                    f"{self.out_features} cannot use quantized kernel due to "
                    "misalignment. Falling back to FP kernels")
            self.valid_for_acceleration = False
        
        if self.valid_for_acceleration:
            self.register_buffer("weight_scales", 
                                 w_qparams.scales.to(device).float())
            self.register_buffer("weight_zero_points", 
                                 w_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales", 
                                 a_qparams.scales.to(device).float())
            self.register_buffer("act_zero_points", 
                                 a_qparams.zero_points.to(device).float())
            self.register_buffer("act_scales_inv", 1 / self.act_scales)
        
    
    @classmethod
    def from_float(cls, float_mod, split=0, ckpt=None):
        assert hasattr(float_mod, 'qconfig') and isinstance(float_mod.qconfig, 
                                                            QConfig)
        weight_process = float_mod.qconfig.weight()
        w_dtype = weight_process.dtype
        num_kernels = float_mod.weight.shape[0]
        device=float_mod.weight.device

        w_qparams, w_qparams_0 = create_qparams_from_dtype(dtype=w_dtype, 
                                                device=device,
                                                is_channel_wise=True,
                                                num_kernels=num_kernels,
                                                ckpt=ckpt,
                                                module_name=\
                                                    float_mod.module_name,
                                                quant_type='weight',
                                                bit_width=float_mod.w_bit,
                                                split=split)
                                              

        act_process = float_mod.qconfig.activation()
        act_dtype = act_process.dtype

        if hasattr(float_mod, 'a_bit'):
            a_qparams, a_qparams_0 = create_qparams_from_dtype(dtype=act_dtype, 
                                                    device=device,
                                                    is_channel_wise=False,
                                                    num_kernels=num_kernels,
                                                    ckpt=ckpt,
                                                    module_name=\
                                                        float_mod.module_name,
                                                    quant_type='act',
                                                    bit_width=float_mod.a_bit,
                                                    split=split)
        else:
            a_qparams = None
            a_qparams_0 = None

        new_mod = cls(float_mod.in_features,
                      float_mod.out_features,
                      float_mod.bias is not None,
                      device=float_mod.weight.device,
                      w_qparams=w_qparams,
                      a_qparams=a_qparams,
                      module_name = float_mod.module_name,
                      )

        weight = float_mod.weight.detach()

        if 'attn2' in float_mod.module_name:
            if 'to_k' in float_mod.module_name or \
                'to_v' in float_mod.module_name:
                new_mod.bos = float_mod.bos
                # new_mod.bos_pre_computed =float_mod.bos_pre_computed.to(device)
                new_mod.register_buffer("bos_pre_computed", float_mod.bos_pre_computed)
                # the input of the org_weight is key_first_token
                # new_mod.register_buffer("org_weight", weight)

        if new_mod.valid_for_acceleration:
            weight_int = torch.quantize_per_channel(
                weight.float(), 
                new_mod.weight_scales, 
                new_mod.weight_zero_points,
                axis=w_qparams.axis, 
                dtype=w_qparams.dtype).int_repr()

            new_mod.register_buffer("weight_int", weight_int)

            # auxiliary structure, used to quickly compute act_zp @ weight
            weight_sum_by_input_channels = weight_int.float().sum(dim=1)
            new_mod.register_buffer("weight_sum_by_input_channels", 
                                    weight_sum_by_input_channels)
            new_mod.register_buffer("scale", 
                                    new_mod.weight_scales*new_mod.act_scales)
            new_mod.register_buffer("bias0", 
                weight_sum_by_input_channels * new_mod.act_zero_points)
        else:
            new_mod.register_buffer("weight", weight)
        if float_mod.bias is not None:
            bias = float_mod.bias.detach()
            new_mod.register_buffer("bias", bias)
        else:
            new_mod.bias = None
        return new_mod
    
    def _get_name(self):
        if self.valid_for_acceleration:
            return "QuantizedLinearW8A8"
        return "QuantizedLinearFPFallback"
    
    def forward_fallback(self, x):
        weight_recovered = self.weight_int.float()* self.weight_scales[:, None]
        weight_recovered = weight_recovered.to(x.dtype)
        return F.linear(x, 
                        weight_recovered, 
                        self.bias.to(x.dtype) if self.bias is not None else None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.valid_for_acceleration:
            return F.linear(x, self.weight, self.bias)

        if not x.dtype == torch.float16:
            return self.forward_fallback(x)

        if not hasattr(self, 'bos') or not self.bos:
            x_int = quantize_per_tensor(x, 
                                        self.act_scales_inv, 
                                        self.act_zero_points)
            output = qlinear(
                x_int,                              # input_int
                self.weight_int,                    # weight_int
                self.weight_scales,                 # weight_scale
                self.act_scales,                    # input_scale
                self.act_zero_points,               # input_zp
                self.weight_sum_by_input_channels,  
                                    # weight_sum_by_input_channels
                self.scale,
                self.bias0,
                self.bias                           # bias (None or tensor)
            )
            return output
        else:
            # use bos and quantize the activation
            x_except_first_token = quantize_per_tensor(x[:,1:,:], 
                                                       self.act_scales_inv, 
                                                       self.act_zero_points)
            out_except_first_token = qlinear(x_except_first_token, 
                                             self.weight_int,
                                             self.weight_scales,
                                             self.act_scales,
                                             self.act_zero_points,
                                             self.weight_sum_by_input_channels,
                                             self.scale,
                                             self.bias0,
                                             self.bias)
            out_first_token = self.bos_pre_computed.expand(x.shape[0], -1, -1)
            output =torch.cat([out_first_token, out_except_first_token], dim=1)
            return output

# mixdq quantized module
######################################################################################################


######################################################################################################
# func for sdxl pipeline
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + \
        (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# func for sdxl pipeline
######################################################################################################


######################################################################################################
# quant inference
def filter_mod_name_prefix(mod_name):
    if 'model.' in mod_name:
        pos = mod_name.index('model.')
        mod_name = mod_name[pos + 6:]
    return mod_name


def register_qconfig_from_input_files(
    unet,
    # args,
    w_bit=8,
    a_bit=None,
    bos=True,
    bos_dict=None
):
    import yaml

    bw_to_dtype = {
        8: torch.qint8,
        4: torch.quint4x2,
        2: torch.quint4x2,  # !!!TODO: 2 is not supported, treat as 4
    }

    # load weight bits
    # with open(w_config, 'r') as input_file:
    
    if w_bit == 8:
        mod_name_to_weight_width = w8_uniform_config
    elif w_bit == 4:
        mod_name_to_weight_width = w4_uniform_config
    else:
        raise RuntimeError("we only support int8/int4 quantization")
    # filter 'model.' from all names
    mod_name_to_weight_width_copy = {}
    for mod_name, bit_width in mod_name_to_weight_width.items():
        new_name = filter_mod_name_prefix(mod_name)
        mod_name_to_weight_width_copy[new_name] = bit_width
    mod_name_to_weight_width = mod_name_to_weight_width_copy

    # add qconfig to all modules whose name are in the yaml
    mod_name_to_weight_width_copy = mod_name_to_weight_width
    for name, mod in unet.named_modules():
        if name in mod_name_to_weight_width:
            assert not hasattr(mod, 'qconfig')
            # get the corresponding bit-width of the layer
            w_bitwidth = mod_name_to_weight_width[name]
            w_dtype = bw_to_dtype[w_bitwidth]
            act_preprocess = PlaceholderObserver.with_args(
                dtype=torch.float16)  # get the statistic info in the tensor
            weight_process = PlaceholderObserver.with_args(dtype=w_dtype)
            mod.qconfig = \
                QConfig(activation=act_preprocess, weight=weight_process)

            # init some parameters for each unquantized module
            mod.module_name = name  # set module name for each module
            # record the bit_width of the weight
            mod.w_bit = mod_name_to_weight_width[name]
            if 'attn2' in name:
                if 'to_k' in name or 'to_v' in name:
                    mod.bos = bos  # set bos for corss attn layers
                    mod.bos_pre_computed = bos_dict[name]

            del mod_name_to_weight_width_copy[name]
    # check if there is any module not in the unet
    if len(mod_name_to_weight_width_copy):
        for name in mod_name_to_weight_width_copy.keys():
            print(f"{name} not found in UNet!")
        raise RuntimeError("Not all keys in weight yaml map to a module in "
                           "UNet.")

    # load activation bits
    if a_bit is None:
        return

    # with open(a_config, 'r') as input_file:
    if a_bit == 8:
        mod_name_to_act_width = a8_mixed_precision_config
    else:
        raise RuntimeError("we only support int8 quantization now")
    # filter 'model.' from all names
    mod_name_to_act_width_copy = {}
    for mod_name, bit_width in mod_name_to_act_width.items():
        new_name = filter_mod_name_prefix(mod_name)
        mod_name_to_act_width_copy[new_name] = bit_width
    mod_name_to_act_width = mod_name_to_act_width_copy

    # add qconfig to all modules whose name are in the yaml
    mod_name_to_act_width_copy = mod_name_to_act_width
    for name, mod in unet.named_modules():
        if name in mod_name_to_act_width:
            a_bitwidth = mod_name_to_act_width[name]
            a_dtype = bw_to_dtype[a_bitwidth]
            act_preprocess = PlaceholderObserver.with_args(dtype=a_dtype)
            if hasattr(mod, 'qconfig') and mod.qconfig:
                assert isinstance(mod.qconfig, QConfig)
                mod.qconfig = QConfig(weight=mod.qconfig.weight,
                                      activation=act_preprocess)
            else:
                weight_process = PlaceholderObserver.with_args(
                    dtype=torch.float16)
                mod.qconfig = QConfig(activation=act_preprocess,
                                      weight=weight_process)

            # init some parameters for each unquantized module
            # record the bit_width of the act
            mod.a_bit = mod_name_to_act_width[name]

            del mod_name_to_act_width_copy[name]
    # check if there is any module not in the unet
    if len(mod_name_to_act_width_copy):
        for name in mod_name_to_act_width_copy.keys():
            print(f"{name} not found in UNet!")
        raise RuntimeError("Not all keys in act yaml map to a module in "
                           "UNet.")


def convert_to_quantized(unet, ckpt):
    # from quantize import convert
    convert(unet,
            mapping={nn.Linear: QuantizedLinear,
                     nn.Conv2d: QuantizedConv2d,
                     },
            inplace=True,
            ckpt=ckpt)

def quantize_unet(
        pipe,
        w_bit=None,
        a_bit=None,
        bos=False,
        # cuda_graph_only=True,
        # run_pipeline=True,
        # compile=False,
    ):
        r"""
        This function helps quantize the UNet in the SDXL Pipeline
        Now we only support quantization with the setting W8A8

        Args:
            w_bit: (`str`):
                the bit width of weight
            a_bit: (`str`):
                the bit width of activation
            bos: (`bool`):
                if to use bos technique
            cuda_graph_only: (`bool`):
                if to use cuda_graph
            run_pipeline: (`bool`):
                run the full pipeline or just the unet
        """
        # load the quant para and the pre-computed bos tensor
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Stein-Fun/mixdq_test",
            filename="bos_pre_computed.pt",
            revision="version_0",
        )
        bos_dict = torch.load(path, map_location='cpu')

        path = hf_hub_download(
            repo_id="Stein-Fun/mixdq_test",
            filename="quant_para_wsym_fp16.pt",
            revision="version_0",
        )
        ckpt = torch.load(path, map_location='cpu')
        register_qconfig_from_input_files(
            pipe.unet,
            # args,
            w_bit=w_bit,
            a_bit=a_bit,
            bos=bos,
            bos_dict=bos_dict
        )
        convert_to_quantized(pipe.unet, ckpt)
        return pipe

# def compile_opt(pipeline, args):
#     if args.run_pipeline:
#         pipeline = pipeline.to('cuda')
#     else:
#         pipeline.unet = pipeline.unet.to("cuda")

#     if args.compile_tool == 'pt2':
#         pipeline.unet = torch.compile(pipeline.unet)
#     elif args.compile_tool == 'onediff':
#         from onediff.infer_compiler import oneflow_compile
#         pipeline.unet = oneflow_compile(pipeline.unet)
#     elif args.compile_tool == 'sfast':
#         # apply stable-fast
#         from sfast.compilers.diffusion_pipeline_compiler import compile_unet
#         from sfast.compilers.diffusion_pipeline_compiler import CompilationConfig

#         sfast_config = CompilationConfig.Default()
#         sfast_config.enable_triton = True
#         sfast_config.enable_cuda_graph = False
#         sfast_config.enable_jit = True

#         pipeline.unet = compile_unet(pipeline.unet, sfast_config)
#     else:
#         print(f"Unknown compile tool {args.compile_tool}")
#     return pipeline


def cuda_graph_opt(unet):

    def hash_arg(arg):
        if isinstance(arg, torch.Tensor):
            arg_device = arg.device
            arg_device_type = arg_device.type
            return (arg_device_type, arg_device.index, arg.dtype, arg.shape,
                    arg.item()
                    if arg_device_type == 'cpu' and arg.numel() == 1 else None)
        if isinstance(arg, (str, int, float, bytes, bool)):
            return arg
        if isinstance(arg, (tuple, list)):
            return tuple(map(hash_arg, arg))
        if isinstance(arg, dict):
            return tuple(
                sorted(((hash_arg(k), hash_arg(v)) for k, v in arg.items()),
                    key=lambda x: x[0]))
        return type(arg)
    
    def copy_args(arg):
        if isinstance(arg, tuple):
            return tuple(map(copy_args, arg))
        if isinstance(arg, list):
            return list(map(copy_args, arg))
        if isinstance(arg, dict):
            d_ = dict()
            for k, v in arg.items():
                d_[k] = copy_args(v)
            return d_
        if isinstance(arg, (str, int, float, bytes, bool)):
            return arg
        if isinstance(arg, torch.Tensor):
            return arg.detach().clone()
        if arg is None:
            return None
        raise ValueError(f"Unknown argument type {arg}")
    
    def copy_args_to_dest(dest_arg, src_arg):
        if isinstance(src_arg, (tuple, list)):
            for i, x in enumerate(src_arg):
                copy_args_to_dest(dest_arg[i], x)
        if isinstance(src_arg, dict):
            for k, v in src_arg.items():
                copy_args_to_dest(dest_arg[k], v)
        if isinstance(src_arg, (str, int, float, bytes, bool)) \
            or src_arg is None:
            pass # should be the same with dest_arg
        if isinstance(src_arg, torch.Tensor):
            dest_arg.copy_(src_arg)
    
    def create_forward_with_cuda_graph(net):
        lock = threading.Lock()
        cached_cuda_graphs = {}

        wrapped = net.forward

        @functools.wraps(wrapped)
        def forward_with_cuda_graph(*args, **kwargs):
            key = (hash_arg(args), hash_arg(kwargs))
            if not (key in cached_cuda_graphs):
                with lock:
                    if not (key in cached_cuda_graphs):
                        args_, kwargs_ = copy_args((args, kwargs))

                        s = torch.cuda.Stream()
                        s.wait_stream(torch.cuda.current_stream())

                        with torch.no_grad():
                            with torch.cuda.stream(s):
                                for _ in range(3):
                                    static_output = wrapped(*args_, **kwargs_)

                        g = torch.cuda.CUDAGraph()
                        with torch.no_grad():
                            with torch.cuda.graph(g):
                                static_output = wrapped(*args_, **kwargs_)

                        cached_cuda_graphs[key] = (
                            (args_, kwargs_),
                            g,
                            static_output
                        )
            static_inputs, graph, static_output = cached_cuda_graphs[key]
            args_, kwargs_ = static_inputs

            copy_args_to_dest((args_, kwargs_), (args, kwargs))
            graph.replay()
            return static_output

        forward_with_cuda_graph.__self__ = net
        forward_with_cuda_graph._cached = cached_cuda_graphs
        return forward_with_cuda_graph

    unet.forward = create_forward_with_cuda_graph(unet)

    # # change static parts of unet to cuda graph
    # for mod in unet.down_blocks:
    #     mod.forward = create_forward_with_cuda_graph(mod)
    # if unet.mid_block is not None:
    #     unet.mid_block.forward = create_forward_with_cuda_graph(unet.mid_block)
    # for mod in unet.up_blocks:
    #     mod.forward = create_forward_with_cuda_graph(mod)
    return unet


def make_memory_friendly(bytes):

    MBs = bytes / (1024*1024)

    B = bytes % 1024
    bytes = bytes // 1024
    kB = bytes % 1024
    bytes = bytes // 1024
    MB = bytes % 1024
    GB = bytes // 1024

    return f"{GB} G {MB} M {B} {kB} K {B} Bytes ({MBs} MBs)"


class MixDQ_SDXLTurbo_Pipeline_W8A8(
    DiffusionPipeline,
    FromSingleFileMixin,
    StableDiffusionXLLoraLoaderMixin,
    TextualInversionLoaderMixin,
    IPAdapterMixin,
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion XL.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *LoRA*: [`loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromSingleFileMixin.from_single_file`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.StableDiffusionXLLoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion XL uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([` CLIPTextModelWithProjection`]):
            Second frozen text-encoder. Stable Diffusion XL uses the text and pool portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        force_zeros_for_empty_prompt (`bool`, *optional*, defaults to `"True"`):
            Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
            `stabilityai/stable-diffusion-xl-base-1-0`.
        add_watermarker (`bool`, *optional*):
            Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
            watermark output images. If not defined, it will default to True if the package is installed, otherwise no
            watermarker will be used.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->unet->vae"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
        "feature_extractor",
    ]
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "add_text_embeds",
        "add_time_ids",
        "negative_pooled_prompt_embeds",
        "negative_add_time_ids",
    ]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__()
        print("HERE!")
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=unet,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.register_to_config(
            force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor)

        self.default_sample_size = self.unet.config.sample_size

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_tiling
    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_tiling
    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionXLLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(
                        self.text_encoder, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder, lora_scale)

            if self.text_encoder_2 is not None:
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(
                        self.text_encoder_2, lora_scale)
                else:
                    scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # Define tokenizers and text encoders
        tokenizers = [self.tokenizer, self.tokenizer_2] if self.tokenizer is not None else [
            self.tokenizer_2]
        text_encoders = (
            [self.text_encoder, self.text_encoder_2] if self.text_encoder is not None else [
                self.text_encoder_2]
        )

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # textual inversion: procecss multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt_2]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, tokenizer)

                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids
                untruncated_ids = tokenizer(
                    prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1: -1])
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {tokenizer.model_max_length} tokens: {removed_text}"
                    )

                prompt_embeds = text_encoder(
                    text_input_ids.to(device), output_hidden_states=True)

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                if clip_skip is None:
                    prompt_embeds = prompt_embeds.hidden_states[-2]
                else:
                    # "2" because SDXL always indexes from the penultimate layer.
                    prompt_embeds = prompt_embeds.hidden_states[-(
                        clip_skip + 2)]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # get unconditional embeddings for classifier free guidance
        zero_out_negative_prompt = negative_prompt is None and self.config.force_zeros_for_empty_prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None and zero_out_negative_prompt:
            negative_prompt_embeds = torch.zeros_like(prompt_embeds)
            negative_pooled_prompt_embeds = torch.zeros_like(
                pooled_prompt_embeds)
        elif do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt

            # normalize str to list
            negative_prompt = batch_size * \
                [negative_prompt] if isinstance(
                    negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size *
                [negative_prompt_2] if isinstance(
                    negative_prompt_2, str) else negative_prompt_2
            )

            uncond_tokens: List[str]
            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = [negative_prompt, negative_prompt_2]

            negative_prompt_embeds_list = []
            for negative_prompt, tokenizer, text_encoder in zip(uncond_tokens, tokenizers, text_encoders):
                if isinstance(self, TextualInversionLoaderMixin):
                    negative_prompt = self.maybe_convert_prompt(
                        negative_prompt, tokenizer)

                max_length = prompt_embeds.shape[1]
                uncond_input = tokenizer(
                    negative_prompt,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                negative_prompt_embeds = text_encoder(
                    uncond_input.input_ids.to(device),
                    output_hidden_states=True,
                )
                # We are only ALWAYS interested in the pooled output of the final text encoder
                negative_pooled_prompt_embeds = negative_prompt_embeds[0]
                negative_prompt_embeds = negative_prompt_embeds.hidden_states[-2]

                negative_prompt_embeds_list.append(negative_prompt_embeds)

            negative_prompt_embeds = torch.concat(
                negative_prompt_embeds_list, dim=-1)

        if self.text_encoder_2 is not None:
            prompt_embeds = prompt_embeds.to(
                dtype=self.text_encoder_2.dtype, device=device)
        else:
            prompt_embeds = prompt_embeds.to(
                dtype=self.unet.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            if self.text_encoder_2 is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=self.text_encoder_2.dtype, device=device)
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=self.unet.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt, -1
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )

        if self.text_encoder is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, StableDiffusionXLLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_image
    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(
                image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(
            num_images_per_prompt, dim=0)

        uncond_image_embeds = torch.zeros_like(image_embeds)
        return image_embeds, uncond_image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(
            self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        prompt_2,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(
                f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height //
                 self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def _get_add_time_ids(
        self, original_size, crops_coords_top_left, target_size, dtype, text_encoder_projection_dim=None
    ):
        add_time_ids = list(
            original_size + crops_coords_top_left + target_size)

        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim *
            len(add_time_ids) + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.StableDiffusionUpscalePipeline.upcast_vae
    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_freeu
    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float):
        r"""Enables the FreeU mechanism as in https://arxiv.org/abs/2309.11497.

        The suffixes after the scaling factors represent the stages where they are being applied.

        Please refer to the [official repository](https://github.com/ChenyangSi/FreeU) for combinations of the values
        that are known to work well for different pipelines such as Stable Diffusion v1, v2, and Stable Diffusion XL.

        Args:
            s1 (`float`):
                Scaling factor for stage 1 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            s2 (`float`):
                Scaling factor for stage 2 to attenuate the contributions of the skip features. This is done to
                mitigate "oversmoothing effect" in the enhanced denoising process.
            b1 (`float`): Scaling factor for stage 1 to amplify the contributions of backbone features.
            b2 (`float`): Scaling factor for stage 2 to amplify the contributions of backbone features.
        """
        if not hasattr(self, "unet"):
            raise ValueError("The pipeline must have `unet` for using FreeU.")
        self.unet.enable_freeu(s1=s1, s2=s2, b1=b1, b2=b2)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_freeu
    def disable_freeu(self):
        """Disables the FreeU mechanism if enabled."""
        self.unet.disable_freeu()

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def quantize_unet(
        self,
        w_bit=None,
        a_bit=None,
        bos=False,
        # cuda_graph_only=True,
        # run_pipeline=True,
        # compile=False,
    ):
        r"""
        This function helps quantize the UNet in the SDXL Pipeline
        Now we only support quantization with the setting W8A8

        Args:
            w_bit: (`str`):
                the bit width of weight
            a_bit: (`str`):
                the bit width of activation
            bos: (`bool`):
                if to use bos technique
            cuda_graph_only: (`bool`):
                if to use cuda_graph
            run_pipeline: (`bool`):
                run the full pipeline or just the unet
        """
        # load the quant para and the pre-computed bos tensor
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id="Stein-Fun/mixdq_test",
            filename="bos_pre_computed.pt",
            revision="version_0",
        )
        bos_dict = torch.load(path, map_location='cpu')

        path = hf_hub_download(
            repo_id="Stein-Fun/mixdq_test",
            filename="quant_para_wsym_fp16.pt",
            revision="version_0",
        )
        ckpt = torch.load(path, map_location='cpu')
        register_qconfig_from_input_files(
            self.unet,
            # args,
            w_bit=w_bit,
            a_bit=a_bit,
            bos=bos,
            bos_dict=bos_dict
        )
        convert_to_quantized(self.unet, ckpt)
    
    def set_cuda_graph(
        self,
        run_pipeline = True,
        compile = False,
        ):

        if run_pipeline:
            self.to('cuda')
        else:
            self.unet.to("cuda")

        batch_size = 1
        prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

        # start = time.time()
        if run_pipeline:
            def run_once():
                latents = self(prompt=[prompt]*batch_size, 
                                guidance_scale=0.0, 
                                num_inference_steps=1).images[0]      
                return latents
        else:
            sample_shape = (
                batch_size * 1, 
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )

            encoder_embedding_shape = (
                batch_size * 1,
                77, # just an example,
                2048,
            )

            device=torch.device('cuda')
            example_sample = torch.rand(*sample_shape, device=device, 
                                        dtype=torch.float16)
            example_embedding = torch.rand(*encoder_embedding_shape, 
                                        device=device, dtype=torch.float16)
            timestep = torch.tensor(999., device=device)
            text_embeds = torch.rand(batch_size, 1280, device=device, 
                                    dtype=torch.float16)
            time_ids = torch.tensor([[512.,512.,0.,0.,512.,512.]], dtype=torch.float16,
                                    device=device)
            time_ids = torch.concat([time_ids] * batch_size)

            def run_once():
                with torch.no_grad():
                    latents = self.unet(sample=example_sample,
                                            timestep=timestep,
                                            encoder_hidden_states=example_embedding,
                                            added_cond_kwargs={
                                                'time_ids': time_ids,
                                                'text_embeds': text_embeds
                                            },
                                            return_dict=False)[0]
                    return latents

        # if cuda_graph_only:
        print("apply the cuda graph!")
        if compile:
            logging.warning("--compile and --cuda_graph_only should not be used"
                            " together, cuda_graph_only is ignored.")
        else:
            self.unet = cuda_graph_opt(self.unet)
            print("start to warm up!")
            print("run the pipeline") if run_pipeline else print("run the unet")
            latents = run_once()
            # if run_pipeline and output_type=='pil':
            #     latents.save('result_01.png')
            print("finish warming up!")


    def run_for_test(
        self,
        device,
        prompt: str = "A black and white photo of an older man skiing.",
        batch_size: int = 1,
        output_type: str = "pil",
        run_pipeline: bool = False,
        memory_snapshot_name: str=None,
        profile: bool=False,
        profile_tool: str="torch_profiler",
        path: str = "result.png"
    ):
        r"""
        run for test the memory reduction after quantization on GPUs

        Args:
            device: (`torch.device`):
                torch device, 'CUDA' is supported only
            prompt: (`str` or `List[str]`, *optional*):
                prompt to be encoded
            batch_size: (`int`):
                the batch size of inputs
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            batch_size: (`int`):
                the batch size of inputs 
            run_pipeline: (`bool`):
                if to run the whole pipeline or just run the UNet
            profile: (`bool`):
                if to profile to test the lantency of the inference
            profile_tool: (`str`):
                choose the profiler: "torch_profiler" or "nsys"
            path: (`str`):
                the path to save the output image
        """
        if run_pipeline:
            self.to('cuda')
        else:
            self.unet.to("cuda")

        model_memory = torch.cuda.memory_allocated()
        print("Static (weights) memory usage:", make_memory_friendly(model_memory))


        # start = time.time()
        if run_pipeline:
            def run_once():
                latents = self(prompt=[prompt]*batch_size, 
                                guidance_scale=0.0, 
                                num_inference_steps=1, 
                                output_type=output_type).images[0]      
                return latents
        else:
            sample_shape = (
                batch_size * 1, 
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )

            encoder_embedding_shape = (
                batch_size * 1,
                77, # just an example,
                2048,
            )

            device=torch.device('cuda')
            example_sample = torch.rand(*sample_shape, device=device, 
                                        dtype=torch.float16)
            example_embedding = torch.rand(*encoder_embedding_shape, 
                                        device=device, dtype=torch.float16)
            timestep = torch.tensor(999., device=device)
            text_embeds = torch.rand(batch_size, 1280, device=device, 
                                    dtype=torch.float16)
            time_ids = torch.tensor([[512.,512.,0.,0.,512.,512.]], dtype=torch.float16,
                                    device=device)
            time_ids = torch.concat([time_ids] * batch_size)

            def run_once():
                with torch.no_grad():
                    latents = self.unet(sample=example_sample,
                                            timestep=timestep,
                                            encoder_hidden_states=example_embedding,
                                            added_cond_kwargs={
                                                'time_ids': time_ids,
                                                'text_embeds': text_embeds
                                            },
                                            return_dict=False)[0]
                    return latents
        
        # latents = run_once()
        # latents = run_once()

        # if run_pipeline:
        #     latents.save('result_00.png')

        # if cuda_graph_only:
        #     if compile:
        #         logging.warning("--compile and --cuda_graph_only should not be used"
        #                         " together, cuda_graph_only is ignored.")
        #     else:
        #         self.unet = cuda_graph_opt(self.unet)
        #         print("start to warm up!")
        #         latents = run_once()
        #         # if run_pipeline and output_type=='pil':
        #         #     latents.save('result_01.png')
        #         print("finish warming up!")
                
        #         print("quant inference!")
        #         latents = run_once()
        #         if run_pipeline and output_type=='pil':
        #             latents.save(path)
        # else:
        latents = run_once()
        if run_pipeline and output_type=='pil':
            latents.save(path)

        peak_memory = torch.cuda.max_memory_allocated()
        print("Dynamic (acts) memory usage:", 
            make_memory_friendly(peak_memory - model_memory))
        print("Peak (total) memory usage:", make_memory_friendly(peak_memory))
        
        if memory_snapshot_name is not None:
            torch.cuda.memory._dump_snapshot(memory_snapshot_name)
        
        if profile:
            if profile_tool == "nsys":
                torch.cuda.cudart().cudaProfilerStart()
                for iter in range(3):
                    torch.cuda.nvtx.range_push(f"iter_{iter}")
                    run_once()
                    torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
            elif profile_tool == "torch_profiler":
                with torch.profiler.profile(
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/sdxl'),
                    with_stack=True
                ):
                    for _ in range(3):
                        run_once()
            else:
                print(f"Unknown profile_tool {profile_tool}, use nsys or torch_profiler")

        # if output_type=="pil":
        #     image = latents

        #     image.save("result.png")
        return latents


    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1 and self.unet.config.time_cond_proj_dim is None

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def denoising_end(self):
        return self._denoising_end

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator,
                                  List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[
            int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in both text-encoders
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
                Anything below 512 pixels won't work well for
                [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
                and checkpoints that are not specifically fine-tuned on low resolutions.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            denoising_end (`float`, *optional*):
                When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
                completed before it is intentionally prematurely terminated. As a result, the returned sample will
                still retain a substantial amount of noise as determined by the discrete timesteps selected by the
                scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
                "Mixture of Denoisers" multi-pipeline setup, as elaborated in [**Refining the Image
                Output**](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#refining-the-image-output)
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor proposed by [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf) `guidance_scale` is defined as `φ` in equation 16. of
                [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf).
                Guidance rescale factor should fix overexposure when using zero terminal SNR.
            original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
                `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
                explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
                `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
                `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                For most cases, `target_size` should be set to the desired height and width of the generated image. If
                not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
                section 2.2 of [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952).
            negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a specific image resolution. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
                To negatively condition the generation process based on a specific crop coordinates. Part of SDXL's
                micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
                To negatively condition the generation process based on a target image resolution. It should be as same
                as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
                [https://huggingface.co/papers/2307.01952](https://huggingface.co/papers/2307.01952). For more
                information, refer to this issue thread: https://github.com/huggingface/diffusers/issues/4208.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get(
                "scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat(
                [negative_add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1)

        if ip_adapter_image is not None:
            image_embeds, negative_image_embeds = self.encode_image(
                ip_adapter_image, device, num_images_per_prompt)
            if self.do_classifier_free_guidance:
                image_embeds = torch.cat([negative_image_embeds, image_embeds])
                image_embeds = image_embeds.to(device)

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(
                self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat(
                    [latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop(
                        "prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop(
                        "add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop(
                        "add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop(
                        "negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(
                    next(iter(self.vae.post_quant_conv.parameters())).dtype)

            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(
                image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


# for quant inference
######################################################################################################


######################################################################################################
# mixed precision config
a8_mixed_precision_config = \
    {
        'add_embedding.linear_1': 8, 'add_embedding.linear_2': 8, 'down_blocks.0.downsamplers.0.conv': 8, 'down_blocks.0.resnets.0.conv1': 8, 'down_blocks.0.resnets.0.time_emb_proj': 8, 'down_blocks.0.resnets.1.conv1': 8, 'down_blocks.0.resnets.1.conv2': 8, 'down_blocks.0.resnets.1.time_emb_proj': 8, 'down_blocks.1.attentions.0.proj_in': 8, 'down_blocks.1.attentions.0.proj_out': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.ff.net.2': 8,
        'down_blocks.1.attentions.1.proj_in': 8, 'down_blocks.1.attentions.1.proj_out': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.ff.net.2': 8, 'down_blocks.1.downsamplers.0.conv': 8, 'down_blocks.1.resnets.0.conv1': 8, 'down_blocks.1.resnets.0.conv2': 8, 'down_blocks.1.resnets.0.conv_shortcut': 8, 'down_blocks.1.resnets.0.time_emb_proj': 8, 'down_blocks.1.resnets.1.conv1': 8, 'down_blocks.1.resnets.1.conv2': 8, 'down_blocks.1.resnets.1.time_emb_proj': 8,
        'down_blocks.2.attentions.0.proj_in': 8, 'down_blocks.2.attentions.0.proj_out': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_v': 8,
        'down_blocks.2.attentions.0.transformer_blocks.2.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_v': 8,
        'down_blocks.2.attentions.0.transformer_blocks.5.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_v': 8,
        'down_blocks.2.attentions.0.transformer_blocks.8.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.ff.net.2': 8, 'down_blocks.2.attentions.1.proj_in': 8, 'down_blocks.2.attentions.1.proj_out': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_out.0': 8,
        'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_out.0': 8,
        'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_v': 8,
        'down_blocks.2.attentions.1.transformer_blocks.7.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.ff.net.2': 8, 'down_blocks.2.resnets.0.conv1': 8, 'down_blocks.2.resnets.0.conv2': 8, 'down_blocks.2.resnets.0.conv_shortcut': 8, 'down_blocks.2.resnets.0.time_emb_proj': 8, 'down_blocks.2.resnets.1.conv1': 8, 'down_blocks.2.resnets.1.conv2': 8, 'down_blocks.2.resnets.1.time_emb_proj': 8, 'mid_block.attentions.0.proj_in': 8, 'mid_block.attentions.0.proj_out': 8, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_k': 8,
        'mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.0.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.1.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.2.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.2.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_k': 8,
        'mid_block.attentions.0.transformer_blocks.3.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.3.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.3.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.4.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.4.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.5.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.5.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_k': 8,
        'mid_block.attentions.0.transformer_blocks.6.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.6.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.6.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.7.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.7.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.8.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.8.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_k': 8,
        'mid_block.attentions.0.transformer_blocks.9.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.9.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.9.ff.net.2': 8, 'mid_block.resnets.0.conv1': 8, 'mid_block.resnets.0.conv2': 8, 'mid_block.resnets.0.time_emb_proj': 8, 'mid_block.resnets.1.conv1': 8, 'mid_block.resnets.1.conv2': 8, 'mid_block.resnets.1.time_emb_proj': 8, 'time_embedding.linear_1': 8, 'time_embedding.linear_2': 8, 'up_blocks.0.attentions.0.proj_in': 8, 'up_blocks.0.attentions.0.proj_out': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_out.0': 8,
        'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_out.0': 8,
        'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_out.0': 8,
        'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.ff.net.2': 8, 'up_blocks.0.attentions.1.proj_in': 8, 'up_blocks.0.attentions.1.proj_out': 8,
        'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.ff.net.2': 8,
        'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.ff.net.2': 8,
        'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.ff.net.2': 8,
        'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.ff.net.2': 8, 'up_blocks.0.attentions.2.proj_in': 8, 'up_blocks.0.attentions.2.proj_out': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_v': 8,
        'up_blocks.0.attentions.2.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_v': 8,
        'up_blocks.0.attentions.2.transformer_blocks.4.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_v': 8,
        'up_blocks.0.attentions.2.transformer_blocks.7.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.ff.net.2': 8, 'up_blocks.0.resnets.0.conv1': 8, 'up_blocks.0.resnets.0.conv2': 8, 'up_blocks.0.resnets.0.conv_shortcut': 8, 'up_blocks.0.resnets.0.time_emb_proj': 8, 'up_blocks.0.resnets.1.conv1': 8, 'up_blocks.0.resnets.1.conv2': 8, 'up_blocks.0.resnets.1.conv_shortcut': 8, 'up_blocks.0.resnets.1.time_emb_proj': 8,
        'up_blocks.0.resnets.2.conv1': 8, 'up_blocks.0.resnets.2.conv2': 8, 'up_blocks.0.resnets.2.conv_shortcut': 8, 'up_blocks.0.resnets.2.time_emb_proj': 8, 'up_blocks.0.upsamplers.0.conv': 8, 'up_blocks.1.attentions.0.proj_in': 8, 'up_blocks.1.attentions.0.proj_out': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.ff.net.2': 8, 'up_blocks.1.attentions.1.proj_in': 8, 'up_blocks.1.attentions.1.proj_out': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k': 8,
        'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.ff.net.2': 8, 'up_blocks.1.attentions.2.proj_in': 8, 'up_blocks.1.attentions.2.proj_out': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj': 8,
        'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.ff.net.2': 8, 'up_blocks.1.resnets.0.conv1': 8, 'up_blocks.1.resnets.0.conv2': 8, 'up_blocks.1.resnets.0.conv_shortcut': 8, 'up_blocks.1.resnets.0.time_emb_proj': 8, 'up_blocks.1.resnets.1.conv1': 8, 'up_blocks.1.resnets.1.conv2': 8, 'up_blocks.1.resnets.1.conv_shortcut': 8, 'up_blocks.1.resnets.1.time_emb_proj': 8, 'up_blocks.1.resnets.2.conv1': 8, 'up_blocks.1.resnets.2.conv2': 8, 'up_blocks.1.resnets.2.conv_shortcut': 8, 'up_blocks.1.resnets.2.time_emb_proj': 8, 'up_blocks.1.upsamplers.0.conv': 8, 'up_blocks.2.resnets.0.conv1': 8, 'up_blocks.2.resnets.0.conv2': 8, 'up_blocks.2.resnets.0.conv_shortcut': 8, 'up_blocks.2.resnets.0.time_emb_proj': 8, 'up_blocks.2.resnets.1.conv1': 8, 'up_blocks.2.resnets.1.conv2': 8,
        'up_blocks.2.resnets.1.conv_shortcut': 8, 'up_blocks.2.resnets.1.time_emb_proj': 8, 'up_blocks.2.resnets.2.conv1': 8, 'up_blocks.2.resnets.2.conv2': 8, 'up_blocks.2.resnets.2.time_emb_proj': 8,
    }

w8_uniform_config = \
    {
        'conv_in': 8, 'time_embedding.linear_1': 8, 'time_embedding.linear_2': 8, 'add_embedding.linear_1': 8, 'add_embedding.linear_2': 8, 'down_blocks.0.resnets.0.conv1': 8, 'down_blocks.0.resnets.0.time_emb_proj': 8, 'down_blocks.0.resnets.0.conv2': 8, 'down_blocks.0.resnets.1.conv1': 8, 'down_blocks.0.resnets.1.time_emb_proj': 8, 'down_blocks.0.resnets.1.conv2': 8, 'down_blocks.0.downsamplers.0.conv': 8, 'down_blocks.1.attentions.0.proj_in': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v': 8,
        'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.1.attentions.0.transformer_blocks.1.ff.net.2': 8, 'down_blocks.1.attentions.0.proj_out': 8, 'down_blocks.1.attentions.1.proj_in': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_out.0': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.1.attentions.1.transformer_blocks.1.ff.net.2': 8, 'down_blocks.1.attentions.1.proj_out': 8, 'down_blocks.1.resnets.0.conv1': 8, 'down_blocks.1.resnets.0.time_emb_proj': 8, 'down_blocks.1.resnets.0.conv2': 8, 'down_blocks.1.resnets.0.conv_shortcut': 8,
        'down_blocks.1.resnets.1.conv1': 8, 'down_blocks.1.resnets.1.time_emb_proj': 8, 'down_blocks.1.resnets.1.conv2': 8, 'down_blocks.1.downsamplers.0.conv': 8, 'down_blocks.2.attentions.0.proj_in': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.1.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_q': 8,
        'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.2.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.3.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.4.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_q': 8,
        'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.5.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.6.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.7.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_q': 8,
        'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.8.ff.net.2': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_q': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_k': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_v': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_out.0': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.ff.net.0.proj': 8, 'down_blocks.2.attentions.0.transformer_blocks.9.ff.net.2': 8, 'down_blocks.2.attentions.0.proj_out': 8, 'down_blocks.2.attentions.1.proj_in': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_v': 8,
        'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.1.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.2.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.3.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_v': 8,
        'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.4.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.5.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.6.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_v': 8,
        'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.7.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.8.ff.net.2': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_q': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_k': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_v': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_out.0': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.ff.net.0.proj': 8, 'down_blocks.2.attentions.1.transformer_blocks.9.ff.net.2': 8, 'down_blocks.2.attentions.1.proj_out': 8, 'down_blocks.2.resnets.0.conv1': 8, 'down_blocks.2.resnets.0.time_emb_proj': 8,
        'down_blocks.2.resnets.0.conv2': 8, 'down_blocks.2.resnets.0.conv_shortcut': 8, 'down_blocks.2.resnets.1.conv1': 8, 'down_blocks.2.resnets.1.time_emb_proj': 8, 'down_blocks.2.resnets.1.conv2': 8, 'up_blocks.0.attentions.0.proj_in': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.0.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.1.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_out.0': 8,
        'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.2.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.3.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.4.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_out.0': 8,
        'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.5.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.6.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.7.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_out.0': 8,
        'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.8.ff.net.2': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_q': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_k': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_v': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_out.0': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.ff.net.0.proj': 8, 'up_blocks.0.attentions.0.transformer_blocks.9.ff.net.2': 8, 'up_blocks.0.attentions.0.proj_out': 8, 'up_blocks.0.attentions.1.proj_in': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.0.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_k': 8,
        'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.1.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.2.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.3.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_k': 8,
        'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.4.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.5.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.6.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_k': 8,
        'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.7.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.8.ff.net.2': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_q': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_k': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_v': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_out.0': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.ff.net.0.proj': 8, 'up_blocks.0.attentions.1.transformer_blocks.9.ff.net.2': 8, 'up_blocks.0.attentions.1.proj_out': 8, 'up_blocks.0.attentions.2.proj_in': 8,
        'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.0.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.1.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.2.ff.net.2': 8,
        'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.3.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.4.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.5.ff.net.2': 8,
        'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.6.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.7.ff.net.2': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.8.ff.net.2': 8,
        'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_q': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_k': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_v': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_out.0': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.ff.net.0.proj': 8, 'up_blocks.0.attentions.2.transformer_blocks.9.ff.net.2': 8, 'up_blocks.0.attentions.2.proj_out': 8, 'up_blocks.0.resnets.0.conv1': 8, 'up_blocks.0.resnets.0.time_emb_proj': 8, 'up_blocks.0.resnets.0.conv2': 8, 'up_blocks.0.resnets.0.conv_shortcut': 8, 'up_blocks.0.resnets.1.conv1': 8, 'up_blocks.0.resnets.1.time_emb_proj': 8, 'up_blocks.0.resnets.1.conv2': 8, 'up_blocks.0.resnets.1.conv_shortcut': 8, 'up_blocks.0.resnets.2.conv1': 8, 'up_blocks.0.resnets.2.time_emb_proj': 8, 'up_blocks.0.resnets.2.conv2': 8, 'up_blocks.0.resnets.2.conv_shortcut': 8, 'up_blocks.0.upsamplers.0.conv': 8, 'up_blocks.1.attentions.0.proj_in': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q': 8,
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.1.attentions.0.transformer_blocks.1.ff.net.2': 8, 'up_blocks.1.attentions.0.proj_out': 8, 'up_blocks.1.attentions.1.proj_in': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_v': 8,
        'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.1.attentions.1.transformer_blocks.1.ff.net.2': 8, 'up_blocks.1.attentions.1.proj_out': 8, 'up_blocks.1.attentions.2.proj_in': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj': 8, 'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_q': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_k': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_v': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_out.0': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.ff.net.0.proj': 8, 'up_blocks.1.attentions.2.transformer_blocks.1.ff.net.2': 8, 'up_blocks.1.attentions.2.proj_out': 8,
        'up_blocks.1.resnets.0.conv1': 8, 'up_blocks.1.resnets.0.time_emb_proj': 8, 'up_blocks.1.resnets.0.conv2': 8, 'up_blocks.1.resnets.0.conv_shortcut': 8, 'up_blocks.1.resnets.1.conv1': 8, 'up_blocks.1.resnets.1.time_emb_proj': 8, 'up_blocks.1.resnets.1.conv2': 8, 'up_blocks.1.resnets.1.conv_shortcut': 8, 'up_blocks.1.resnets.2.conv1': 8, 'up_blocks.1.resnets.2.time_emb_proj': 8, 'up_blocks.1.resnets.2.conv2': 8, 'up_blocks.1.resnets.2.conv_shortcut': 8, 'up_blocks.1.upsamplers.0.conv': 8, 'up_blocks.2.resnets.0.conv1': 8, 'up_blocks.2.resnets.0.time_emb_proj': 8, 'up_blocks.2.resnets.0.conv2': 8, 'up_blocks.2.resnets.0.conv_shortcut': 8, 'up_blocks.2.resnets.1.conv1': 8, 'up_blocks.2.resnets.1.time_emb_proj': 8, 'up_blocks.2.resnets.1.conv2': 8, 'up_blocks.2.resnets.1.conv_shortcut': 8, 'up_blocks.2.resnets.2.conv1': 8, 'up_blocks.2.resnets.2.time_emb_proj': 8, 'up_blocks.2.resnets.2.conv2': 8, 'up_blocks.2.resnets.2.conv_shortcut': 8, 'mid_block.attentions.0.proj_in': 8, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0': 8,
        'mid_block.attentions.0.transformer_blocks.0.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.0.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.1.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.1.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.2.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.2.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_out.0': 8,
        'mid_block.attentions.0.transformer_blocks.3.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.3.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.3.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.4.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.4.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.5.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.5.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_out.0': 8,
        'mid_block.attentions.0.transformer_blocks.6.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.6.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.6.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.7.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.7.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.8.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.8.ff.net.2': 8, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_q': 8, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_k': 8, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_v': 8, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_out.0': 8,
        'mid_block.attentions.0.transformer_blocks.9.attn2.to_q': 8, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_k': 8, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_v': 8, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_out.0': 8, 'mid_block.attentions.0.transformer_blocks.9.ff.net.0.proj': 8, 'mid_block.attentions.0.transformer_blocks.9.ff.net.2': 8, 'mid_block.attentions.0.proj_out': 8, 'mid_block.resnets.0.conv1': 8, 'mid_block.resnets.0.time_emb_proj': 8, 'mid_block.resnets.0.conv2': 8, 'mid_block.resnets.1.conv1': 8, 'mid_block.resnets.1.time_emb_proj': 8, 'mid_block.resnets.1.conv2': 8, 'conv_out': 8,
    }

w4_uniform_config = \
    {
        'conv_in': 4, 'time_embedding.linear_1': 4, 'time_embedding.linear_2': 4, 'add_embedding.linear_1': 4, 'add_embedding.linear_2': 4, 'down_blocks.0.resnets.0.conv1': 4, 'down_blocks.0.resnets.0.time_emb_proj': 4, 'down_blocks.0.resnets.0.conv2': 4, 'down_blocks.0.resnets.1.conv1': 4, 'down_blocks.0.resnets.1.time_emb_proj': 4, 'down_blocks.0.resnets.1.conv2': 4, 'down_blocks.0.downsamplers.0.conv': 4, 'down_blocks.1.attentions.0.proj_in': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj': 4, 'down_blocks.1.attentions.0.transformer_blocks.0.ff.net.2': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_k': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_v': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_out.0': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_q': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v': 4,
        'down_blocks.1.attentions.0.transformer_blocks.1.attn2.to_out.0': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.ff.net.0.proj': 4, 'down_blocks.1.attentions.0.transformer_blocks.1.ff.net.2': 4, 'down_blocks.1.attentions.0.proj_out': 4, 'down_blocks.1.attentions.1.proj_in': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj': 4, 'down_blocks.1.attentions.1.transformer_blocks.0.ff.net.2': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_q': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_k': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_v': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn1.to_out.0': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_q': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_k': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_v': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.attn2.to_out.0': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.ff.net.0.proj': 4, 'down_blocks.1.attentions.1.transformer_blocks.1.ff.net.2': 4, 'down_blocks.1.attentions.1.proj_out': 4, 'down_blocks.1.resnets.0.conv1': 4, 'down_blocks.1.resnets.0.time_emb_proj': 4, 'down_blocks.1.resnets.0.conv2': 4, 'down_blocks.1.resnets.0.conv_shortcut': 4,
        'down_blocks.1.resnets.1.conv1': 4, 'down_blocks.1.resnets.1.time_emb_proj': 4, 'down_blocks.1.resnets.1.conv2': 4, 'down_blocks.1.downsamplers.0.conv': 4, 'down_blocks.2.attentions.0.proj_in': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.0.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.1.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_q': 4,
        'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.2.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.3.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.4.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_q': 4,
        'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.5.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.6.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.7.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_q': 4,
        'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.8.ff.net.2': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn1.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_q': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_k': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_v': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.attn2.to_out.0': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.ff.net.0.proj': 4, 'down_blocks.2.attentions.0.transformer_blocks.9.ff.net.2': 4, 'down_blocks.2.attentions.0.proj_out': 4, 'down_blocks.2.attentions.1.proj_in': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.0.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_v': 4,
        'down_blocks.2.attentions.1.transformer_blocks.1.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.1.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.2.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.3.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_v': 4,
        'down_blocks.2.attentions.1.transformer_blocks.4.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.4.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.5.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.6.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_v': 4,
        'down_blocks.2.attentions.1.transformer_blocks.7.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.7.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.8.ff.net.2': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn1.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_q': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_k': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_v': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.attn2.to_out.0': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.ff.net.0.proj': 4, 'down_blocks.2.attentions.1.transformer_blocks.9.ff.net.2': 4, 'down_blocks.2.attentions.1.proj_out': 4, 'down_blocks.2.resnets.0.conv1': 4, 'down_blocks.2.resnets.0.time_emb_proj': 4,
        'down_blocks.2.resnets.0.conv2': 4, 'down_blocks.2.resnets.0.conv_shortcut': 4, 'down_blocks.2.resnets.1.conv1': 4, 'down_blocks.2.resnets.1.time_emb_proj': 4, 'down_blocks.2.resnets.1.conv2': 4, 'up_blocks.0.attentions.0.proj_in': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.0.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn1.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.1.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.attn1.to_out.0': 4,
        'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.2.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn1.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.3.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn1.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.4.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.attn1.to_out.0': 4,
        'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.5.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn1.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.6.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn1.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.7.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.attn1.to_out.0': 4,
        'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.8.ff.net.2': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn1.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_q': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_k': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_v': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.attn2.to_out.0': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.ff.net.0.proj': 4, 'up_blocks.0.attentions.0.transformer_blocks.9.ff.net.2': 4, 'up_blocks.0.attentions.0.proj_out': 4, 'up_blocks.0.attentions.1.proj_in': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.0.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_k': 4,
        'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.1.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.2.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.3.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_k': 4,
        'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.4.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.5.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.6.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_k': 4,
        'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.7.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.8.ff.net.2': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn1.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_q': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_k': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_v': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.attn2.to_out.0': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.ff.net.0.proj': 4, 'up_blocks.0.attentions.1.transformer_blocks.9.ff.net.2': 4, 'up_blocks.0.attentions.1.proj_out': 4, 'up_blocks.0.attentions.2.proj_in': 4,
        'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.0.ff.net.2': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.1.ff.net.2': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.2.ff.net.2': 4,
        'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.3.ff.net.2': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.4.ff.net.2': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.5.ff.net.2': 4,
        'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.6.ff.net.2': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.7.ff.net.2': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.8.ff.net.2': 4,
        'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.attn1.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_q': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_k': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_v': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.attn2.to_out.0': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.ff.net.0.proj': 4, 'up_blocks.0.attentions.2.transformer_blocks.9.ff.net.2': 4, 'up_blocks.0.attentions.2.proj_out': 4, 'up_blocks.0.resnets.0.conv1': 4, 'up_blocks.0.resnets.0.time_emb_proj': 4, 'up_blocks.0.resnets.0.conv2': 4, 'up_blocks.0.resnets.0.conv_shortcut': 4, 'up_blocks.0.resnets.1.conv1': 4, 'up_blocks.0.resnets.1.time_emb_proj': 4, 'up_blocks.0.resnets.1.conv2': 4, 'up_blocks.0.resnets.1.conv_shortcut': 4, 'up_blocks.0.resnets.2.conv1': 4, 'up_blocks.0.resnets.2.time_emb_proj': 4, 'up_blocks.0.resnets.2.conv2': 4, 'up_blocks.0.resnets.2.conv_shortcut': 4, 'up_blocks.0.upsamplers.0.conv': 4, 'up_blocks.1.attentions.0.proj_in': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q': 4,
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj': 4, 'up_blocks.1.attentions.0.transformer_blocks.0.ff.net.2': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_q': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_k': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_v': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn1.to_out.0': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_q': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_k': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_v': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.attn2.to_out.0': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.ff.net.0.proj': 4, 'up_blocks.1.attentions.0.transformer_blocks.1.ff.net.2': 4, 'up_blocks.1.attentions.0.proj_out': 4, 'up_blocks.1.attentions.1.proj_in': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.0.proj': 4, 'up_blocks.1.attentions.1.transformer_blocks.0.ff.net.2': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_q': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_k': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_v': 4,
        'up_blocks.1.attentions.1.transformer_blocks.1.attn1.to_out.0': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_q': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_k': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_v': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.attn2.to_out.0': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.ff.net.0.proj': 4, 'up_blocks.1.attentions.1.transformer_blocks.1.ff.net.2': 4, 'up_blocks.1.attentions.1.proj_out': 4, 'up_blocks.1.attentions.2.proj_in': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.0.proj': 4, 'up_blocks.1.attentions.2.transformer_blocks.0.ff.net.2': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_q': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_k': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_v': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn1.to_out.0': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_q': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_k': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_v': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.attn2.to_out.0': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.ff.net.0.proj': 4, 'up_blocks.1.attentions.2.transformer_blocks.1.ff.net.2': 4, 'up_blocks.1.attentions.2.proj_out': 4,
        'up_blocks.1.resnets.0.conv1': 4, 'up_blocks.1.resnets.0.time_emb_proj': 4, 'up_blocks.1.resnets.0.conv2': 4, 'up_blocks.1.resnets.0.conv_shortcut': 4, 'up_blocks.1.resnets.1.conv1': 4, 'up_blocks.1.resnets.1.time_emb_proj': 4, 'up_blocks.1.resnets.1.conv2': 4, 'up_blocks.1.resnets.1.conv_shortcut': 4, 'up_blocks.1.resnets.2.conv1': 4, 'up_blocks.1.resnets.2.time_emb_proj': 4, 'up_blocks.1.resnets.2.conv2': 4, 'up_blocks.1.resnets.2.conv_shortcut': 4, 'up_blocks.1.upsamplers.0.conv': 4, 'up_blocks.2.resnets.0.conv1': 4, 'up_blocks.2.resnets.0.time_emb_proj': 4, 'up_blocks.2.resnets.0.conv2': 4, 'up_blocks.2.resnets.0.conv_shortcut': 4, 'up_blocks.2.resnets.1.conv1': 4, 'up_blocks.2.resnets.1.time_emb_proj': 4, 'up_blocks.2.resnets.1.conv2': 4, 'up_blocks.2.resnets.1.conv_shortcut': 4, 'up_blocks.2.resnets.2.conv1': 4, 'up_blocks.2.resnets.2.time_emb_proj': 4, 'up_blocks.2.resnets.2.conv2': 4, 'up_blocks.2.resnets.2.conv_shortcut': 4, 'mid_block.attentions.0.proj_in': 4, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0': 4,
        'mid_block.attentions.0.transformer_blocks.0.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.0.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.1.attn1.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.1.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.1.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.1.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.2.attn1.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.2.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.2.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.2.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.3.attn1.to_out.0': 4,
        'mid_block.attentions.0.transformer_blocks.3.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.3.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.3.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.3.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.4.attn1.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.4.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.4.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.4.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.5.attn1.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.5.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.5.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.5.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.6.attn1.to_out.0': 4,
        'mid_block.attentions.0.transformer_blocks.6.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.6.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.6.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.6.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.7.attn1.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.7.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.7.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.7.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.8.attn1.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.8.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.8.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.8.ff.net.2': 4, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_q': 4, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_k': 4, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_v': 4, 'mid_block.attentions.0.transformer_blocks.9.attn1.to_out.0': 4,
        'mid_block.attentions.0.transformer_blocks.9.attn2.to_q': 4, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_k': 4, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_v': 4, 'mid_block.attentions.0.transformer_blocks.9.attn2.to_out.0': 4, 'mid_block.attentions.0.transformer_blocks.9.ff.net.0.proj': 4, 'mid_block.attentions.0.transformer_blocks.9.ff.net.2': 4, 'mid_block.attentions.0.proj_out': 4, 'mid_block.resnets.0.conv1': 4, 'mid_block.resnets.0.time_emb_proj': 4, 'mid_block.resnets.0.conv2': 4, 'mid_block.resnets.1.conv1': 4, 'mid_block.resnets.1.time_emb_proj': 4, 'mid_block.resnets.1.conv2': 4, 'conv_out': 4,
    }