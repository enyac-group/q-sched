import torch
import torchao
from dataclasses import dataclass
# from torchao.quantization.utils import _get_per_token_block_size
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.dtypes import to_affine_quantized_intx
from torchao.quantization.linear_activation_quantized_tensor import to_linear_activation_quantized
from torchao.dtypes import CutlassInt4PackedLayout, Layout
from torchao.core.config import AOBaseConfig
from torchao.quantization.quant_api import register_quantize_module_handler, _linear_extra_repr
import types

def _get_per_token_block_size(x: torch.Tensor):
    if x.dim() == 2:
        block_size = [1, x.shape[-1]]
    else:
        block_size = []
        for _ in range(len(x.shape) - 1):
            block_size.append(1)
        block_size.append(x.shape[-1] // 4)
    return block_size


def _int4_symm_cutlass_quant(x: torch.Tensor) -> torch.Tensor:
    out = to_affine_quantized_intx(
        x,
        mapping_type=MappingType.SYMMETRIC,
        block_size=_get_per_token_block_size(x),
        target_dtype=torch.int8,
        quant_min=-8,
        quant_max=7,
        scale_dtype=torch.float16,
        eps=torch.finfo(torch.float32).eps,
        zero_point_domain=ZeroPointDomain.NONE,
        _layout=CutlassInt4PackedLayout(),
    )
    return out

def _int4_symm_quant(x: torch.Tensor) -> torch.Tensor:
    out = to_affine_quantized_intx(
        x,
        mapping_type=MappingType.SYMMETRIC,
        block_size=_get_per_token_block_size(x),
        target_dtype=torch.int8,
        quant_min=-8,
        quant_max=7,
        scale_dtype=torch.float16,
        eps=torch.finfo(torch.float32).eps,
        zero_point_domain=ZeroPointDomain.NONE,
    )
    return out

def _int5_symm_quant(x: torch.Tensor) -> torch.Tensor:
    out = to_affine_quantized_intx(
        x,
        mapping_type=MappingType.SYMMETRIC,
        block_size=_get_per_token_block_size(x),
        target_dtype=torch.int8,
        quant_min=-16,
        quant_max=15,
        scale_dtype=torch.float16,
        eps=torch.finfo(torch.float32).eps,
        zero_point_domain=ZeroPointDomain.NONE,
    )
    return out

def _int6_symm_quant(x: torch.Tensor) -> torch.Tensor:
    out = to_affine_quantized_intx(
        x,
        mapping_type=MappingType.SYMMETRIC,
        block_size=_get_per_token_block_size(x),
        target_dtype=torch.int8,
        quant_min=-32,
        quant_max=31,
        scale_dtype=torch.float16,
        eps=torch.finfo(torch.float32).eps,
        zero_point_domain=ZeroPointDomain.NONE,
    )
    return out

@dataclass
class Int4DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Applies int4 dynamic per token symmetric activation quantization and int4 per row weight symmetric quantization to linear

    Args:
        `layout`: layout type for quantized weight tensor, only supports `MarlinQQQLayout()` and `CutlassInt4PackedLayout()` for now
        `mapping_type`: quantization type for weight, controls the weight quantization is symmetric or asymmetric
        `act_mapping_type`: quantization type for activation, controls the activation quantization is symmetric or asymmetric
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    layout: Layout = CutlassInt4PackedLayout()
    mapping_type: MappingType = MappingType.SYMMETRIC
    act_mapping_type: MappingType = MappingType.SYMMETRIC
    set_inductor_config: bool = True


# for bc
int4_dynamic_activation_int4_weight = Int4DynamicActivationInt4WeightConfig


@register_quantize_module_handler(Int4DynamicActivationInt4WeightConfig)
def _int4_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module, config: Int4DynamicActivationInt4WeightConfig
) -> torch.nn.Module:
    weight = module.weight
    layout = config.layout
    mapping_type = config.mapping_type
    act_mapping_type = config.act_mapping_type
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    if not isinstance(layout, CutlassInt4PackedLayout):
        raise NotImplementedError(
            f"Only CutlassInt4PackedLayout layout is supported. Received {layout}."
        )
    if mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only mapping_type=SYMMETRIC is supported.")
    if act_mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only act_mapping_type=SYMMETRIC is supported.")

    weight = _int4_symm_quant(weight)
    weight_shape = weight.shape
    if weight.dim() == 3:
        weight = weight.view(weight.shape[0], weight.shape[1] * weight.shape[2])
    weight = to_linear_activation_quantized(
        weight,
        _int4_symm_quant,
    )


    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module

@dataclass
class Int6DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Applies int4 dynamic per token symmetric activation quantization and int4 per row weight symmetric quantization to linear

    Args:
        `layout`: layout type for quantized weight tensor, only supports `MarlinQQQLayout()` and `CutlassInt4PackedLayout()` for now
        `mapping_type`: quantization type for weight, controls the weight quantization is symmetric or asymmetric
        `act_mapping_type`: quantization type for activation, controls the activation quantization is symmetric or asymmetric
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    layout: Layout = CutlassInt4PackedLayout()
    mapping_type: MappingType = MappingType.SYMMETRIC
    act_mapping_type: MappingType = MappingType.SYMMETRIC
    set_inductor_config: bool = True


@register_quantize_module_handler(Int6DynamicActivationInt4WeightConfig)
def _int6_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module, config: Int6DynamicActivationInt4WeightConfig
) -> torch.nn.Module:
    weight = module.weight
    layout = config.layout
    mapping_type = config.mapping_type
    act_mapping_type = config.act_mapping_type
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    if not isinstance(layout, CutlassInt4PackedLayout):
        raise NotImplementedError(
            f"Only CutlassInt4PackedLayout layout is supported. Received {layout}."
        )
    if mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only mapping_type=SYMMETRIC is supported.")
    if act_mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only act_mapping_type=SYMMETRIC is supported.")

    weight = _int4_symm_quant(weight)
    weight_shape = weight.shape
    if weight.dim() == 3:
        weight = weight.view(weight.shape[0], weight.shape[1] * weight.shape[2])
    weight = to_linear_activation_quantized(
        weight,
        _int6_symm_quant,
    )


    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


int6_dynamic_activation_int4_weight = Int6DynamicActivationInt4WeightConfig

@dataclass
class Int5DynamicActivationInt4WeightConfig(AOBaseConfig):
    """Applies int4 dynamic per token symmetric activation quantization and int4 per row weight symmetric quantization to linear

    Args:
        `layout`: layout type for quantized weight tensor, only supports `MarlinQQQLayout()` and `CutlassInt4PackedLayout()` for now
        `mapping_type`: quantization type for weight, controls the weight quantization is symmetric or asymmetric
        `act_mapping_type`: quantization type for activation, controls the activation quantization is symmetric or asymmetric
        `set_inductor_config`: if True, adjusts `torchinductor` settings to recommended values.
    """

    layout: Layout = CutlassInt4PackedLayout()
    mapping_type: MappingType = MappingType.SYMMETRIC
    act_mapping_type: MappingType = MappingType.SYMMETRIC
    set_inductor_config: bool = True


@register_quantize_module_handler(Int5DynamicActivationInt4WeightConfig)
def _int5_dynamic_activation_int4_weight_transform(
    module: torch.nn.Module, config: Int5DynamicActivationInt4WeightConfig
) -> torch.nn.Module:
    weight = module.weight
    layout = config.layout
    mapping_type = config.mapping_type
    act_mapping_type = config.act_mapping_type
    if config.set_inductor_config:
        torchao.quantization.utils.recommended_inductor_config_setter()

    if not isinstance(layout, CutlassInt4PackedLayout):
        raise NotImplementedError(
            f"Only CutlassInt4PackedLayout layout is supported. Received {layout}."
        )
    if mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only mapping_type=SYMMETRIC is supported.")
    if act_mapping_type != MappingType.SYMMETRIC:
        raise NotImplementedError("Only act_mapping_type=SYMMETRIC is supported.")

    weight = _int4_symm_quant(weight)
    weight_shape = weight.shape
    if weight.dim() == 3:
        weight = weight.view(weight.shape[0], weight.shape[1] * weight.shape[2])
    weight = to_linear_activation_quantized(
        weight,
        _int5_symm_quant,
    )


    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    module.extra_repr = types.MethodType(_linear_extra_repr, module)
    return module


int5_dynamic_activation_int4_weight = Int5DynamicActivationInt4WeightConfig