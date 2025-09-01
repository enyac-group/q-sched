import torch
import re
from torchao.dtypes import to_affine_quantized_intx
from torchao.quantization.quant_primitives import MappingType

def _get_per_token_block_size2(x: torch.Tensor):
    block_size = []
    blk_sz = x.shape[0] // 2
    for _ in range(len(x.shape) - 1):
        block_size.append(0)
    block_size.append(blk_sz)
    return block_size


def int8act_quant(weight):
    return to_affine_quantized_intx(weight, MappingType.SYMMETRIC, _get_per_token_block_size2(weight), torch.int8, eps=torch.finfo(torch.float32).eps, zero_point_dtype=torch.int64)

def quantize_qdiff(model):
    # new_modules = {}
    for n, m2 in model.unet.named_modules():
        if isinstance(m2, torch.nn.Linear):
            # convert string format to dict format
            # eg: layers.i.conv2d --> layers[i].conv2d
            regexp = re.compile(r'\.\d+\.')
            if regexp.search(n):
                matches = regexp.finditer(n)
                num_added = 0
                for match in matches:
                    i = match.start() + num_added
                    j = match.end() + num_added
                    n = n[0:i] +  "[" + n[i+1:j-1] + "]" + n[j-1:]
                    num_added += 1
            cmd1 = "model.unet."+str(n)+".weight = torch.nn.Parameter(int8wo_quant(m2.weight))"
            exec(cmd1) 
            input_quant_func = int8act_quant
            cmd2 = "model.unet."+str(n)+".weight" + "= torch.nn.Parameter(to_linear_activation_quantized(m2.weight, input_quant_func))"
            exec(cmd2) 

    return model