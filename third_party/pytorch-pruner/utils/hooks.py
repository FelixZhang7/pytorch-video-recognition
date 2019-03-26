import torch
import numpy as np


# Module.register_forward_hook
def bind_output_hook(module, input, output):
    # module manage buffer as tensor, we can't add a list into buffer
    if hasattr(module, 'output'):
        module.output.append(output)
    else:
        module.output = [output]
    if hasattr(module, 'output_shape'):
        module.output_shape.append(list(output.shape))
    else:
        module.output_shape = [list(output.shape)]


# Module.register_backward_hook
def compute_rank_hook(module, grad_input, grad_output):
    assert isinstance(grad_output[0], torch.Tensor), grad_output[0]
    assert hasattr(module, 'output')
    assert module.output[0].shape == grad_output[0].shape
    ranks = module.output[0] * grad_output[0]
    # L2 Norm inside layer at rank calc, so div it or not is not critical
    # dim = ranks.size()
    # ranks = ranks / (dim[2] * dim[3])
    while ranks.dim() > 2:
        ranks = ranks.sum(dim=-1)
    ranks = torch.abs(ranks)
    ranks = ranks.sum(0)  # sum for batch
    if not hasattr(module, 'rank'):
        module.register_buffer('rank', torch.zeros(module.out_channels).cuda())
    if not hasattr(module, 'batch_size'):
        module.register_buffer('batch_size', torch.zeros(1).cuda())
    module.rank += ranks.data
    module.batch_size += torch.from_numpy(np.array([module.output[0].shape[0]])).float().cuda()
    module.output.pop(0)


def bind_retain_grad(module, input, output):
    output.retain_grad()
