import torch
from distutils.version import LooseVersion

is_torch_1_2_plus = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')
datatype = torch.bool if is_torch_1_2_plus else torch.uint8


def subsequent_mask(size, device="cpu", dtype=datatype):
    """Create mask for subsequent steps (1, size, size)

    :param int size: size of mask
    :param str device: "cpu" or "cuda" or torch.Tensor.device
    :param torch.dtype dtype: result dtype
    :rtype: torch.Tensor
    >>> subsequent_mask(3)
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=dtype)
    return torch.tril(ret, out=ret)
