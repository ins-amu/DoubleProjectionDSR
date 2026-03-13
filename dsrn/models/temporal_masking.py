
from collections.abc import Sequence

import torch

def get_skip_tuple(n_ignore):
    if n_ignore is None:
        return (0,0)
    elif isinstance(n_ignore, Sequence) and len(n_ignore) == 2:
        return (n_ignore[0], n_ignore[1])
    else:
        return (n_ignore, n_ignore)


def skip_tuple_to_mask(skip_tuple, nt):
    mask = torch.ones(nt, dtype=bool)
    mask[0:skip_tuple[0]]     = False
    mask[nt-skip_tuple[1]:nt] = False

    return mask