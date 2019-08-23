#!/usr/bin/env python3 

import numpy as np
import torch 


def is_batch_broadcasting_possible(tt_a, tt_b):
    """Check that the batch broadcasting possible for the given batch sizes.
    Returns true if the batch sizes are the same or if one of them is 1.
    If the batch size that is supposed to be 1 is not known on compilation stage,
    broadcasting is not allowed.
    Args:
        tt_a: TensorTrain or TensorTrainBatch
        tt_b: TensorTrain or TensorTrainBatch
    Returns:
        Bool
    """
    try:
        if tt_a.batch_size is None and tt_b.batch_size is None:
            # If both batch sizes are not available on the compilation stage,
            # we cannot say if broadcasting is possible so we will not allow it.
            return False
        if tt_a.batch_size == tt_b.batch_size:
            return True
        if tt_a.batch_size == 1 or tt_b.batch_size == 1:
            return True
        return False
    except AttributeError:
        # One or both of the arguments are not batch tensor, but single TT tensors.
        # In this case broadcasting is always possible.
        return True
    

def squeeze_batch_dim(tt):
    """Converts batch size 1 TensorTrainBatch into TensorTrain.
    Args:
        tt: TensorTrain or TensorTrainBatch.
    Returns:
        TensorTrain if the input is a TensorTrainBatch with batch_size == 1 (known
          at compilation stage) or a TensorTrain.
        TensorTrainBatch otherwise.
    """
    try:
        if tt.batch_size == 1:
            return tt[0]
        else:
            return tt
    except AttributeError:
        # tt object does not have attribute batch_size, probably already
        # a TensorTrain.
        return tt

def expand_batch_dim(tt):
    """Creates an one-element TensorTrainBatch from a TensorTrain.
    Args: 
      tt: TensorTrain or TensorTrainBatch.
    Returns:
        TensorTrainBatch
    """
    if hasattr(tt, 'batch_size'):
        return tt 
    else:
        from tc.tc_cores import TensorTrainBatch 
        tt_cores = [] 
        for core_idx in range(tt.ndims):
            tt_cores.append(tt.tt_cores[core_idx].unsqueeze(0))

        return TensorTrainBatch(tt_cores, tt.get_raw_shape(), tt.get_tt_ranks(), 
            batch_size=1)
