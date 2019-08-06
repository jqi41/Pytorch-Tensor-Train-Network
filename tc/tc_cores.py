#!/usr/bin/env python3 

import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from functools import reduce 
import tc

class TensorTrain(object):
    """Represents a Tensor-Train object (a TT-tensor or TT-matrix) as a 
    tuple of TT-cores."""

    def __init__(self, tt_cores, shape=None, tt_ranks=None):
        """Creates a `TensorTrain`.
        Args:
            tt_cores: A tuple of 3d or 4d tensor-like objects of shape `[r_k-1, n_k, r_k]` 
                or [r_k-1, n_k, m_k, r_k]. Tensor-like can be numpy array, or torch.tensor
            shape: Shape of the underlying tensor. If None, tries to infer from the cores 
                (not always possible even if it should be, e.g. if ranks are unknown, than 
                the whole shape of a core can be unknown).
            tt_ranks: a TensorShape of length d+1 (d is the dimensionality of the 
                underlying tensor). The first and the last ranks are assumed to equal 
                to 1. If None, tries to infer the ranks from the cores.
            convert_to_tensors: bool, if True than convert each element of the tt_cores 
                tuple into a tf.Tensor (e.g. to initialize from np.array)
        Returns:
            A `TensorTrain`.
        Raises:
            ValueError if the provided TT-cores are not valid or inconsistent with the 
            provided shape.
        """
        tt_cores = list(tt_cores)
        if not _are_tt_cores_valid(tt_cores, shape, tt_ranks):
            raise ValueError('The tt_cores are not valid due to inconsistent dtypes')
        self._tt_cores = tuple(tt_cores)
        self._raw_shape = clean_raw_shape(shape)
        if self._raw_shape is None:
            self._raw_shape = _infer_raw_shape(self._tt_cores)
        self._tt_ranks = None if tt_ranks is None else tt_ranks
        if self._tt_ranks is None:
            self._tt_ranks = _infer_tt_ranks(self._tt_cores)

    def get_raw_shape(self):
        """Get tuple of `TensorShapes` representing the shapes of the underlying TT-tensor.
        Tuple contains one `TensorShape` for TT-tensor and 2 `TensorShapes` for TT-matrix
        Returns:
            A tuple of `TensorShape` objects.
        """
        return self._raw_shape 

    def get_shape(self):
        """Get the `torch.tensor' representing the shape of the dense tensor.
        Returns:
            A `torch.tensor' object.
        """
        raw_shape = self.get_raw_shape()
        if self.is_tt_matrix():
            prod_f = lambda arr: reduce(lambda x, y: x*y, arr)
            m = prod_f(list(raw_shape[0]))
            n = prod_f(list(raw_shape[1]))
            return tuple((m, n))
        else:
            return raw_shape[0]

    def is_tt_matrix(self):
        """ Return True if the TensorTrain object represents a TT-matrix."""
        return len(self.get_raw_shape()) == 2

    def get_tt_ranks(self):
        """ A tuple of TT-ranks """
        return self._tt_ranks 

    def assign(self, tt_new):
        if self._raw_shape != tt_new.get_raw_shape():
            raise ValueError('There is a mismatch for the two tensors.')
        if self._tt_ranks != tt_new.tt_ranks():
            raise ValueError('There is a mismatch for the two tt_ranks.')
        for i in range(len(self._tt_cores)):
            self._tt_cores = tt_new.tt_cores[i]

    @property
    def tt_cores(self):
        """A tuple of TT-cores.
        Returns:
            A tuple of 3d or 4d tensors shape `[r_k-1, n_k, r_k]' or
            `[r_k-1, n_k, m_k, r_k]'
        """
        return self._tt_cores 

    @property
    def ndims(self):
        """Get the number of dimensions of the underlying TT-tensor.
        Returns:
            A number.
        """
        return len(self.tt_cores)

    @property
    def type(self):
        """The type of elements in this tensor."""
        return self.tt_cores[0].type()

    @property
    def left_tt_rank_dim(self):
        """The dimension of the left TT-rank of each TT-core."""
        return 0

    @property 
    def right_tt_rank_dim(self):
        """The dimension of the right TT-rank of each TT-core."""
        if self.is_tt_matrix():
            # The dimension of each TT-core are [left_rank, n, m, right_rank]
            return 3 
        else:
            # The dimension of each TT-core are [left_rank, n, right_rank]
            return 2 

    def __str__(self):
        """A string describing the TensorTrain object, its TT-rank, and shape."""
        shape = self.get_shape()
        tt_ranks = self.get_tt_ranks()
        if self.is_tt_matrix():
            raw_shape = self.get_raw_shape()
            return "A TT-Matrix of size %d x %d, underlying tensor " \
                    "shape: %s x %s, TT-ranks: %s" %(shape[0], shape[1], raw_shape[0], raw_shape[1], tt_ranks)
        else:
            return "A Tensor Train of shape %s, TT-ranks: %s" %(shape, tt_ranks) 

    def __add__(self, other):
        """Returns a TensorTrain corresponding to element-wise sum tt_a + tt_b.
        Supports broadcasting (e.g. you can add TensorTrainBatch and TensorTrain).
        Just calls t3f.add, see its documentation for details."""
        import tc.tc_math
        return tc.tc_math.add(self, tc_math.multiply(other, -1))

    def __neg__(self):
        """Returns a TensorTrain corresponding to element-wise negative -tt_a.
        Just calls t3f.multiply(self, -1.), see its documentation for details.
        """
        import tc.tc_math 
        return tc.tc_math.multiply(self, -1)

    def __mul__(self, other):
        """Returns a TensorTrain corresponding to element-wise product tt_a * tt_b."""
        import tc.tc_math 
        return tc.tc_math.multiply(self, other)


def _are_tt_cores_valid(tt_cores, shape, tt_ranks):
    """Check if dimensions of the TT-cores are consistent and the dtypes coincide.
    Args:
        tt_cores: a tuple of `Tensor` objects
           shape: An np.array, a torch.tensor, a tuple of torch.tensor
                    (for TT-matrices or tensors), or None
        tt_ranks: An np.array or a torch.tensor of length len(tt_cores)+1.
    Returns:
        boolean, True if the dimensions and dtypes are consistent.
    """
    shape = clean_raw_shape(shape)
    num_dims = len(tt_cores)

    for core_idx in range(1, num_dims):
        if tt_cores[core_idx].type() != tt_cores[0].type():
            return False 
    try:
        for core_idx in range(num_dims):
            curr_core_shape = tt_cores[core_idx].shape 
            if len(curr_core_shape) != len(tt_cores[0].shape):
                # Shapes are not consistent. 
                return False 
            if shape is not None:
                for i in range(len(shape)):
                    if curr_core_shape[i+1] != shape[i][core_idx]:
                        # The TT-cores are not aligned with the given shape.
                        return False 
            if core_idx >= 1:
                prev_core_shape = tt_cores[core_idx - 1].shape 
                if curr_core_shape[0] != prev_core_shape[-1]:
                   # TT-ranks are inconsistent.
                   return False 
            if tt_ranks is not None:
                if curr_core_shape[0] != tt_ranks[core_idx]:
                    # The TT-ranks are not aligned with the TT-cores shape.
                    return False 
                if curr_core_shape[-1] != tt_ranks[core_idx + 1]:
                    # The TT-ranks are not aligned with the TT-cores shape.
                   return False 
        if tt_cores[0].shape[0] != 1 or tt_cores[-1].shape[-1] != 1:
            # The first or the last rank is not 1
            return False 
    except ValueError:
        # The shape of the TT-cores is undermined, cannot validate it.
        pass 
    return True 

def _infer_raw_shape(tt_cores):
    """Tries to infer the (static) raw shape from the TT-cores."""
    num_dims = len(tt_cores)
    num_tensor_shapes = len(tt_cores[0].shape) - 2
    raw_shape = [[] for _ in range(num_tensor_shapes)]
    for dim in range(num_dims):
        curr_core_shape = tt_cores[dim].shape 
        for i in range(num_tensor_shapes):
            raw_shape[i].append(curr_core_shape[i+1])
    for i in range(num_tensor_shapes):
        raw_shape[i] = list(raw_shape[i])

    return tuple(raw_shape)

def _infer_tt_ranks(tt_cores):
    """Tries to infer the (static) raw shape from TT-cores."""
    tt_ranks = []
    for i in range(len(tt_cores)):
        tt_ranks.append(tt_cores[i].shape[0])
    tt_ranks.append(tt_cores[-1].shape[-1])

    return list(tt_ranks)

def clean_raw_shape(shape):
    """Returns a tuple of TensorShapes for any valid shape representation.
    Args:
        shape: An np.array, a tf.TensorShape (for tensors), a tuple of
                tf.TensorShapes (for TT-matrices or tensors), or None
    Returns:
        A tuple of tf.TensorShape, or None if the input is None"""
    if shape is None:
        return None
    if isinstance(shape, torch.Tensor) or isinstance(shape[0], torch.Tensor):
        # Assume torch.Tensor.
        if isinstance(shape, torch.Tensor):
            shape = tuple(shape)
    else:
        np_shape = np.array(shape)
        # Make sure that the shape is 2-d array both for tensors and TT-matrices.
        np_shape = np.squeeze(np_shape)
        if len(np_shape.shape) == 1:
         # A tensor.
            np_shape = [np_shape]
        shape = []
        for i in range(len(np_shape)):
            shape.append(list(np_shape[i]))
        shape = tuple(shape)

    return shape



