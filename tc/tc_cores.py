#!/usr/bin/env python3 

import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from functools import reduce 

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
        import tc_math
        return tc_math.add(self, tc_math.multiply(other, -1))

    def __neg__(self):
        """Returns a TensorTrain corresponding to element-wise negative -tt_a.
        Just calls t3f.multiply(self, -1.), see its documentation for details.
        """
        import tc_math 
        return tc_math.multiply(self, -1)

    def __mul__(self, other):
        """Returns a TensorTrain corresponding to element-wise product tt_a * tt_b."""
        import tc_math 
        return tc_math.multiply(self, other)


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


# A batch of TensorTrain class
class TensorTrainBatch(object):
    """ 
    Representing a batch of Tensor-Train objects (TT-tensors or TT-matrices)
    as a tuple of TT-cores.
    """
    def __init__(self, tt_cores, shape=None, tt_ranks=None, batch_size=None):
        """Creates a `TensorTrainBatch`.
        Args:
            tt_cores: A tuple of 4d or 5d tensor-like objects of shape
                        `[batch_size, r_k-1, n_k, r_k]` or
                        `[batch_size, r_k-1, n_k, m_k, r_k]`
                        Tensor-like can be numpy array, tf.Tensor, of tf.Variable
            batch_size: number of elements in the batch. If None, tries to infer from
                        the TT-cores (not always possible even if it should be, e.g. if 
                        ranks are unknown, than the whole shape of a core can be unknown).
                 shape: Shape of the underlying tensor. If None, tries to infer from 
                        the TT-cores.
              tt_ranks: a TensorShape of length d+1 (d is the dimensionality of 
                        the underlying tensor). The first and the last ranks are assumed 
                        to equal to 1. 
            If None, tries to infer the ranks from the cores.
        Returns:
            A `TensorTrainBatch`.
        """
        tt_cores = list(tt_cores)
        if not _are_batch_tt_cores_valid(tt_cores, shape, tt_ranks, batch_size):
            raise ValueError('The tt_cores provided to TensorTrainBatch constructor '
                    'are not valid, have different dtypes, or are '
                    'inconsistent with the provided batch_size, shape, or '
                    'TT-ranks.')
        self._tt_cores = tuple(tt_cores)
        if batch_size is None:
            self._batch_size = tt_cores[0].shape[0]
        else:
            self._batch_size = batch_size
        self._raw_shape = clean_raw_shape(shape)
        if self._raw_shape is None:
            self._raw_shape = _infer_batch_raw_shape(self._tt_cores)
        self._tt_ranks = None if tt_ranks is None else list(tt_ranks)
        if self._tt_ranks is None:
            self._tt_ranks = _infer_batch_tt_ranks(self._tt_cores)
        
    def get_shape(self):
        """Get the `torch.tensor' representing the shape of a dense tensor.
        Returns:
            A `torch.tensor' object.
        """
        raw_shape = self.get_raw_shape()
        if self.is_tt_matrix():
            # Use the python prod instead of np.prod to avoid overflows.
            prod_f = lambda arr: reduce(lambda x, y: x*y, arr)
            # TODO: as the list is not available if the shape is partly known.
            m = prod_f(list(raw_shape[0]))
            n = prod_f(list(raw_shape[1]))
            shape = tuple((m, n))
        else:
            shape = raw_shape[0]
        
        return tuple(np.hstack((self._batch_size, shape)))
    
    def is_tt_matrix(self):
        """Returns True if the TensorTrain object represents a TT-matrix."""
        return len(self.get_raw_shape()) == 2

    def get_raw_shape(self):
        """Get tuple of `TensorShapes` representing the shapes of the underlying TT-tensor.
        Tuple contains one `TensorShape` for TT-tensor and 2 `TensorShapes` for TT-matrix
        Returns:
            A tuple of `TensorShape` objects.
        """
        return self._raw_shape

    def get_tt_ranks(self):
        """A tuple of TT-ranks"""
        return self._tt_ranks   

    @property
    def ndims(self):
        """Get the number of dimensions of the underlying TT-tensor.
        Returns:
            A number.
        """
        return len(self._tt_cores)

    @property 
    def tt_cores(self):
        """A tuple of TT-cores.
        Returns:
            A tuple of 4d or 5d tensors shape
                `[batch_size, r_k-1, n_k, r_k]`
            or
                `[batch_size, r_k-1, n_k, m_k, r_k]`"""
        return self._tt_cores

    @property
    def batch_size(self):
        """The number of elements or None if not known."""
        return self._batch_size

    @property
    def left_tt_rank_dim(self):
        """The dimension of the left TT-rank in each TT-core."""
        return 1

    @property
    def right_tt_rank_dim(self):
        """The dimension of the right TT-rank in each TT-core."""
        if self.is_tt_matrix():
        # The dimensions of each TT-core are
        # [batch_idx, left_rank, n, m, right_rank]
            return 4
        else:
        # The dimensions of each TT-core are
        # [batch_idx, left_rank, n, right_rank]
            return 3

    def __str__(self):
        """A string describing the TensorTrainBatch, its TT-rank and shape."""
        shape = self.get_shape()
        tt_ranks = self.get_tt_ranks()

        if self.batch_size is None:
            batch_size_str = None 
        else:
            batch_size_str = str(self.batch_size)

        if self.is_tt_matrix():
            raw_shape = self.get_raw_shape()
            
            return "A %s element batch of the size %d x %d, underlying tensor shape: %s x %s, TT-ranks: %s" \
                %(batch_size_str, shape[1], shape[2], raw_shape[0], raw_shape[1], tt_ranks)
        else:
            type_str = 'Tensor Train'
            return "A %s element batch of %s of shape %s, TT-ranks: %s" % \
                (batch_size_str, type_str, shape[1:], tt_ranks)

    def __add__(self, other):
        """Returns a TensorTrain corresponding to element-wise sum tt_a + tt_b.
        Supports broadcasting (e.g. you can add TensorTrainBatch and TensorTrain).
        Just calls t3f.add, see its documentation for details.
        """
        import tc_math
        return tc_math.add(self, other)
            
    def __sub__(self, other):
        """Returns a TensorTrain corresponding to element-wise difference tt_a - tt_b.
        Supports broadcasting (e.g. you can subtract TensorTrainBatch and
        TensorTrain).
        Just calls t3f.add(self, (-1) * other), see its documentation for details.
        """
        # We cannot import ops in the beginning since it creates cyclic dependencies.
        import tc_math 
        return tc_math.add(self, tc_math.multiply(other, -1))

    def __neg__(self):
        """Returns a TensorTrain corresponding to element-wise negative -tt_a.
        Just calls t3f.multiply(self, -1.), see its documentation for details.
        """
        import tc_math
        return tc_math.multiply(self, -1)

    def __mul__(self, other):
        """Returns a TensorTrain corresponding to element-wise product tt_a * tt_b.
        Supports broadcasting (e.g. you can multiply TensorTrainBatch and
        TensorTrain).
        Just calls t3f.multiply, see its documentation for details.
        """
        # We can't import ops in the beginning since it creates cyclic dependencies.
        import tc_math 
        return tc_math.multiply(self, other)


def _are_batch_tt_cores_valid(tt_cores, shape, tt_ranks, batch_size):
    """Check if dimensions of the TT-cores are consistent and the dtypes coincide.
    Args:
        tt_cores: a tuple of `Tensor` objects
           shape: An np.array, a tf.TensorShape (for tensors), a tuple of
                  tf.TensorShapes (for TT-matrices or tensors), or None
        tt_ranks: An np.array or a tf.TensorShape of length len(tt_cores)+1.
      batch_size: a number or None
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
                # Shapes are inconsistent
                return False 
            if batch_size is not None and curr_core_shape[0] is not None:
                if curr_core_shape[0] != batch_size:
                    # The TT-cores are not aligned with the given batch_size.
                    return False 
            if shape is not None:
                for i in range(len(shape)):
                    if curr_core_shape[i + 2] != shape[i][core_idx]:
                        # The TT-cores are not aligned with the given shape. 
                        return False
            if core_idx >= 1:
                prev_core_shape = tt_cores[core_idx - 1].shape
                if curr_core_shape[1] != prev_core_shape[-1]:
                    # TT-ranks are inconsistent.
                    return False
            if tt_ranks is not None:
                if curr_core_shape[1] != tt_ranks[core_idx]:
                # The TT-ranks are not aligned with the TT-cores shape.
                    return False
                if curr_core_shape[-1] != tt_ranks[core_idx + 1]:
                # The TT-ranks are not aligned with the TT-cores shape.
                    return False
        if tt_cores[0].shape[1] != 1 or tt_cores[-1].shape[-1] != 1:
            # The first or the last rank is not 1.
            return False
    except ValueError:
        # The shape of the TT-cores is undetermined, can not validate it.
        pass
    return True


def _infer_batch_raw_shape(tt_cores):
    """Tries to infer the (static) raw shape from the TT-cores."""
    num_dims = len(tt_cores)
    num_tensor_shapes = len(tt_cores[0].shape) - 3
    raw_shape = [[] for _ in range(num_tensor_shapes)]
    for dim in range(num_dims):
        curr_core_shape = tt_cores[dim].shape
        for i in range(num_tensor_shapes):
            raw_shape[i].append(curr_core_shape[i + 2])
    for i in range(num_tensor_shapes):
        raw_shape[i] = list(raw_shape[i])

    return tuple(raw_shape)

def _infer_batch_tt_ranks(tt_cores):
    """Tries to infer the (static) raw shape from the TT-cores."""
    tt_ranks = []
    for i in range(len(tt_cores)):
        tt_ranks.append(tt_cores[i].shape[1])
    tt_ranks.append(tt_cores[-1].shape[-1])

    return list(tt_ranks)

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
    """Creates a 1-element TensorTrainBatch from a TensorTrain.
    Args:
        tt: TensorTrain or TensorTrainBatch.
    Returns:
        TensorTrainBatch
    """
    if hasattr(tt, 'batch_size'):
        return tt
    else:
        tt_cores = []
        for core_idx in range(tt.ndims()):
            tt_cores.append(tt.tt_cores[core_idx].squeeze())

        return TensorTrainBatch(tt_cores, tt.get_raw_shape(), tt.get_tt_ranks(), batch_size=1)

