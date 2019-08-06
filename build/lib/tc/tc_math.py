#!/usr/bin/env python3

import numpy as np 
import torch 
import tc
import tc.tc_init 
from tc.tc_decomp import orthogonalize_tt_cores
from tc.tc_cores import TensorTrain


def tt_dense_matmul(tt_matrix_a, matrix_b):
    """Multiplies a TT-matrix by a regular matrix, returns a regular matrix.
    Args:
        tt_matrix_a: `TensorTrain` object containing a TT-matrix of size M x N
        matrix_b: torch.tensor of size N x P
    Returns
        torch.tensor of size M x P
    """
    if not isinstance(tt_matrix_a, TensorTrain) or not tt_matrix_a.is_tt_matrix():
        raise ValueError('The first argument should be a TT-matrix')
    ndims = tt_matrix_a.ndims 
    a_columns = tt_matrix_a.get_shape()[1]
    b_rows = matrix_b.shape[0]
    if a_columns is not None and b_rows is not None:
        if a_columns != b_rows:
            raise ValueError('Arguments shapes should align got %s and %s instead.'
                    %(tt_matrix_a.get_shape(), matrix_b.shape))
    a_shape = tt_matrix_a.get_shape()
    a_raw_shape = tt_matrix_a.get_raw_shape()
    b_shape = matrix_b.shape
    a_ranks = tt_matrix_a.get_tt_ranks()
    # If A is (i0, ..., id-1) x (j0, ..., jd-1) and B is (j0, ..., jd-1) x K,
    # data is (K, j0, ..., jd-2) x jd-1 x 1  
    data = matrix_b.t()
    data = data.reshape(-1, a_raw_shape[1][-1], 1)
    for core_idx in reversed(range(ndims)):
        curr_core = tt_matrix_a.tt_cores[core_idx]
        # On the k = core_idx iteration, after applying einsum the shape of data
        # becomes ik x (ik-1..., id-1, K, j0, ..., jk-1) x rank_k
        data = torch.einsum('aijb,rjb->ira', [curr_core, data])
        if core_idx > 0:
            # After reshape the shape of data becomes
            # (ik, ..., id-1, K, j0, ..., jk-2) x jk-1 x rank_k
            new_data_shape = (-1, a_raw_shape[1][core_idx-1], a_ranks[core_idx])
            data = data.reshape(new_data_shape)
    # At the end the shape of the data is (i0, ..., id-1) x K
    return data.reshape(int(a_shape[0]), int(b_shape[1]))


def transpose(tt_matrix):
    """ Transpose a TT-matrix.
    Args:
        tt_matrix: `TensorTrain` or `TensorTrainBatch` object containing a TT-matrix
            (or a batch of TT-matrices).
    Returns:
        `TensorTrain` or `TensorTrainBatch` object containing a transposed TT-matrix
        (or a batch of TT-matrices).
    Raises:
        ValueError if the argument is not a TT-matrix. """
    if not isinstance(tt_matrix, TensorTrain) or not tt_matrix.is_tt_matrix():
        raise ValueError('The argument should be a TT-matrix.')

    transposed_tt_cores = []
    for core_idx in range(tt_matrix.ndims):
        curr_core = tt_matrix.tt_cores[core_idx]
        if isinstance(tt_matrix, TensorTrain):
            transposed_tt_cores.append(curr_core.permute(0, 2, 1, 3))

    tt_matrix_shape = tt_matrix.get_raw_shape()
    transposed_shape = tt_matrix_shape[1], tt_matrix_shape[0]
    tt_ranks = tt_matrix.get_tt_ranks()
    if isinstance(tt_matrix, TensorTrain):
        return TensorTrain(transposed_tt_cores, transposed_shape, tt_ranks)


def dense_tt_matmul(matrix_a, tt_matrix_b):
    """Multiplies a regular matrix by a TT-matrix, returns a regular matrix.
    Args:
        matrix_a: torch.tensor of size M x N
        tt_matrix_b: `TensorTrain` object containing a TT-matrix of size N x P
    Returns
        torch.tensor of size M x P
    """
    a_t = matrix_a.t()
    b_t = transpose(tt_matrix_b)
    return tt_dense_matmul(b_t, a_t).t()


def tt_tt_matmul(tt_matrix_a, tt_matrix_b):
    """Multiplies two TT-matrices and returns the TT-matrix of the result.
    Args:
        tt_matrix_a: `TensorTrain` or `TensorTrainBatch` object containing
            a TT-matrix (a batch of TT-matrices) of size M x N
        tt_matrix_b: `TensorTrain` or `TensorTrainBatch` object containing
            a TT-matrix (a batch of TT-matrices) of size N x P
    Returns
        `TensorTrain` object containing a TT-matrix of size M x P if both arguments
            are `TensorTrain`s
        `TensorTrainBatch` if any of the arguments is a `TensorTrainBatch`
    Raises:
        ValueError is the arguments are not TT matrices or if their sizes are not
        appropriate for a matrix-by-matrix multiplication.
    """
    if not isinstance(tt_matrix_a, TensorTrain) or not isinstance(tt_matrix_b, TensorTrain) or \
        not tt_matrix_a.is_tt_matrix() or not tt_matrix_b.is_tt_matrix():
        raise ValueError('Arguments should be TT-matrices.')

    ndims = tt_matrix_a.ndims 
    if tt_matrix_b.ndims != ndims:
        raise ValueError('Arguments should have the same number of dimensions,  \
            got %d and %d instead.' % (ndims, tt_matrix_b.ndims()))
    
    einsum_str = 'aijb,cjkd->acikbd'

    result_cores = []
    a_shape = tt_matrix_a.get_raw_shape()
    a_ranks = tt_matrix_a.get_tt_ranks()
    b_shape = tt_matrix_b.get_raw_shape()
    b_ranks = tt_matrix_b.get_tt_ranks()

    for core_idx in range(ndims):
        a_core = tt_matrix_a.tt_cores[core_idx]
        b_core = tt_matrix_b.tt_cores[core_idx]
        curr_res_core = torch.einsum(einsum_str, [a_core, b_core])

        res_left_rank = a_ranks[core_idx] * b_ranks[core_idx]
        res_right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
        left_mode = a_shape[0][core_idx]
        right_mode = b_shape[1][core_idx]

        core_shape = (res_left_rank, left_mode, right_mode, res_right_rank)
        curr_res_core = curr_res_core.reshape(core_shape)
        result_cores.append(curr_res_core)

    res_shape = (tt_matrix_a.get_raw_shape()[0], tt_matrix_b.get_raw_shape()[1])
    static_a_ranks = tt_matrix_a.get_tt_ranks()
    static_b_ranks = tt_matrix_b.get_tt_ranks()
    out_ranks = [a_r * b_r for a_r, b_r in zip(static_a_ranks, static_b_ranks)]
    
    return TensorTrain(result_cores, res_shape, out_ranks)


def matmul(a, b):
    """Multiplies two matrices that can be TT-, dense, or sparse.
    Note that multiplication of two TT-matrices returns a TT-matrix with much
    larger ranks.
    Also works for multiplying two batches of TT-matrices or a product between a
    TT-matrix and a batch of TT-matrices.
    Args:
        a: `TensorTrain`
        b: `TensorTrain`
    Returns
        If both arguments are `TensorTrain` objects, returns a `TensorTrain`
        object containing a TT-matrix of size M x P.
        Otherwise, returns torch.Tensor of size M x P.
    """
    if isinstance(a, TensorTrain) and isinstance(b, TensorTrain):
        return tt_tt_matmul(a, b)
    elif isinstance(a, TensorTrain) and isinstance(b, torch.Tensor):
        return tt_dense_matmul(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, TensorTrain):
        return dense_tt_matmul(a, b)
    else:
        raise ValueError('Argument types are not supported in matmul: %s x %s'%(a, b))


def tt_tt_flat_inner(tt_a, tt_b):
    """ Inner product between two TT-tensors or TT-matrices along all axis. 
    The shapes of tt_a and tt_b should coincide.
  
    Args:
        tt_a: `TensorTrain` object
        tt_b: `TensorTrain` object
    Returns
        a number or a Tensor with numbers for each element in the batch.
        sum of products of all the elements of tt_a and tt_b
    Raises:
        ValueError if the arguments are not `TensorTrain` objects, have different
        number of TT-cores, different underlying shape, or if you are trying to
        compute inner product between a TT-matrix and a TT-tensor.
      
    Complexity:
        Multiplying two single TT-objects is O(d r^3 n) where d is the number of
        TT-cores (tt_a.ndims()), r is the largest TT-rank
            max(tt_a.get_tt_rank(), tt_b.get_tt_rank())
        and n is the size of the axis dimension, e.g.
            for a tensor of size 4 x 4 x 4, n is 4;
            for a 9 x 64 matrix of raw shape (3, 3, 3) x (4, 4, 4) n is 12
        A more precise complexity is O(d r1 r2 n max(r1, r2)) where r1 is the largest 
        TT-rank of tt_a and r2 is the largest TT-rank of tt_b.
        The complexity of this operation for batch input is O(batch_size d r^3 n).
    """
    if not isinstance(tt_a, TensorTrain) or not isinstance(tt_b, TensorTrain):
        raise ValueError('Arguments should be TensorTrain')

    if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
        raise ValueError('One of the arguments is a TT-tensor, the other is a TT-matrix, disallowed.')

    are_both_matrices = tt_a.is_tt_matrix() and tt_b.is_tt_matrix()

    # TODO: compare shapes and raise if not consistent.
    ndims = tt_a.ndims
    if tt_b.ndims != ndims:
        raise ValueError('Arguments should have the same number of dimensions, got %d and \
            %d instead.' %(ndims, tt_b.ndims))
    axes_str = 'ij' if are_both_matrices else 'i'

    a_core = tt_a.tt_cores[0]
    b_core = tt_b.tt_cores[0]
    # Simplify an example of this operation:
    # if both arguments are TT-tensors, then it is 
    init_einsum_str = 'a{0}b,c{0}d->bd'.format(axes_str)
    print(a_core.shape)
    print(b_core.shape)
    res = torch.einsum(init_einsum_str, [a_core, b_core])

    # TODO: name the operation and the resulting tensor.
    einsum_str = 'ac,a{0}b,c{0}d->bd'.format(axes_str)
    for core_idx in range(1, ndims):
        a_core = tt_a.tt_cores[core_idx]
        b_core = tt_b.tt_cores[core_idx]
        # Simplify example of this operation:
        # if both arguments are TT-tensors, then it is 
        res = torch.einsum(einsum_str, [res, a_core, b_core])

    return res.squeeze()


def flat_inner(a, b):
    """Inner product along all axis.
    The shapes of a and b should coincide.
    Args:
        a: `TensorTrain` object
        b: `TensorTrain` object
    Returns
        a number
            sum of products of all the elements of a and b
    """
    if isinstance(a, TensorTrain) and isinstance(b, TensorTrain):
        return tt_tt_flat_inner(a, b)
    else:
        raise ValueError('Arugment types are not supported in flat_inner: %s x %s' %(a, b))


def full_tt(tt):
    """ Converts a TensorTrain into a regular tensor or matrix (torch.tesnor).
    Args:
        tt: 'TensorTrain' object.
    Returns:
        torch.tensor.
    """
    num_dims = tt.ndims 
    ranks = tt.get_tt_ranks()
    shape = tt.get_shape()
    raw_shape = tt.get_raw_shape()

    res = tt.tt_cores[0]
    for i in range(1, num_dims):
        res = res.reshape(-1, ranks[i])
        curr_core = tt.tt_cores[i].reshape(int(ranks[i]), -1)
        res = torch.mm(res, curr_core)
    
    if tt.is_tt_matrix():
        intermediate_shape = []
        for i in range(num_dims):
            intermediate_shape.append(raw_shape[0][i])
            intermediate_shape.append(raw_shape[1][i])
        res = res.reshape(intermediate_shape)
        transpose = []
        for i in range(0, 2*num_dims, 2):
            transpose.append(i)
        for i in range(1, 2*num_dims, 2):
            transpose.append(i)
        res = res.permute(transpose)
        return res.reshape(shape)
    else:
        return res.reshape(shape)


def frobenius_norm_squared(tt, differentiable=False):
    """Frobenius norm squared of `TensorTrain' or of each TT in `TensorTrainBatch'.
    Frobenius norm squared is the sum of squares of all elements in a tensor.
    Args:
        tt: 'TensorTrain' object
        differentiable: bool, whether to use a differentiable implementation or a fast and
            stable implementation based on QR decomposition.
    Returns
        a number which is the Frobenius norm squared of `tt', if it is `TensorTrain' or
        a Tensor of the size tt.batch_size, consisting of the Frobenius norms squared of
        each TensorTrain in 'tt', if it is 'TensorTrainBatch'
    """
    if differentiable:
        if tt.is_tt_matrix():
            einsum_str = 'aijb,cijd->bd'
            running_prod = torch.einsum(einsum_str, [tt.tt_cores[0], tt.tt_cores[0]])
        else:
            einsum_str = 'aib,cid->bd'
            running_prod = torch.einsum(einsum_str, [tt.tt_cores[0], tt.tt_cores[0]])

        for core_idx in range(1, tt.ndims):
            curr_core = tt.tt_cores[core_idx]
            if tt.is_tt_matrix():
                einsum_str = 'ac,aijb,cijd->bd'
                running_prod = torch.einsum(einsum_str, [running_prod, curr_core, curr_core])
            else:
                einsum_str = 'ac,aib,cid->bd'
                running_prod = torch.einsum(einsum_str, [running_prod, curr_core, curr_core])

        return running_prod.squeeze(-1).squeeze(-2)

    else:
        orth_tt = orthogonalize_tt_cores(tt, left_to_right=True)
        # All the cores of orth_tt except the last one are orthogonal, hence
        # the Frobenius norm of orth_tt equals to the norm of the last core.  
        return torch.norm(orth_tt.tt_cores[-1]) ** 2


def frobenius_norm(tt, epsilon=1e-5, differentiable=False):
    """Frobenius norm of `TensorTrain' or of each TT in `TensorTrainBatch'
    Frobenius norm is the sqrt of the sum of squares of all elements in a tensor.
    Args:
        tt: `TensorTrain' object
        epsilon: the function actually computes sqrt(norm_squared + epsilon) for 
            numerical stability (e.g. gradient of the sqrt at zero is inf).
        differentiable: bool, whether to use a differentiable implementation or 
            a fast and stable implementation based on the QR decomposition.
    
    Returns:
        a number which is the Frobenius norm of 'tt', if it is 'TensorTrain'.
    """
    return torch.sqrt(frobenius_norm_squared(tt, differentiable) + epsilon)


def quadratic_form(A, b, c):
    """Quadratic form b^t A c; A is a TT-matrix, b and c can be batches.
    Args:
        A: 'TensorTrain' object containing a TT-matrix of the size N x M.
        b: 'TensorTrain' object containing a TT-matrix of the size N x 1.
        c: 'TensorTrain' object containing a TT-matrix of the size M x 1
    Returns:
        A number, the value of the quadratic form if all the arguments are 
            'TensorTrain's.
    Raises:
        ValueError if the arguments are not TT-matrices or if the shapes are not consistent.

    Complexity:
        O(r_A r_c r_b n d (r_b + r_A n + r_c))
        d is the number of TT-cores (A.ndims());
        r_A is the largest TT-rank of A max(A.get_tt_ranks())
        n is the size of the axis dimensions e.g.
            if b and c are tensors of shape (3, 3, 3),
            A is a 27 x 27 matrix of tensor shape (3, 3, 3) x (3, 3, 3)
            then n is 3
    """
    if not isinstance(A, TensorTrain) or not A.is_tt_matrix():
        raise ValueError('A should be either TensorTrain, and the arguments \
            should be a TT-matrix.')
    
    # TODO: support torch.tensor as b and c.
    if not isinstance(b, TensorTrain) or not b.is_tt_matrix():
        raise ValueError('b should be either TensorTrain, and the arguments \
            should be a TT-matrix.')
    
    if not isinstance(c, TensorTrain) or not c.is_tt_matrix():
        raise ValueError('c should be either TensorTrain, and the arguments \
            should be a TT-matrix.')

    ndims = A.ndims 
    curr_core_1 = b.tt_cores[0]
    curr_core_2 = c.tt_cores[0]
    curr_matrix_core = A.tt_cores[0]

    # We enumerate teh dummy dimension (that takes 1 value) with `k`. 
    einsum_str = 'aikb,cijd,ejkf->bdf'

    res = torch.einsum(einsum_str, [curr_core_1, curr_matrix_core, curr_core_2])
    for core_idx in range(1, ndims):
        curr_core_1 = b.tt_cores[core_idx]
        curr_core_2 = c.tt_cores[core_idx]
        curr_matrix_core = A.tt_cores[core_idx]
        einsum_str = 'ace,aikb,cijd,ejkf->bdf'

        res = torch.einsum(einsum_str, [res, curr_core_1, curr_matrix_core, curr_core_2])

    # Squeeze to make the result a number instead of 1 x 1 for the NON batch case and 
    # to make teh result a tensor of the size batch instead of batch_size x 1 x 1 in 
    # the batch case.    
    return res.squeeze()


def renormalize_tt_cores(tt, epsilon=1e-8):
    """Renormalizes TT-cores to make them of the same Frobenius norm.

    Doesn't change the tensor represented by `tt` object, but renormalizes the
    TT-cores to make further computations more stable.

    Args:
      tt: `TensorTrain` or `TensorTrainBatch` object
      epsilon: parameter for numerical stability of sqrt
    Returns:
      `TensorTrain` which represents the same tensor as tt, but with all cores 
      having equal norm.
    """
    if isinstance(tt, TensorTrain):
        new_cores = []
        running_log_norm = 0
        core_norms = []
        for core in tt.tt_cores:
            cur_core_norm = torch.sqrt(max(torch.sum(core ** 2), epsilon))
            core_norms.append(cur_core_norm)
            running_log_norm += np.log(cur_core_norm)
        
        running_log_norm = running_log_norm / tt.ndims 
        fact = np.exp(running_log_norm)
        for i, core in enumerate(tt.tt_cores):
            new_cores.append(core * fact / core_norms[i])

        return TensorTrain(new_cores)


def _add_tensor_cores(tt_a, tt_b):
    """Internal function to be called from add for two TT-tensors.
    Does the actual assembling of the TT-cores to add two TT-tensors.
    """
    ndims = tt_a.ndims 
    shape = tt_a.get_raw_shape()
    a_ranks = tt_a.get_tt_ranks()
    b_ranks = tt_b.get_tt_ranks()
    tt_cores = []
    for core_idx in range(ndims):
        a_core = tt_a.tt_cores[core_idx]
        b_core = tt_b.tt_cores[core_idx]
        if core_idx == 0:
            curr_core = torch.cat([a_core, b_core], 2)
        elif core_idx == ndims - 1:
            curr_core = torch.cat([a_core, b_core], 0)
        else:
            upper_zeros = torch.zeros((a_ranks[core_idx], shape[0][core_idx], 
                                       b_ranks[core_idx+1]))
            lower_zeros = torch.zeros((b_ranks[core_idx], shape[0][core_idx], 
                                       a_ranks[core_idx+1]))
            upper = torch.cat([a_core, upper_zeros], 2)
            lower = torch.cat([lower_zeros, b_core], 2)
            curr_core = torch.cat([upper, lower], 0)
        tt_cores.append(curr_core)

    return tt_cores


def _add_matrix_cores(tt_a, tt_b):
    """Internal function to be called from add for two TT-matrices.
    Does the actual assembling of the TT-cores to add two TT-matrices.
    """  
    ndims = tt_a.ndims 
    dtype = tt_a.type 
    shape = tt_a.get_raw_shape()
    a_ranks = tt_a.get_tt_ranks()
    b_ranks = tt_b.get_tt_ranks()
    tt_cores = []
    for core_idx in range(ndims):
        a_core = tt_a.tt_cores[core_idx]
        b_core = tt_b.tt_cores[core_idx]
        if core_idx == 0:
            curr_core = torch.cat([a_core, b_core], 3)
        elif core_idx == ndims - 1:
            curr_core = torch.cat([a_core, b_core], 0)
        else:
            upper_zeros = torch.zeros((a_ranks[core_idx], shape[0][core_idx], 
                                    shape[1][core_idx], b_ranks[core_idx+1])).type(dtype)
            lower_zeros = torch.zeros((b_ranks[core_idx], shape[0][core_idx],
                                    shape[1][core_idx], a_ranks[core_idx+1])).type(dtype)
            upper = torch.cat([a_core, upper_zeros], 3)
            lower = torch.cat([lower_zeros, b_core], 3)
            curr_core = torch.cat([upper, lower], 0)
        tt_cores.append(curr_core)

    return tt_cores 


def add(tt_a, tt_b):
    """Returns a TensorTrain corresponding to elementwise sum tt_a + tt_b.
    The shapes of tt_a and tt_b should coincide.
    Args:
        tt_a: `TensorTrain`, TT-tensor, or TT-matrix
        tt_b: `TensorTrain`, TT-tensor, or TT-matrix
    Returns
        a `TensorTrain` object corresponding to the element-wise sum of arguments if
         both arguments are `TensorTrain`s.
    Raises
        ValueError if the arguments shapes do not coincide
    """
    ndims = tt_a.ndims 
    if tt_a.is_tt_matrix() != tt_b.is_tt_matrix():
        raise ValueError('The arguments should be both TT-tensors or both TT-matrices')

    if tt_a.get_raw_shape() != tt_b.get_raw_shape():
        raise ValueError('The arguments should have the same shape.')

    if tt_a.is_tt_matrix():
        tt_cores = _add_matrix_cores(tt_a, tt_b)
    else:
        tt_cores = _add_tensor_cores(tt_a, tt_b)

    out_ranks = [1]
    static_a_ranks = tt_a.get_tt_ranks()
    static_b_ranks = tt_b.get_tt_ranks()
    for core_idx in range(1, ndims):
        out_ranks.append(static_a_ranks[core_idx] + static_b_ranks[core_idx])
    out_ranks.append(1)

    return TensorTrain(tt_cores, tt_a.get_raw_shape(), out_ranks)


def multiply(tt_left, right):
    """Returns a TensorTrain corresponding to element-wise product tt_left * right.
    Args:
        tt_left: `TensorTrain` object
        right: `TensorTrain` OR a number.
    Returns
        a `TensorTrain` object corresponding to the element-wise product of the arguments.
    Raises
        ValueError if the arguments shapes do not coincide or broadcasting is not possible.
    """
    ndims = tt_left.ndims 
    if not isinstance(right, TensorTrain):
        # Assume right is a number, not TensorTrain.
        # To squash right uniformly across TT-cores we pull its absolute value
        # and raise to the power 1/ndims. First TT-core is multiplied by the sign
        # of right.
        tt_cores = list(tt_left.tt_cores)
        right = torch.tensor(right)
        fact = torch.pow(torch.abs(right), 1.0/ndims)
        sign = torch.sign(right)
        for i in range(len(tt_cores)):
            tt_cores[i] = fact * tt_cores[i]

        tt_cores[0] = tt_cores[0] * sign 
        out_ranks = tt_left.get_tt_ranks()
    else:
        if tt_left.is_tt_matrix() != right.is_tt_matrix():
            raise ValueError('The arugments should be both TT-tensors or both '
                            'TT-matrices.')
        if tt_left.get_raw_shape() != right.get_raw_shape():
            raise ValueError('The arguments should have the same shape.')
        
        a_ranks = tt_left.get_tt_ranks()
        b_ranks = right.get_tt_ranks()
        shape = tt_left.get_raw_shape()
    
        is_matrix = tt_left.is_tt_matrix()
        tt_cores = []

        for core_idx in range(ndims):
            a_core = tt_left.tt_cores[core_idx]
            b_core = right.tt_cores[core_idx]
            left_rank = a_ranks[core_idx] * b_ranks[core_idx]
            right_rank = a_ranks[core_idx + 1] * b_ranks[core_idx + 1]
            if is_matrix:
                einsum_str = 'aijb,cijd->acijbd'
                curr_core = torch.einsum(einsum_str, [a_core, b_core])
                curr_core = curr_core.reshape((left_rank, shape[0][core_idx], shape[1][core_idx], right_rank))
            else:
                einsum_str = 'aib,cid->acibd'
                curr_core = torch.einsum(einsum_str, [a_core, b_core])
                curr_core = curr_core.reshape((left_rank, shape[0][core_idx], right_rank))
            tt_cores.append(curr_core)
    
        combined_ranks = zip(tt_left.get_tt_ranks(), right.get_tt_ranks())
        out_ranks = [a * b for a, b in combined_ranks]

    return TensorTrain(tt_cores, tt_left.get_raw_shape(), out_ranks)


if __name__ == "__main__":
    
    from tc_init import lecun_initializer, he_initializer, glorot_initializer
    shape1 = [[4, 7, 4, 7], [5, 5, 5, 5]]
    shape2 = [[5, 5, 5, 5], [4, 3, 4, 3]]
    shape3 = [[4, 7, 4, 7], [5, 5, 5, 5]]
    rng_tt1 = lecun_initializer(shape1, tt_rank=3)
    rng_tt2 = he_initializer(shape2, tt_rank=3)
    rng_tt3 = tt_tt_matmul(rng_tt1, rng_tt2)

    rng_tt3 = transpose(transpose(glorot_initializer(shape3, tt_rank=3)))
    norm_rng_tt3 = renormalize_tt_cores(rng_tt3)
    print(flat_inner(rng_tt1, rng_tt3))

    print(full_tt(rng_tt1).shape)
    print(frobenius_norm(rng_tt1, 1e-6))
    
    rng_tt4 = rng_tt1 + rng_tt3
    print(rng_tt4.get_shape())

    print(rng_tt1)
    print(rng_tt4)
    print(multiply(rng_tt4, -1))

    matrix_1 = torch.randn(625, 15)
    print(tt_dense_matmul(rng_tt1, matrix_1))
    print(rng_tt1 * rng_tt1)

    matrix_2 = torch.randn(768, 200)
    shape4 = [[16, 16, 3], [4, 8, 5]]
    shape5 = [[4, 8, 5], [4, 8, 8]]
    rng_tt4 = glorot_initializer(shape4, tt_rank=3)
    rng_tt5 = glorot_initializer(shape5, tt_rank=3)
    res1 = dense_tt_matmul(matrix_2.t(), rng_tt4)
    res2 = dense_tt_matmul(res1, rng_tt5)
    print(res1.shape)
    print(res2.shape)



