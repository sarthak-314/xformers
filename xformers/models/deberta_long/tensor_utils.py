import jax.numpy as jnp
import numpy as np
import jax

def split_into_blocks(array, block_size, axis):
    """
    Splits an array into block along the given `axis`.
    
    Args:
        array: Array of shape [..., seq_len, ...].
        block_size: Length of each block.
        axis: A valid axis in `array` to split along. 
    Returns:
        Array of shape [..., num_blocks, block_size, ...], where num_blocks = seq_len // block_size
    """
    seq_len = array.shape[axis]
    num_blocks = seq_len // block_size
    output_shape = (array.shape[:axis] + (num_blocks, block_size) + array.shape[(axis + 1):])
    return array.reshape(output_shape)

def concat_3_blocks(blocked_seq, block_axis, seq_axis, pad_value=0):
    """
    Concatenates 3 consecutive blocks for each input block for local attention.
    Warning: This is slow when used with scan due to slicing, I think.

    Args:
        blocked_seq: [..., num_blocks, block_size, ...] shaped Array.
        block_axis: axis of the `num_blocks` dimension.
        seq_axis: integer axis of the `block_size` dimension.
        pad_value: The scalar pad value to use for the first and last input blocks.
    Returns:
        Array of shape [..., num_blocks, 3 * block_size, ...].
    """
    num_blocks, block_size = blocked_seq.shape[block_axis], blocked_seq.shape[seq_axis]

    # Add padding to the first and last blocks
    # (..., num_blocks, block_size, ...) -> (..., num_blocks + 2, block_size, ...)
    pad = [(0, 0)] * blocked_seq.ndim
    pad[block_axis] = (1, 1)
    blocked_seq = jnp.pad(blocked_seq, pad, constant_values=pad_value)

    # PAD blocks PAD -> PAD blocks[:-1]
    left_indices = [slice(0, None)] * blocked_seq.ndim
    left_indices[block_axis] = slice(0, num_blocks)
    left_blocks = blocked_seq[tuple(left_indices)]

    # PAD blocks PAD -> blocks
    center_indices = [slice(0, None)] * blocked_seq.ndim
    center_indices[block_axis] = slice(1, num_blocks + 1)
    center_blocks = blocked_seq[tuple(center_indices)]

    # PAD blocks PAD -> blocks[1:] PAD
    right_indices = [slice(0, None)] * blocked_seq.ndim
    right_indices[block_axis] = slice(2, num_blocks + 2)
    right_blocks = blocked_seq[tuple(right_indices)]

    # (..., num_blocks, 3 * block_size, ...)
    result = jnp.concatenate([left_blocks, center_blocks, right_blocks], axis=seq_axis)
    return result


def make_log_bucket_position(relative_pos_matrix, num_buckets, max_position):
    """
    Translate relative position to a log bucket relative position.
    The relative position is defined as key_position - query_position, 
    i.e. the distance in tokens from the attending position to the attended-to position.

    We use smaller buckets for small absolute relative_position and larger buckets for larger absolute relative_positions. 
    All relative positions >=max_distance map to the same bucket. 
    All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on.    
    """
    sign = jnp.sign(relative_pos_matrix)
    
    # Half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_pos_in_exact_range = (relative_pos_matrix < max_exact) & (relative_pos_matrix > -max_exact)
    abs_pos = jnp.where(is_pos_in_exact_range, max_exact-1, jnp.abs(relative_pos_matrix))
    
    x = jnp.log((max_position - 1) / max_exact)
    xx = jnp.log(abs_pos/max_exact) / x
    log_pos = jnp.ceil(xx*(max_exact-1)) + max_exact
    bucket_pos = jnp.where(abs_pos<=max_exact, relative_pos_matrix, log_pos*sign).astype(jnp.int32)
    return bucket_pos