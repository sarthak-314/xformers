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