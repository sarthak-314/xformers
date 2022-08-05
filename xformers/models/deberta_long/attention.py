import jax.numpy as jnp
import flax.linen as nn
import jax

from xformers.modeling_utils import ACT2FN, with_sharding_constraint, ModelConfig
from tensor_utils import split_into_blocks, concat_3_blocks

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


class SelfAttentionOutput(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.output_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.post_attention_layer_norm = nn.LayerNorm(self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.hidden_dropout_rate)

    def __call__(self, hidden_states, input_tensor, deterministic):
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'max_seq_len', 'hidden_size'))
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.post_attention_layer_norm(hidden_states + input_tensor)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'max_seq_len', 'hidden_size'))
        return hidden_states


class LocalSlidingWindowAttention(nn.Module):
    """
    TODO: Write documentation
    """
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.n_heads = self.config.num_attention_heads
        self.per_head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.hidden_size = self.config.hidden_size
        self.block_size = self.config.block_size

        self.query_proj = nn.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            dtype=self.dtype,
        )
        self.key_proj = nn.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            dtype=self.dtype,
        )
        self.value_proj = nn.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            dtype=self.dtype,
        )
        self.attention_dropout = nn.Dropout(self.config.attention_probs_dropout_prob)
        self.pos_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.position_buckets = self.config.position_buckets
        self.max_relative_positions = self.config.position_buckets

        self.output_proj = nn.Dense(
            self.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            dtype=self.dtype,
        )
        self.output_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.output_layernorm = nn.LayerNorm(self.config.layer_norm_eps)

    def _split_heads(self, x):
        if x.ndim == 3:
            batch_size, seq_len, hidden_dim = x.shape
            new_x_shape = (batch_size, seq_len, self.n_heads, self.per_head_dim)
            return x.reshape(new_x_shape)
        
        new_x_shape = x.shape[:-1] + (self.n_heads, self.per_head_dim)
        return x.reshape(new_x_shape)
    
    def __call__(
        self, 
        hidden_states,
        rel_pos_embeddings=None,
        deterministic=True,
    ):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # (batch_size, seq_len, n_heads, per_head_dim)
        content_query_layer = self._split_heads(self.query_proj(hidden_states))
        content_key_layer = self._split_heads(self.key_proj(hidden_states))
        content_value_layer = self._split_heads(self.value_proj(hidden_states))

        # Add sharding constraints
        content_query_layer = with_sharding_constraint(
            x=content_query_layer,
            logical_axis_resources=('batch', 'length', 'heads', 'per_head_dim')
        )
        content_key_layer = with_sharding_constraint(
            x=content_key_layer,
            logical_axis_resources=('batch', 'length', 'heads', 'per_head_dim')
        )
        content_value_layer = with_sharding_constraint(
            x=content_value_layer,
            logical_axis_resources=('batch', 'length', 'heads', 'per_head_dim')
        )

        # Split into blocks -> (batch_size, num_blocks, block_size, n_heads, per_head_dim)
        content_query_layer = split_into_blocks(content_query_layer, self.block_size, axis=1)
        content_key_layer = split_into_blocks(content_key_layer, self.block_size, axis=1)
        content_value_layer = split_into_blocks(content_value_layer, self.block_size, axis=1)

        # Concatenate 3 blocks for keys and values
        # (batch_size, num_blocks, block_size*3, n_heads, per_head_dim)
        content_key_layer = concat_3_blocks(content_key_layer, block_axis=1, seq_axis=2, pad_value=0)
        content_value_layer = concat_3_blocks(content_value_layer, block_axis=1, seq_axis=2, pad_value=0)

        # Compute content to content attention weights
        # (batch_size, num_blocks, n_heads, block_size, 3*block_size)
        c2c_attention_weights = jnp.einsum(
            '...qhd,...khd->...hqk', content_query_layer, content_key_layer
        )

        # (2*max_relative_pos_embeddings, n_heads, per_head_dim)
        rel_pos_embeddings = self.pos_dropout(rel_pos_embeddings, deterministic=deterministic)
        pos_query_layer = self._split_heads(self.query_proj(rel_pos_embeddings))
        pos_key_layer = self._split_heads(self.key_proj(rel_pos_embeddings))
        
        # Spread to batch size
        # (batch_size, 2*max_relative_pos_embeddings, n_heads, per_head_dim)
        pos_query_layer = jnp.stack([pos_query_layer]*batch_size)
        pos_key_layer = jnp.stack([pos_key_layer]*batch_size)

        # Split & concatenate blocks -> (batch_size, num_blocks, block_size, n_heads, per_head_dim)
        pos_query_layer = split_into_blocks(pos_query_layer, self.block_size, axis=1)
        pos_key_layer = split_into_blocks(pos_key_layer, self.block_size, axis=1)
        pos_key_layer = concat_3_blocks(pos_key_layer, block_axis=1, seq_axis=2, pad_value=0)

        # (block_size, 3*block_size)
        query_position = jnp.arange(self.block_size, dtype=jnp.int32)[:, jnp.newaxis]
        key_position = jnp.arange(3*self.block_size, dtype=jnp.int32)[jnp.newaxis, :]
        relative_pos_matrix = query_position - key_position
        relative_pos_matrix = make_log_bucket_position(
            relative_pos_matrix, 
            self.position_buckets,
            self.max_relative_positions,
        )
        print('relative_pos_matrix:', relative_pos_matrix)

        # Content to Position Attention Weights
        # (batch_size, num_blocks, n_heads, block_size, 3*block_size)
        c2p_attention_weights = jnp.einsum(
            '...qhd,...khd->...hqk', 
            content_query_layer, 
            pos_key_layer,
        )
        attention_span = self.position_buckets
        c2p_pos_idx = jnp.clip(relative_pos_matrix + attention_span, 0, attention_span * 2 - 1)
        print('c2p_pos_idx:', c2p_pos_idx)
        c2p_attention_weights = jnp.take(c2p_attention_weights, c2p_pos_idx)

        # Position to Content Attention Weights
        p2c_pos_idx = jnp.clip(relative_pos_matrix + attention_span, 0, attention_span * 2 - 1)
        p2c_attention_weights = jnp.einsum(
            '...qhd,...khd->...hqk', 
            pos_query_layer, 
            content_key_layer,
        )
        p2c_attention_weights = jnp.take(p2c_attention_weights, p2c_pos_idx)

        # Compute attention probs my multiplying attention weights with values
        scale = jnp.sqrt(self.per_head_dim * 3)
        attention_weights = c2c_attention_weights + c2p_attention_weights + p2c_attention_weights
        attention_weights /= scale
        
        attention_probs = jax.nn.softmax(attention_weights)
        attention_probs = self.attention_dropout(attention_probs, deterministic=deterministic)
        context_layer = jnp.einsum(
            '...hqk,...khd->...qhd',
            attention_probs,
            content_value_layer,
        )

        batch_size, num_blocks, block_size, n_heads, per_head_dim = context_layer.shape
        new_shape = (batch_size, num_blocks*block_size, n_heads*per_head_dim)
        context_layer = jnp.reshape(context_layer, new_shape)

        # Output projection + Skip Connection + Post Layer normalization
        context_layer = self.output_proj(context_layer)
        context_layer = self.output_dropout(context_layer, deterministic=deterministic)
        context_layer = self.output_layernorm(context_layer + hidden_states)
        return context_layer


class FullSelfAttention(nn.Module):
    pass