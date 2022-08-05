import jax.numpy as jnp
import flax.linen as nn

import jax

from flax.linen import partitioning as nn_partitioning
with_sharding_constraint = nn_partitioning.with_sharding_constraint
scan_with_axes = nn_partitioning.scan_with_axes
remat = nn_partitioning.remat

from ...layers import Config, WordEmbed

class Embeddings(nn.Module):
    config: Config 
    dtype: jnp.dtype

    def setup(self):
        self.word_embeddings = WordEmbed(
            num_embeddings=self.config.vocab_size,
            features=self.config.embedding_size,
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.attention_probs_dropout_prob)
        
    def __call__(self, input_ids, attention_mask, deterministic=False):
        inputs_embeds = self.word_embeddings(input_ids.astype(jnp.int32))
        inputs_embeds = self.layer_norm(inputs_embeds)
        inputs_embeds = self.dropout(inputs_embeds, deterministic=deterministic)
        inputs_embeds = inputs_embeds.astype(self.dtype)
        inputs_embeds = with_sharding_constraint(inputs_embeds, ('batch', 'seq_length', 'embed'))
        return inputs_embeds


class SelfAttention(nn.Module):
    """
    Multi-headed dot-product attention.

    Attributes:
        n_heads: Number of attention heads.
        per_head_dim: Dimension of each head
        attention_probs_dropout_rate: Dropout rate
    """

    n_heads: int
    per_head_dim: int
    attention_probs_dropout_rate: float

    def setup(self):
        self.hidden_size = self.n_heads * self.per_head_dim
        
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

    def _split_heads(self, x):
        if x.ndim == 3:
            batch_size, seq_len, hidden_dim = x.shape
            new_x_shape = (batch_size, seq_len, self.n_heads, self.per_head_dim)
            return x.reshape(new_x_shape)
        
        new_x_shape = x.shape[:-1] + (self.n_heads, self.per_head_dim)
        return x.reshape(new_x_shape)

    def __call__(self, hidden_states, attention_mask, deterministic=True):
        batch_size, seq_len, hidden_size = hidden_states.shape

        # (batch_size, seq_len, n_heads, per_head_dim)
        query_layer = self._split_heads(self.query_proj(hidden_states))
        key_layer = self._split_heads(self.key_proj(hidden_states))
        value_layer = self._split_heads(self.value_proj(hidden_states))

        # Add sharding constraints
        query_layer = with_sharding_constraint(query_layer,('batch', 'seq_length', 'heads', 'per_head_dim'))
        depth_scaling = jnp.sqrt(self.per_head_dim).astype(self.dtype)

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            attention_bias = jax.lax.select(
                attention_mask>0,
                jnp.full(attention_mask.shape, 0.).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e-10).astype(self.dtype),
            )

class AttentionOutput(nn.Module):
    """
    Feed-forward layer after computing self attention
    Post-LN variant
    """

    config: Config
    dtype: jnp.dtype

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
   
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class LayerOutput(nn.Module):
    """
    Feed-forward layer after computing self attention
    Post-LN variant
    """

    config: Config
    dtype: jnp.dtype

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
   
    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class IntermediateMLP(nn.Module):
    config: Config
    dtype: jnp.dtype

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states
 


class Encoder(nn.Module):
    """
    A Stack of encoder layers
    """

    config: Config
    dtype: jnp.dtypes = jnp.float32





        


    

        

