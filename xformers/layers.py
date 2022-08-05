import jax.numpy as jnp
import flax.linen as nn

import flax 
import jax

from flax.linen import partitioning as nn_partitioning
param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint

class WordEmbed(nn.Module):
    """
    Map token ids from integers [0, vocab_size] to d-dimensional vectors.
    
    Performs the gather with one-hot contraction rather than true gather
    because it's needed for SPMD partitioning.

    Attributes:
        num_embeddings: number of embeddings (vocab_size)
        features: number of feature dimensions (d)
        dtype: the dtype of the embedding vectors
    """
    num_embeddings: int
    features: int 
    dtype: jnp.dtype

    def setup(self):
        self.embedding = param_with_axes(
            name='embedding',
            init_fn=jax.nn.initializers.zeros,
            axes=('vocab_size', 'hidden_size'),
        )
    
    def __call__(self, input_ids):
        """
        Embeds the input ids
        Args:
            input_ids: tokenized input ids of shape [batch_size, seq_len]
        Returns:
            Embedded vectors of shape [batch_size, seq_len, features]
        """
        iota_arange = jax.lax.iota(dtype=jnp.int32, size=self.num_embeddings)
        one_hot = jnp.array(input_ids[..., jnp.newaxis]==iota_arange, dtype=self.dtype)
        embedding = jnp.asarray(self.embedding, dtype=self.dtype)
        output = jnp.dot(one_hot, embedding)
        return output