
import flax.linen as nn
import jax.numpy as jnp
import jax

import flax.linen.partitioning as nn_partitioning
param_with_axes = nn_partitioning.param_with_axes


class WordEmbed(nn.Module):
  """
  Word Embedding Layer to map integers [0, vocab_size] to d-dimensional vectors.
  
  Attributes:
    num_embeddings: number of embeddings (vocab_size).
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  num_embeddings: int
  features: int
  dtype: jnp.dtype = jnp.float32
  one_hot: bool = False

  def setup(self):
    self.embedding = param_with_axes(
        'embedding',
        jax.nn.initializers.glorot_uniform,
        (self.num_embeddings, self.features),
        dtype=self.dtype,
        axes=('vocab', 'embed'),
    )

  def __call__(self, input_ids):
    """
    Embeds the input tokens using the embedding matrix.
    Args:
        input_ids: input tokens as integers.
    Returns:
        Embedded input tokens with an additional `features` dimension appended.
    """
    # iota is an array of sequential indices
    iota = jax.lax.iota(jnp.int32, self.num_embeddings)
    one_hot = jnp.array(input_ids[..., None] == iota, dtype=self.dtype)
    output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    return output


