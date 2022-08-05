import jax.numpy as jnp
import flax.linen as nn
import jax

import flax.linen.partitioning as nn_partitioning
from xformers.models.deberta_long.attention import FullSelfAttention, LocalSelfAttention

from ...modeling_utils import ACT2FN, ModelConfig
from tensor_utils import split_into_blocks, concat_3_blocks

with_sharding_constraint = nn_partitioning.with_sharding_constraint

class SelfAttentionOutput(nn.Module):
    """
    Post-LN variant
    """
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.output_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(self.config.layer_norm_epsilon, dtype=self.dtype)
        self.dropout = nn.Dropout(self.config.hidden_dropout_rate)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        hidden_states = self.dropout(hidden_states)
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'hidden_size'))
        return hidden_states


class IntermediateMLP(nn.Module):
    """
    Transformer MLP / Feed Forward layer.
    
    Attributes:
        intermediate_dim: Shared dimension of hidden layers.
        hidden_activation: string function name in flax.linen
        intermediate_dropout_rate: Dropout rate used after the intermediate layers.
        dtype: Type for the dense layer.
    """
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.intermediate_proj = nn.Dense(
            self.config.intermediate_dim,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.intermediate_act_fn = ACT2FN[self.config.hidden_activation]

    def __call__(self, hidden_states):
        hidden_states = self.intermediate_proj(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'intermediate_dim'))
        return hidden_states

class EncoderLayerOutput(nn.Module):
    """
    Post-LN variant
    """
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(self.config.layer_norm_epsilon)
        self.dropout = nn.Dropout(self.config.hidden_dropout_rate)
    
    def __call__(self, hidden_states, input_tensor, deterministic=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'hidden_size'))
        return hidden_states

class EncoderLayer(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        if self.config.attention_type == 'full':
            self.attention = FullSelfAttention(self.config, self.dtype)
        elif self.config.attention_type == 'sliding_window':
            self.attention = LocalSelfAttention(self.config, self.dtype)
        
        self.intermediate = IntermediateMLP(self.config, self.dtype)
        self.output = EncoderLayerOutput(self.config, self.dtype)
    
    def __call__(self, hidden_states, attention_mask, deterministic=False):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(hidden_states, attention_output, deterministic=deterministic)
        return layer_output, intermediate_output




class DebertaV2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output

class DebertaLongFeedForwardLayer(nn.Module):
    """
    Feed forward layer applied to output of attention layer in transformer encoder
    """
    config: transformers.DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.intermediate_act_fn = ACT2FN[self.config.hidden_act]
        self.intermediate_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            dtype=self.dtype,
        )
        self.output_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=0.01),
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def __call__(self, attention_output, deterministic=True):
        intermediate_output = self.intermediate_act_fn(self.intermediate_proj(attention_output))
        intermediate_output = with_sharding_constraint(
            x=intermediate_output,
            logical_axis_resources=('batch', 'length', 'intermediate_mlp')
        )
        layer_output = self.output_proj(intermediate_output)
        layer_output = self.dropout(layer_output, deterministic=deterministic)
        layer_output = self.layer_norm(layer_output + attention_output)
        layer_output = with_sharding_constraint(
            x=layer_output,
            logical_axis_resources=('batch', 'length', 'embed')
        )
        return layer_output

class DebertaLongLayer(nn.Module):
    config: transformers.DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = DebertaLongLocalAttention(config=self.config, dtype=self.dtype)
        self.feed_forward = DebertaLongFeedForwardLayer(config=self.config, dtype=self.dtype)

    def __call__(self, hidden_states, rel_pos_embeddings, deterministic=True):
        attention_output = self.attention(
            hidden_states=hidden_states, 
            rel_pos_embeddings=rel_pos_embeddings,
            deterministic=deterministic,
        )
        layer_output = self.feed_forward(attention_output, deterministic=deterministic)
        return layer_output

class DebertaLongBackbone(nn.Module):
    config: transformers.DebertaV2Config
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.word_embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.word_embeddings_layer_norm = nn.LayerNorm(self.config.layer_norm_eps)
        self.word_embeddings_dropout = nn.Dropout(self.config.hidden_dropout_prob)

        num_relative_position_embeddings = self.config.position_buckets*2
        print('num_relative_position_embeddings:', num_relative_position_embeddings)
        self.relative_position_embeddings = nn.Embed(
            num_embeddings=num_relative_position_embeddings,
            features=self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        self.relative_position_embeddings_layer_norm = nn.LayerNorm(self.config.layer_norm_eps)
        
        self.layers = [
            DebertaLongLayer(self.config, self.dtype)
            for _ in range(self.config.num_hidden_layers)
        ]
        
    def __call__(self, input_ids, deterministic=True):
        inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = with_sharding_constraint(
            x=inputs_embeds,
            logical_axis_resources=('batch', 'length', 'embed'),
        )
        inputs_embeds = self.word_embeddings_layer_norm(inputs_embeds)
        inputs_embeds = self.word_embeddings_dropout(inputs_embeds, deterministic=deterministic)

        rel_pos_embeddings = self.relative_position_embeddings_layer_norm(
            self.relative_position_embeddings.embedding
        )
        
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                rel_pos_embeddings=rel_pos_embeddings,
                deterministic=deterministic,
            )
        return hidden_states







