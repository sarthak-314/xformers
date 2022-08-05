import jax.numpy as jnp
import flax.linen as nn
import jax

from xformers.modeling_utils import ACT2FN, ModelConfig, with_sharding_constraint, remat
from xformers.layers import WordEmbed
from xformers.models.deberta_long.attention import FullSelfAttention, LocalSlidingWindowAttention

class Embeddings(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.word_embeddings = WordEmbed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(self.config.layer_norm_epsilon)
        self.dropout = nn.Dropout(self.config.attention_probs_dropout_rate)
    
    def __call__(self, input_ids, attention_mask, deterministic=False):
        input_ids = with_sharding_constraint(input_ids, ('batch_size', 'max_seq_len'))
        inputs_embeds = self.word_embeddings(input_ids)
        inputs_embeds = self.layer_norm(inputs_embeds)
        inputs_embeds = self.dropout(inputs_embeds, deterministic=deterministic)
        inputs_embeds = with_sharding_constraint(inputs_embeds, ('batch_size', 'max_seq_len', 'hidden_size'))
        return inputs_embeds


class IntermediateMLP(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.intermediate_proj = nn.Dense(
            self.config.intermediate_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.intermediate_act_fn = ACT2FN[self.config.hidden_activation]

    def __call__(self, hidden_states):
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'max_seq_len', 'hidden_size'))
        hidden_states = self.intermediate_proj(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'intermediate_size'))
        return hidden_states


class EncoderLayerOutput(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.output_proj = nn.Dense(
            self.config.hidden_size,
            use_bias=True,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.post_mlp_layer_norm = nn.LayerNorm(self.config.layer_norm_epsilon)
        self.dropout = nn.Dropout(self.config.hidden_dropout_rate)
    
    def __call__(self, hidden_states, input_tensor, deterministic=False):
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'intermediate_size'))
        hidden_states = self.output_proj(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.post_mlp_layer_norm(hidden_states + input_tensor)
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'hidden_size'))
        return hidden_states


class EncoderLayer(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        if self.config.attention_type == 'full':
            self.attention = FullSelfAttention(self.config, self.dtype)
        elif self.config.attention_type == 'sliding_window':
            self.attention = LocalSlidingWindowAttention(self.config, self.dtype)
        self.intermediate = IntermediateMLP(self.config, self.dtype)
        self.output = EncoderLayerOutput(self.config, self.dtype)
    
    def __call__(self, hidden_states, attention_mask, relative_position_embeddings, deterministic=False):
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'max_seq_len', 'hidden_size'))
        attention_output = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            relative_position_embeddings=relative_position_embeddings,
            deterministic=deterministic,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, deterministic=deterministic)
        layer_output = with_sharding_constraint(layer_output, ('batch_size', 'max_seq_len', 'hidden_size'))
        return layer_output

class Encoder(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        if self.config.remat_policy == 'minimal':
            policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
        else: 
            policy = None
        
        EncoderLayer = remat(
            EncoderLayer,
            variables=True,
            rngs=True,
            concrete=False,
            prevent_cse=not self.config.scan_layers,
            policy=policy,
        )
        self.layers = [
            EncoderLayer(self.config, self.dtype)
            for _ in range(self.config.num_hidden_layers)
        ]
        
        self.relative_position_embeddings = nn.Embed(
            num_embeddings=self.config.num_relative_position_embeddings,
            features=self.config.hidden_size,
            dtype=self.dtype,
        )
        self.relative_position_embeddings_layer_norm = nn.LayerNorm(self.config.layer_norm_epsilon)

    def __call__(self, inputs_embeds, attention_mask, deterministic=False):
        hidden_states = with_sharding_constraint(inputs_embeds, ('batch_size', 'max_seq_len', 'hidden_size'))
        
        relative_position_embeddings = self.relative_position_embeddings.embeddings
        relative_position_embeddings = self.relative_position_embeddings_layer_norm(
            relative_position_embeddings
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                relative_position_embeddings=relative_position_embeddings,
                deterministic=deterministic,
            )
        
        hidden_states = with_sharding_constraint(hidden_states, ('batch_size', 'max_seq_len', 'hidden_size'))
        return hidden_states

class Backbone(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype

    def setup(self):
        self.embeddings = Embeddings(self.config, self.dtype)
        self.encoder = Encoder(self.config, self.dtype)
        self.pooler = Pooler(self.config, self.dtype)

    def __call__(self, inputs_embeds, attention_mask, deterministic=False):
        hidden_states = self.embeddings(inputs_embeds)
        hidden_states = self.encoder(hidden_states, attention_mask, deterministic=deterministic)
        sequence_output = self.pooler(hidden_states)
        return sequence_output



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







