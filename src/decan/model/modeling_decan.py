""" Pytorch DeCAN model """

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers.cache_utils import Cache, DynamicCache
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.modeling_utils import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_decan import DeCANConfig


logger = logging.get_logger( __name__ )


# ====================================================================
# DeCAN Cache Classes
# ====================================================================


class DeCANCacheMixin():
    """
    A class containing all functions to provide DeCAN cache functionality. To be used as a mixin for `Cache` subclasses.

    The class exposes the following functions:
        - `stack_heads` which stacks key and value heads from previous layers
        - `cache_to` which implements `.to(...)` for the key and value caches
        - `detach_cache_to` which implements `.detach().to(...)` for the key and value cache
    """

    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]
    document_ids: torch.Tensor | None

    def stack_heads( self, layer_idx: int, cache_kwargs: dict[str, Any] ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Stacks KV heads from selected layers.

        Args:
            layer_idx (int): The index of the layer to cache the states for.
            cache_kwargs (dict[str, Any]): Additional arguments for the cache subclass. Must contain `layers: list[int]` to select which layers to stack.

        Returns:
            tuple[torch.Tensor,torch.Tensor]: Tuple of stacked key and value heads from layers given by the `layers` kwarg.
        """

        # Get list of layers to stack
        layers: list[int] = cache_kwargs[ 'layers' ]
        assert isinstance( layers, list ), "kwarg 'layers' must be a list of ints"

        # Ensure layer indices are all valid
        assert not any( i > layer_idx for i in layers ), "kwarg 'layers' receive index greater than `layer_idx`"

        # Construct key and value stacks
        key_out = torch.cat( [ key for i, key in enumerate( self.key_cache ) if i in layers ], dim=-3 )
        value_out = torch.cat( [ value for i, value in enumerate( self.value_cache ) if i in layers ], dim=-3 )

        return key_out, value_out

    def cache_to( self, **kwargs ):
        """ Moves the key and value cache to the give device or dtype.

        Args:
            **kwargs: same arguments as `.to(...)`
        """
        self.key_cache = [ k.to( **kwargs ) for k in self.key_cache ]
        self.value_cache = [ v.to( **kwargs ) for v in self.value_cache ]

        if hasattr( self, 'document_ids' ) and self.document_ids is not None:
            self.document_ids = self.document_ids.to( **{ k: v for k, v in kwargs.items() if k != 'dtype' } )

    def detach_cache_to( self, **kwargs ) -> None:
        """ Moves the key and value cache to the give device or dtype and detaches from the autograd graph.

        Args:
            **kwargs: same arguments as `.to(...)`
        """
        self.key_cache = [ k.detach().to( **kwargs ) for k in self.key_cache ]
        self.value_cache = [ v.detach().to( **kwargs ) for v in self.value_cache ]

        if hasattr( self, 'document_ids' ) and self.document_ids is not None:
            self.document_ids = self.document_ids.to( **{ k: v for k, v in kwargs.items() if k != 'dtype' } )

    def update_document_ids( self, document_ids: torch.Tensor ) -> torch.Tensor:
        max_cache_length: int | None = self.get_max_cache_shape() # type: ignore
        
        if not hasattr( self, 'document_ids' ) or self.document_ids is None:
            self.document_ids = document_ids
        elif max_cache_length is None:
            self.document_ids = torch.cat( [ self.document_ids, document_ids ], dim=1 )
        else:
            self.document_ids = torch.cat( [ self.document_ids, document_ids ], dim=1 )[ :, -max_cache_length : ]

        return document_ids[ :, None, :, None ] == self.document_ids[ :, None, None, : ]


class DeCANTrainingCache( Cache, DeCANCacheMixin ):
    """ Implements a dynamic cache for DeCAN with bounded maximum size to be used during training.
    """

    def __init__( self, *, max_cache_length: int ):
        """ Instatiates a dynamic training cache for DeCAN with a bounded maximum size.

        Args:
            max_cache_length (int): Maximum cache size.
        """

        super().__init__()
        self._seen_tokens = 0
        self.max_cache_length = max_cache_length

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_tokens = key_states.shape[-2]

        if new_tokens > self.max_cache_length:
            raise ValueError( f'Tried adding {new_tokens} which is larger than max_cache_length of {self.max_cache_length}' )

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += new_tokens

        # Update the cache
        if len( self.key_cache ) <= layer_idx:
            self.key_cache.append( key_states )
            self.value_cache.append( value_states )
        else:
            cached_tokens = self.key_cache[layer_idx].shape[-2]
            start_idx = max( 0, cached_tokens - self.max_cache_length + new_tokens )

            self.key_cache[layer_idx] = torch.cat( [ self.key_cache[layer_idx][ :, :, start_idx :, : ], key_states ], dim=-2 )
            self.value_cache[layer_idx] = torch.cat( [ self.value_cache[layer_idx][ :, :, start_idx :, : ], value_states ], dim=-2 )

        assert cache_kwargs is not None, '`cache_kwargs` must not be `None`'
        return self.stack_heads( layer_idx, cache_kwargs )

    def get_seq_length( self, layer_idx: int | None = None ) -> int:
        layer_idx = layer_idx or 0
        is_empty_layer = len( self.key_cache ) == 0 or len( self.key_cache ) <= layer_idx 
        layer_seq_length = self.key_cache[layer_idx].shape[-2] if not is_empty_layer else 0
        return layer_seq_length

    def get_max_cache_shape( self ) -> int:
        return self.max_cache_length

    def get_max_length( self ) -> int | None:
        return self.get_max_cache_shape()

    def forward( self, *args, **kwargs ):
        raise NotImplementedError( 'There is no forward method.' )

    def pre_trim( self, sequence_length: int ):
        new_length = self.max_cache_length - sequence_length
        if hasattr( self, 'document_ids' ) and self.document_ids is not None:
            self.document_ids = self.document_ids[ :, -new_length : ]

        for i, ( key_cache, value_cache ) in enumerate( zip( self.key_cache, self.value_cache ) ):
            self.key_cache[i] = key_cache[ :, :, -new_length :, : ]
            self.value_cache[i] = value_cache[ :, :, -new_length :, : ]


class DeCANDynamicCache( DynamicCache, DeCANCacheMixin ):
    """ Implements a dynamic cache for DeCAN with unbounded maximum size to be used during inference.
    """

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        super().update( key_states, value_states, layer_idx, cache_kwargs )
        assert cache_kwargs is not None, '`cache_kwargs` must not be `None`'
        return self.stack_heads( layer_idx, cache_kwargs )

    def batch_split( self, full_batch_size: int, split_size: int, num_hidden_layers=None ) -> list[DynamicCache]:
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = DeCANDynamicCache()
            current_split._seen_tokens = self._seen_tokens # pylint: disable=W0212
            current_split.key_cache = [ tensor[ i : i + split_size ] for tensor in self.key_cache ]
            current_split.value_cache = [ tensor[ i : i + split_size ] for tensor in self.value_cache ]
            out.append( current_split )
        return out

    def forward( self, *args, **kwargs ):
        raise NotImplementedError( 'There is no forward method.' )


# ====================================================================
# DeCAN Layer Classes
# ====================================================================


class DeCANRMSNorm(nn.Module):
    """ DeCANRMSNorm is equivalent to T5LayerNorm """

    def __init__( self, config: DeCANConfig ):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps

        self.weight = nn.Parameter( torch.ones( self.hidden_size ) )

    def forward( self, hidden_states: torch.Tensor ) -> torch.Tensor:
        """ Performs RMS normalisation with a learnable scale. 

        Args:
            hidden_states (torch.Tensor): Layer inputs of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Normalised outputs of same shape is inputs
        """
        
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to( torch.float32 )
        variance = hidden_states.pow( 2 ).mean( -1, keepdim=True )
        hidden_states = hidden_states * torch.rsqrt( variance + self.rms_norm_eps )
        return self.weight * hidden_states.to( input_dtype )

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.rms_norm_eps}'

ALL_LAYERNORM_LAYERS.append( DeCANRMSNorm ) # type: ignore


class DeCANRotaryEmbedding( torch.nn.Module ):
    """  Rotary Embedding class which also computes the attention mask """

    def __init__( self, config: DeCANConfig ):
        super().__init__()
        self.config = config

        self.base_freq = self.config.rope_theta
        self.dim = self.config.head_dim

    def compute_rope( self, seq_length: int, device: torch.device ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Computes the sine and cosine rotary partials """

        idxs = torch.arange( seq_length, device=device, dtype=torch.float32 )
        dims = torch.arange( 0, self.dim, 2, device=device, dtype=torch.float32 )

        inv_freqs = 1.0 / ( self.base_freq ** ( dims / self.dim ) )

        freqs = torch.einsum( 'i,j->ij', idxs, inv_freqs )
        freqs = torch.cat( ( freqs, freqs ), dim=-1 )

        return freqs.cos(), freqs.sin()

    def compute_mask( self, q_len: int, k_len: int, device: torch.device ) -> torch.Tensor:
        """ Compute the causal attention mask """
        mask = torch.ones( q_len, k_len, device=device, dtype=torch.bool ).tril( k_len - q_len )
        mask = mask[ None, None, :, : ]
        return mask

    @torch.no_grad()
    def forward(
        self,
        embeddings: torch.Tensor,
        document_ids: torch.Tensor | None,
        past_key_values: Cache,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """ Computes rotary embeddings and causal attention mask.
        Must be called BEFORE the cache is modified using `.update(...)`

        Args:
            embeddings (torch.Tensor): Token embeddings prior to entering transformer backbone.
            past_key_values (Cache): KV cache container.

        Returns:
            tuple[tuple[torch.Tensor,torch.Tensor],torch.Tensor]: Tuple of ((sin, cos), mask)
        """

        assert isinstance( past_key_values, DeCANCacheMixin )

        # Get properties
        dtype = embeddings.dtype
        device = embeddings.device
        device_type = device.type if device.type != 'mps' else 'cpu'

        # Get the number of new input tokens, i.e. query length
        query_len: int = embeddings.shape[1]

        # Get usable size of cache + new tokens, i.e. key length
        key_len: int = past_key_values.get_usable_length( query_len ) + query_len

        # Compute sine and cosine rotary partials in full precision
        with torch.autocast( device_type=device_type, enabled=False ):
            cos, sin = self.compute_rope( key_len, device )

        # Compute the attention mask
        mask = self.compute_mask( query_len, key_len, device )

        # If document_ids was passed, apply document masking
        if document_ids is not None:
            mask = mask * past_key_values.update_document_ids( document_ids )

        # Cast rotary partials and return with mask
        return ( cos.to( dtype ), sin.to( dtype ) ), mask

def rotate_half( x: torch.Tensor ) -> torch.Tensor:
    """ Rotates half the hidden dims of the input """
    x1, x2 = x.chunk( 2, dim=-1 )
    return torch.cat( ( -x2, x1 ), dim=-1 )

def apply_rope( cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor ) -> torch.Tensor:
    """ Applies rope to the input with automatic var-len compensation """
    length = x.shape[-2]
    return x * cos[ ..., -length :, : ] + rotate_half( x ) * sin[ ..., -length :, : ]


class DeCANAttention( nn.Module ):
    """
    Multi-headed attention with Densely Connected Attention Network head stacking.
    """
    
    def __init__( self, config: DeCANConfig, layer_idx: int ):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx
        self.layer_num = layer_idx + 1

        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_k_heads = config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.attention_bias = config.attention_bias

        # Compute number of heads at current layer and clip by maximum number of attention heads
        self.num_q_heads = min( self.layer_num * self.num_k_heads, self.num_attention_heads )

        # Get layer connection list needed to match the query head count: [layer_idx - q // k, ..., layer_idx ]
        self.layer_select = list( range( self.layer_num - self.num_q_heads // self.num_k_heads, self.layer_num ) )

        self.q_proj = nn.Linear( self.hidden_size, self.head_dim * self.num_q_heads, bias=self.attention_bias )
        self.k_proj = nn.Linear( self.hidden_size, self.head_dim * self.num_k_heads, bias=self.attention_bias )
        self.v_proj = nn.Linear( self.hidden_size, self.head_dim * self.num_k_heads, bias=self.attention_bias )
        self.o_proj = nn.Linear( self.head_dim * self.num_q_heads, self.hidden_size, bias=self.attention_bias )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache,
        attention_mask: torch.Tensor,
        output_attentions: bool,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache]:
        """ Performs DeCAN-augmented multi-head attention.

        Args:
            hidden_states (torch.Tensor): Layer inputs of shape [batch_size, q_len, hidden_size].
            past_key_values (Cache): DeCAN augmented KV cache. Must accept `layers: list[int]` in the cache_kwargs.
            attention_mask (torch.Tensor): Boolean attention mask broadcastable to [batch, heads, q_len, k_len].
            output_attentions (bool): When `True` returns the attention weights. Currently not supported.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Tuple of sine and cosine rotary positional embeddings. Must be broadcastable to [batch, heads, k_len, head_dim].

        Returns:
            tuple[torch.Tensor,torch.Tensor,Cache]: Tuple of (outputs, attention_weights, cache). When output_attention is `False` attention_weights is `None`.
        """

        # Get batch_size and q_len
        B, L, _ = hidden_states.size() # pylint: disable=C0103

        # Project hidden states to qkv, reshape into heads, and swap the sequence and head dimensions
        q_states = self.q_proj( hidden_states ).view( B, L, self.num_q_heads, self.head_dim ).transpose( 1, 2 ).contiguous()
        k_states = self.k_proj( hidden_states ).view( B, L, self.num_k_heads, self.head_dim ).transpose( 1, 2 ).contiguous()
        v_states = self.v_proj( hidden_states ).view( B, L, self.num_k_heads, self.head_dim ).transpose( 1, 2 ).contiguous()

        # Update cache with new keys and values and return the stacked heads
        k_states, v_states = past_key_values.update(
            key_states=k_states,
            value_states=v_states,
            layer_idx=self.layer_idx,
            cache_kwargs={ 'layers': self.layer_select },
        )

        # Apply rotary embeddings to queries and keys
        q_states = apply_rope( *position_embeddings, q_states )
        k_states = apply_rope( *position_embeddings, k_states )

        if output_attentions:
            attention_matrix = torch.einsum( 'bhqd,bhkd->bhqk', q_states, k_states  ) * self.head_dim ** -0.5 + attention_mask.to( q_states.dtype ).log().repeat( 1, self.num_q_heads, 1, 1 )
            attention_matrix = attention_matrix.softmax( -1 )

            attn_output = torch.einsum( 'bhqk,bhkd->bhqd', attention_matrix, v_states )
        else:
            # Compute multi-head SDPA
            attn_output = F.scaled_dot_product_attention( # pylint: disable=E1102
                q_states,
                k_states,
                v_states,
                attn_mask=attention_mask.to( q_states.dtype ).log().repeat( 1, self.num_q_heads, 1, 1 ),
            )

            attention_matrix = None

        # Transpose and combine heads
        attn_output = attn_output.transpose( 1, 2 ).contiguous()
        attn_output = attn_output.view( B, L, self.head_dim * self.num_q_heads )

        # Project outputs for residual stream
        attn_output = self.o_proj( attn_output )

        return attn_output, attention_matrix, past_key_values


class DeCANMLP( nn.Module ):
    """ DeCANMLP is equivalent to LlammaMLP """

    def __init__( self, config: DeCANConfig ):
        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mlp_bias = config.mlp_bias

        self.gate_proj = nn.Linear( self.hidden_size, self.intermediate_size, bias=self.mlp_bias )
        self.up_proj = nn.Linear( self.hidden_size, self.intermediate_size, bias=self.mlp_bias )
        self.down_proj = nn.Linear( self.intermediate_size, self.hidden_size, bias=self.mlp_bias )
        self.act_fn = ACT2FN[ config.intermediate_act ]

    def forward( self, hidden_states: torch.Tensor ) -> torch.Tensor:
        """ Computes the gated MLP residual.

        Args:
            hidden_states (torch.Tensor): Layer inputs of shape [batch_size, seq_len, hidden_size].

        Returns:
            torch.Tensor: Residual outputs of same shape is inputs.
        """
        return self.down_proj( self.act_fn( self.gate_proj( hidden_states ) ) * self.up_proj( hidden_states ) )


class DeCANDecoderLayer( nn.Module ):
    """ DeCAN transformer block containing Attention and MLP layers """
    
    def __init__( self, config: DeCANConfig, layer_idx: int ):
        super().__init__()

        self.config = config

        self.layer_idx = layer_idx

        # self.attention_input_norm = DeCANRMSNorm( config )
        self.attention_input_norm = nn.LayerNorm( config.hidden_size )
        self.attention = DeCANAttention( config, layer_idx )

        # self.mlp_input_norm = DeCANRMSNorm( config )
        self.mlp_input_norm = nn.LayerNorm( config.hidden_size )
        self.mlp = DeCANMLP( config )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache,
        attention_mask: torch.Tensor,
        output_attentions: bool,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache]:
        """
        Args:
            hidden_states (torch.Tensor): Layer inputs of shape [batch_size, q_len, hidden_size].
            past_key_values (Cache): DeCAN augmented KV cache. Must accept `layers: list[int]` in the cache_kwargs.
            attention_mask (torch.Tensor): Boolean attention mask broadcastable to [batch, heads, q_len, k_len].
            output_attentions (bool): When `True` returns the attention weights. Currently not supported.
            position_embeddings (tuple[torch.Tensor, torch.Tensor]): Tuple of sine and cosine rotary positional embeddings. Must be broadcastable to [batch, heads, k_len, head_dim].

        Returns:
            tuple[torch.Tensor,torch.Tensor,Cache]: Tuple of (outputs, attention_weights, cache). When output_attention is `False` attention_weights is `None`.
        """

        # Normalise input, compute attention, and add residual to stream
        normed_states = self.attention_input_norm( hidden_states )
        residual, attention_weights, present_key_values = self.attention(
            hidden_states=normed_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + residual

        # Normalise input, compute MLP, and add residual to stream
        normed_states = self.mlp_input_norm( hidden_states )
        residual = self.mlp( normed_states )
        hidden_states = hidden_states + residual

        return hidden_states, attention_weights, present_key_values


# ====================================================================
# DeCAN Model Classes
# ====================================================================


class DeCANPreTrainedModel( PreTrainedModel ): # pylint: disable=W0223
    """ The bare DeCAN model with no implemented functionality """
    
    config_class = DeCANConfig
    base_model_prefix = 'model'
    _no_split_modules = [ 'DeCANDecoderLayer' ]
    _skip_keys_device_placement = [ "past_key_values" ]
    _supports_cache_class = False # Note: actually supports certain DeCAN specific caches

    def _init_weights( self, module ):
        std = self.config.initializer_range
        
        if isinstance( module, nn.Linear ):
            module.weight.data.normal_( mean=0.0, std=std )
            if module.bias is not None:
                module.bias.data.zero_()
        
        elif isinstance( module, nn.Embedding ):
            module.weight.data.normal_( mean=0.0, std=std )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class DeCANModel( DeCANPreTrainedModel ):
    """ DeCAN-augmented decoder, outputting raw hidden-states without any specific head on top. """

    def __init__( self, config: DeCANConfig ):
        super().__init__( config )
        self.padding_idx: int = config.pad_token_id
        self.vocab_size: int = config.vocab_size

        self.embed_tokens = nn.Embedding( config.vocab_size, config.vocab_dim )
        self.embed_rotary = DeCANRotaryEmbedding( config )

        self.input_proj = nn.Linear( config.vocab_dim, config.hidden_size, bias=False )
        # self.input_norm = DeCANRMSNorm( config )
        self.input_norm = nn.LayerNorm( config.hidden_size )

        self.layers = nn.ModuleList( [
            DeCANDecoderLayer( config, layer_idx ) for layer_idx in range( config.num_hidden_layers )
        ] )

        # self.final_norm = DeCANRMSNorm( config )
        self.final_norm = nn.LayerNorm( config.hidden_size )

        self.post_init()

    def get_input_embeddings( self ):
        return self.embed_tokens

    def set_input_embeddings( self, value ):
        self.embed_tokens = value

    def get_position_embeddings( self ):
        raise NotImplementedError( 'DeCAN uses RoPE and therefor does not have position embeddings' )

    def resize_position_embeddings( self, new_num_position_embeddings: int ):
        raise NotImplementedError( 'DeCAN uses RoPE and therefor does not have position embeddings' )

    def ids_to_embeddings( self, input_ids: torch.Tensor ) -> torch.Tensor:
        """ Computes embeddings from input ids using the embed_tokens and input_proj

        Args:
            input_ids (torch.Tensor): Long tensor of shape [batch_size, seq_length]

        Returns:
            torch.Tensor: Projected embeddings of shape [batch_size, seq_length, hidden_size]
        """
        return self.input_proj( self.embed_tokens( input_ids ) )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        # TODO: consider attention_mask
        # TODO: consider cache_position
        past_key_values: Cache | None = None,
        use_cache: bool | None = True,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPast:
        """ Decoder only forward pass. Returns raw hidden states without any head on top.

        Note: exactly one of `input_ids` or `inputs_embeds` must be passed.
        Note: if document_ids are passed you must continue to pass them or the cache will break.

        Args:
            input_ids (torch.Tensor, optional): Input token sequence of shape [batch_size, seq_length].
            inputs_embeds (torch.Tensor, optional): Input embeddings of shape [batch_size, seq_length, hidden_size].
            document_ids (torch.Tensor, optional): Document indices of shape [batch_size, seq_length], enables document masking.
            past_key_values (Cache, optional): KV cache container. Must be a DeCAN specific cache instance or None.
            use_cache (bool, optional): Enables use of the KV cache container. This MUST be True. Defaults to True.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers. Currently not supported. Defaults to False.
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers.
            return_dict (bool, optional): Whether or not to return a `BaseModelOutputWithPast` instead of a plain tuple.
        """
        config: DeCANConfig = self.config # type: ignore
        
        use_cache = config.use_cache if use_cache is None else use_cache
        return_dict = config.return_dict if return_dict is None else return_dict
        output_attentions = config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = config.output_hidden_states if output_hidden_states is None else output_hidden_states

        # Cache use is mandatory
        if not use_cache:
            raise ValueError( 'use_cache must be True for DeCAN models' )

        # Enforce only one of input_ids and inputs_embeds
        if ( input_ids is None ) ^ ( inputs_embeds is not None ):
            raise ValueError( 'You must specify exactly one of input_ids or inputs_embeds' )

        # If past_key_values is None we create a new dynamic cache
        if past_key_values is None and use_cache:
            past_key_values = DeCANDynamicCache()

        # Otherwise if a cache was passed it must be of the correct type # TODO: auto cache casting?
        if not isinstance( past_key_values, ( DeCANTrainingCache, DeCANDynamicCache ) ):
            raise ValueError( 'past_key_values must be a DeCAN-specific cache type' )

        # If input ids was specified, embed and project to embeddings
        if input_ids is not None:
            inputs_embeds = self.ids_to_embeddings( input_ids )
        elif TYPE_CHECKING:
            assert inputs_embeds is not None

        # Normalise inputs
        hidden_states = self.input_norm( inputs_embeds )

        # Create position embeddings to be shared across the decoder layers and compute the causal mask
        position_embeddings, causal_mask = self.embed_rotary( hidden_states, document_ids, past_key_values )

        # Model outputs
        all_hidden_states = () if output_hidden_states else None
        all_attention_weights = () if output_attentions else None
        next_decoder_cache = None

        # Decoder layers
        for decoder_layer in self.layers:

            # Append hidden_states if enabled
            if all_hidden_states is not None:
                all_hidden_states += ( hidden_states, )

            hidden_states, attention_weights, present_key_values = decoder_layer(
                hidden_states=hidden_states,
                past_key_values=past_key_values,
                attention_mask=causal_mask,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
            )

            # Set next cache if enabled (should be)
            if use_cache:
                next_decoder_cache = present_key_values

            # Append attention_weights if enabled
            if all_attention_weights is not None:
                all_attention_weights += ( attention_weights, )

        # Perform final layer norm
        hidden_states = self.final_norm( hidden_states )

        # Append final hidden_states if enabled
        if all_hidden_states is not None:
            all_hidden_states += ( hidden_states, )

        # Set final cache reference (for some reason)
        next_cache = next_decoder_cache

        if not return_dict:
            return tuple( v for v in [ hidden_states, next_cache, all_hidden_states, all_attention_weights ] if v is not None )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attention_weights,
        )

class DeCANForCausalLM( DeCANPreTrainedModel, GenerationMixin ):
    _tied_weights_keys = [ 'lm_head.weight' ]

    def __init__( self, config: DeCANConfig ):
        super().__init__( config )
        self.model = DeCANModel( config )
        self.vocab_size = config.vocab_size

        self.final_proj = nn.Linear( config.hidden_size, config.vocab_dim, bias=False )
        self.lm_head = nn.Linear( config.vocab_dim, config.vocab_size, bias=False )

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings( self ):
        return self.model.get_input_embeddings()

    def set_input_embeddings( self, value ):
        self.model.set_input_embeddings( value )

    def get_position_embeddings( self ):
        raise NotImplementedError( 'DeCAN uses RoPE and therefor does not have position embeddings' )

    def resize_position_embeddings( self, new_num_position_embeddings: int ):
        raise NotImplementedError( 'DeCAN uses RoPE and therefor does not have position embeddings' )

    def get_output_embeddings( self ):
        return self.lm_head

    def set_output_embeddings( self, new_embeddings ):
        self.lm_head = new_embeddings

    def set_decoder( self, decoder ):
        self.model = decoder

    def get_decoder( self ):
        return self.model

    def ids_to_embeddings( self, input_ids: torch.Tensor ) -> torch.Tensor:
        """ Computes embeddings from input ids using the embed_tokens and input_proj

        Args:
            input_ids (torch.Tensor): Long tensor of shape [batch_size, seq_length]

        Returns:
            torch.Tensor: Projected embeddings of shape [batch_size, seq_length, hidden_size]
        """
        return self.model.ids_to_embeddings( input_ids )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        document_ids: torch.Tensor | None = None,
        # TODO: consider attention_mask
        # TODO: consider cache_position
        past_key_values: Cache | None = None,
        use_cache: bool | None = True,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        """ Decoder forward pass with LM head.

        Note: exactly one of `input_ids` or `inputs_embeds` must be passed.
        Note: if document_ids are passed you must continue to pass them or the cache will break.

        Args:
            input_ids (torch.Tensor, optional): Input token sequence of shape [batch_size, seq_length].
            inputs_embeds (torch.Tensor, optional): Input embeddings of shape [batch_size, seq_length, hidden_size].
            document_ids (torch.Tensor, optional): Document indices of shape [batch_size, seq_length], enables document masking.
            past_key_values (Cache, optional): KV cache container. Must be a DeCAN specific cache instance or None.
            use_cache (bool, optional): Enables use of the KV cache container. This MUST be True. Defaults to True.
            output_attentions (bool, optional): Whether or not to return the attentions tensors of all attention layers. Currently not supported. Defaults to False.
            output_hidden_states (bool, optional): Whether or not to return the hidden states of all layers.
            return_dict (bool, optional): Whether or not to return a `CausalLMOutputWithPast` instead of a plain tuple.
        """
        config: DeCANConfig = self.config # type: ignore
        
        use_cache = config.use_cache if use_cache is None else use_cache
        return_dict = config.return_dict if return_dict is None else return_dict
        output_attentions = config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = config.output_hidden_states if output_hidden_states is None else output_hidden_states

        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            document_ids=document_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Get final hidden states from decoder model
        hidden_states = outputs[0]

        # Perform final projection and compute logits
        hidden_states = self.final_proj( hidden_states )

        if not self.config.tie_word_embeddings:
            logits = self.lm_head( hidden_states )
        else:
            logits = F.linear( hidden_states, self.model.embed_tokens.weight )

        if not return_dict:
            return ( logits, ) + outputs[ 1 : ]

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        attention_mask: torch.Tensor | None = None, # TODO: consider attention_mask
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None, # TODO: consider cache_position
        **kwargs
    ):
        # Determine if cache is empty
        empty_cache = past_key_values is None or past_key_values.get_seq_length() == 0

        # If the cache is not only keep last input id
        if not empty_cache:
            input_ids = input_ids[ :, -1 : ]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and empty_cache:
            model_inputs: dict = { 'inputs_embeds': inputs_embeds }
        else:
            model_inputs: dict = { 'input_ids': input_ids }

        model_inputs.update( {
            'past_key_values': past_key_values,
            'use_cache': True,
        } )
        return model_inputs

    def _reorder_cache( self, past_key_values: Cache, beam_idx: torch.LongTensor ):
        past_key_values.reorder_cache( beam_idx )
        return past_key_values
