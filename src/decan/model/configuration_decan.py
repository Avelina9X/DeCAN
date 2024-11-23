""" DeCAN model configuration """

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger( __name__ )

class DeCANConfig( PretrainedConfig ):
    model_type = 'decan'
    keys_to_ignore_at_inference = [ 'past_key_values' ]

    def __init__(
        self,
        vocab_size: int = 50272,
        vocab_dim: int = 768,
        hidden_size: int = 768,
        intermediate_size: int = 2048,
        intermediate_act: str = 'silu',
        num_hidden_layers: int = 12,
        num_key_value_heads: int = 1,
        num_attention_heads: int = 12,
        head_dim: int = 64,
        max_position_embeddings: int = 8192,
        rope_theta: int = 500000,
        rope_scaling: dict | None = None,
        tie_word_embeddings: bool = True,
        attention_bias: bool = True,
        mlp_bias: bool = True,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        sep_token_id: int = 50265,
        cls_token_id: int = 50266,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.vocab_dim = vocab_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.intermediate_act = intermediate_act
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {}
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        
        # Must set here rather than in super().__init__
        self.cls_token_id = cls_token_id

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            sep_token_id=sep_token_id,
            use_bfloat16=True, # this is a hack to enable bf16 mixed precision training
            **kwargs
        )