""" Utils module for the DeCAN model """

import os
import torch
from transformers import AutoTokenizer, AddedToken, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from transformers import AutoModelForCausalLM, PreTrainedModel
from tokenizers.processors import TemplateProcessing

def load_tokenizer( cache_dir: str | None = None, separate_bos_eos=False ) -> PreTrainedTokenizerBase:
    """ Loads the OPT tokenizer and modifies it for use with DeCAN

    Args:
        cache_dir (str, optional): Cache directory to store tokenizer. If `None` uses the `HF_CACHE_DIR` envar. Defaults to None.
        separate_bos_eos (bool): Determines if BOS/EOS are shared (False) ore separate (True). Defaults to False.

    Returns:
        PreTrainedTokenizerBase: Modified GPT2TokenizerFast using OPT vocabulary, and with DeCAN's special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/opt-125m',
        cache_dir=cache_dir or os.environ[ 'HF_CACHE_DIR' ],
        use_fast=True,
        bos_token='<s>' if separate_bos_eos else '</s>',
        # sep_token=AddedToken( '<|im_start|>', rstrip=False, lstrip=False, single_word=False, normalized=True, special=True ),
        # cls_token=AddedToken( '<|im_end|>', rstrip=False, lstrip=False, single_word=False, normalized=True, special=True ),
        # add_eos_token=True,
    )

    assert isinstance( tokenizer, PreTrainedTokenizerFast )

    # tokenizer._tokenizer.post_processor = TemplateProcessing( # type: ignore # pylint: disable=W0212
    #     single=f'{tokenizer.bos_token}:0 $A:0 {tokenizer.eos_token}:0',
    #     pair=f'{tokenizer.bos_token}:0 $A:0 {tokenizer.eos_token}:0 {tokenizer.bos_token}:1 $B:1 {tokenizer.eos_token}:1',
    #     special_tokens=[
    #         ( tokenizer.bos_token, tokenizer.bos_token_id ),
    #         ( tokenizer.eos_token, tokenizer.eos_token_id ),
    #     ]
    # )

    return tokenizer

def load_sft_tokenizer( original_tokenizer: PreTrainedTokenizerBase, cache_dir: str | None = None ) -> PreTrainedTokenizerBase:
    """ Loads the OPT tokenizer and modifies it for use with finetuning DeCAN

    Args:
        original_tokenizer (PreTrainedTokenizerBase): The original tokenizer used by the pre-trained model.
        cache_dir (str, optional): Cache directory to store tokenizer. If `None` uses the `HF_CACHE_DIR` envar. Defaults to None.

    Returns:
        PreTrainedTokenizerBase: Modified GPT2TokenizerFast using OPT vocabulary, and with DeCAN's special tokens.
    """

    # If the original tokenizer has a chat template we can use it as is
    if original_tokenizer.chat_template is not None:
        return original_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        'facebook/opt-125m',
        cache_dir=cache_dir or os.environ[ 'HF_CACHE_DIR' ],
        bos_token=original_tokenizer.bos_token,
        sep_token=AddedToken( '<|im_start|>', rstrip=False, lstrip=False, single_word=False, normalized=True, special=True ),
        cls_token=AddedToken( '<|im_end|>', rstrip=False, lstrip=False, single_word=False, normalized=True, special=True ),
        add_bos_token=False,
        use_fast=True,
    )

    assert isinstance( tokenizer, PreTrainedTokenizerFast )

    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant<|im_end|>\n' }}{% endif %}"
        "{% if (message['role'] != 'assistant') %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% elif (message['role'] == 'assistant')%}{{'<|im_start|>' + message['role'] + '\n' }}{% generation %}{{message['content'] + '<|im_end|>'}}{% endgeneration %}{{'\n'}}"
        "{% endif %}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    return tokenizer


@torch.no_grad
def set_pretrained_embeddings(
    model: PreTrainedModel,
    source_model_path: str = 'facebook/opt-125m',
    cache_dir: str | None = None,
):
    """ Overwrites the trainable embeddings from a source model.

    Args:
        model (PreTrainedModel): Model to write the embeddings to.
        source_model_path (str, optional): Path of model to retrieve embeddings from. Defaults to 'facebook/opt-125m'.
        cache_dir (str, optional): Cache directory to store tokenizer. If `None` uses the `HF_CACHE_DIR` envar. Defaults to None.
    """

    # Load source model onto CPU
    source_model = AutoModelForCausalLM.from_pretrained(
        source_model_path,
        cache_dir=cache_dir or os.environ[ 'HF_CACHE_DIR' ],
    )

    # Grab the embeddings data
    src_embeddings: torch.nn.Parameter = source_model.get_input_embeddings().weight
    dst_embeddings: torch.nn.Parameter = model.get_input_embeddings().weight

    # Copy embeddings inplace from source to destination
    dst_embeddings.copy_( src_embeddings )
