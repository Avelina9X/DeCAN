
import os
import torch
from transformers import AutoTokenizer, AddedToken, PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, PreTrainedModel

def load_tokenizer( cache_dir: str | None = None ) -> PreTrainedTokenizerBase:
    """ Loads the OPT tokenizer and modifies it for use with DeCAN

    Args:
        cache_dir (str, optional): Cache directory to store tokenizer. If `None` uses the `HF_CACHE_DIR` envar. Defaults to None.

    Returns:
        PreTrainedTokenizerBase: Modified GPT2TokenizerFast using OPT vocabulary, and with DeCAN's special tokens.
    """
    return AutoTokenizer.from_pretrained(
        'facebook/opt-125m',
        cache_dir=cache_dir or os.environ[ 'HF_CACHE_DIR' ],
        use_fast=True,
        bos_token='<s>',
        sep_token=AddedToken( '<|im_start|>', rstrip=False, lstrip=False, single_word=False, normalized=True, special=True ),
        cls_token=AddedToken( '<|im_end|>', rstrip=False, lstrip=False, single_word=False, normalized=True, special=True ),
    )


@torch.no_grad
def set_pretrained_embeddings(
    model: PreTrainedModel,
    source_model_path: str = 'facebook/opt-125m',
    cache_dir: str | None = None,
):
    """_summary_

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
    
    