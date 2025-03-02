# DeCAN: Densely Connected Attention Networks
Taking inspiration from Densely Connected Convolution Networks, the Densely Connected Attention network (DeCAN) architecture introduces skip connections to the multi-head attention layers to encourage multi-level feature re-use and elicit the learning of high-rank key-value representations.

## Environment Variables
There are some __**required**__ environment variables to correctly utilise this library:
- `WANDB_API_KEY` - the WandB API key used for logging runs.
- `HF_CACHE_DIR` - the cache directory for **SFT** and **validation** datasets used during training. *Note: should be persistent storage.*
- `HF_TEMP_DIR` - the cache directory for **pretraining** dataset shards used during training. *Note: can be temporary storage.*
- `HF_DATASETS_TRUST_REMOTE_CODE=true` - allows LM Eval Harness to load older benchmarks.
- `PYTORCH_JIT=0` - disables broken JIT checks in some Torch versions.


There are also some __**recommended**__ environment variables which improve QoL and may break default configs if not set:
- `WANDB_PROJECT_NAME` - the WandB project name to log runs to. *Note: used by string templates in configs.*
- `CHECKPOINT_DIR` - default directory used for checkpoints. *Note: used by string templates in configs.*
- `HF_HOME` - specifies the Hugging Face cache directory for LM Eval Harness datasets used during **validation** and **evaluation**. *Note: if unset will use default dir.*
- `LM_HARNESS_CACHE_PATH` - directory to cache formatted query strings for the LM Eval Harness. *Note: speeds up evaluation when set.*