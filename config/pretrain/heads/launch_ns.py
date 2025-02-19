import os
import subprocess
import itertools

import rich
import tqdm

# subprocess.call( [
#     'python3', 'src/decan/__main__.py', 'pretrain', 'dummy',
#     '--model-config', './config/pretrain/heads/universal.yaml',
#     '--trainer-config', './config/pretrain/heads/universal.yaml',
#     '--run-name', 'hexp_Vanilla-MHA',
#     '--model-kwargs', 'num_key_value_heads=12',
#     '--cpu-offload-cache', 'true',
# ] )

INIT_SEED = 42

def get_args( variant: str, exp_type: str | None = None, exp_init: str | None = None, smoothing: bool | None = None ):
    match variant:
        case 'mha':
            assert exp_type is None
            assert exp_init is None
            assert smoothing is None
            
            run_name = f'hexp{INIT_SEED}NS_Vanilla-MHA'
            
            num_key_value_heads = 12
            head_expansion = None
            frozen = None
            cpu_offload_cache = 'True'
            tags = [ 'baseline', 'mha' ]
            
        case 'mqa':
            assert exp_type is None
            assert exp_init is None
            assert smoothing is None
            
            run_name = f'hexp{INIT_SEED}NS_Vanilla-MQA'
            
            num_key_value_heads = 1
            head_expansion = [ 'exp_type=scalar', 'exp_init=mqa', 'head_smoothing=0.0' ]
            frozen = [ 'k_exp', 'v_exp' ]
            cpu_offload_cache = 'False'
            tags = [ 'baseline', 'mqa' ]
        
        case 'decan':
            assert exp_type is None
            assert exp_init is None
            assert smoothing is None
            
            run_name = f'hexp{INIT_SEED}NS_DeCAN'
            
            num_key_value_heads = 1
            head_expansion = None
            frozen = None
            cpu_offload_cache = 'False'
            tags = [ 'baseline', 'decan' ]
        
        case 'decan+':
            assert exp_type in [ 'S', 'V', 'M' ]
            assert exp_init in [ 'base', 'hybrid', 'mqa' ]
            assert smoothing is not None
            
            run_name = f'hexp{INIT_SEED}NS_DeCAN_{exp_type}_{exp_init}'
            if smoothing:
                run_name += '_HS'
                
            exp_type = { 'S': 'scalar', 'V': 'vector', 'M': 'matrix' }[ exp_type ]
            smoothing = 0.1 if smoothing else 0.0
            
            num_key_value_heads = 1
            head_expansion = [ f'exp_type={exp_type}', f'exp_init={exp_init}', f'head_smoothing={smoothing}' ]
            frozen = None
            cpu_offload_cache = 'False'
            tags = [ 'decan', 'decan+' ]
        
        case _:
            assert False, 'Wrong variant!'
    
    args = []
    args += [ '--run-name', run_name ]
    args += [ '--model-config', './config/pretrain/heads/universal.yaml' ]
    args += [ '--trainer-config', './config/pretrain/heads/universal.yaml' ]
    args += [ '--wandb-tags' ] + tags
    args += [ '--model-kwargs', f'num_key_value_heads={num_key_value_heads}', 'rms_norm_scaling=True' ]
    
    if head_expansion is not None:
        args += [ '--model-kwargs-hexp' ] + head_expansion
    
    if frozen is not None:
        args += [ '--frozen-params' ] + frozen
    
    args += [ '--cpu-offload-cache', cpu_offload_cache ]
    args += [ '--set-init-seed', str( INIT_SEED ) ]
    args += [ '--manifest-file', f'runs{INIT_SEED}NS.yaml' ]
    
    return args

variant = [ 'decan+' ]
exp_types = [ 'S', 'V', 'M' ]
exp_inits = [ 'base', 'hybrid', 'mqa' ]
smoothings = [ False, True ]

combos = list( itertools.product( variant, exp_types, exp_inits, smoothings ) )
combos = [ ( 'mha', ), ( 'mqa', ), ( 'decan', ) ] + combos

combo_args = [ get_args( *args ) for args in combos ]

if input( 'Type `skip` to skip checking the run configs.' ) != 'skip':
    print( 'Checking run configs...' )
    for args in tqdm.tqdm( combo_args ):
        ret_code = subprocess.call( [ 'python3', 'src/decan/__main__.py', 'pretrain', 'dummy' ] + args, stdout=subprocess.DEVNULL )
        
        if ret_code != 0:
            raise ValueError( f'Failed dummy test for: {args}' )
    print( 'Done!' )
    print()
    input( 'Press any key to set up runs, or Ctrl-C to cancel!' )
print()
print( 'Setting up real runs!' )
for args in tqdm.tqdm( combo_args ):
    ret_code = subprocess.call( [ 'python3', 'src/decan/__main__.py', 'pretrain', 'setup' ] + args, stdout=subprocess.DEVNULL )
    
    if ret_code != 0:
        raise ValueError( f'Failed setup for: {args}\nAborting!' )
print( 'Done!' )