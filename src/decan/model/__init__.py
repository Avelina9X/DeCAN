"""
DeCAN modeling package.

Only configuration_decan.py and modeling_decan.py will be exported to HF.

Any other modules in this package are purely to assist with training and inference,
but are NOT required for full model functionality.
"""

from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
)

_import_structure = {
    'configuration_decan': [ 'DeCANConfig' ]
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure[ 'modeling_decan' ] = [
        'DeCANPreTrainedModel',
        'DeCANModel',
        'DeCANForCausalLM',
    ]

if TYPE_CHECKING:
    from .configuration_decan import DeCANConfig

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_decan import (
            DeCANPreTrainedModel,
            DeCANModel,
            DeCANForCausalLM,
        )
else:
    import sys

    sys.modules[ __name__ ] = _LazyModule( __name__, globals()[ '__file__' ], _import_structure, module_spec=__spec__ )
