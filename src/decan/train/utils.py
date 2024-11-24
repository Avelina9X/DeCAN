""" Utils module for training """

import dataclasses
from typing import Any, Callable, Optional

def attr(
    *,
    doc: Optional[str] = None,
    track: bool = True,
    aliases: Optional[str | list[str]] = None,
    default: Any | dataclasses._MISSING_TYPE = dataclasses.MISSING,
    default_factory: Callable[[], Any] | dataclasses._MISSING_TYPE = dataclasses.MISSING,
    **kwargs,
) -> dataclasses.Field:
    metadata = {
        'doc': doc,
        'track': track,
        'aliases': aliases,
    }
    
    return dataclasses.field( # type: ignore # pylint: disable=E3701
        metadata=metadata,
        default=default,
        default_factory=default_factory,
        **kwargs
    )
