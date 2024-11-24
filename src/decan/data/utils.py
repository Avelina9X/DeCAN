""" Utils module for data """

from abc import ABC, abstractmethod
from typing import Any, Mapping

class SerializableMixin( ABC ):
    """ Adds basic serialisation support to any class """

    @abstractmethod
    def state_dict( self ) -> Mapping[str, Any]:
        ...

    @abstractmethod
    def load_state_dict( self, state_dict: Mapping[str, Any] ):
        ...
