"""This library provides classes and methods that enable bidirectional communication between project Ataraxis systems.

See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer for more details.
API documentation: https://ataraxis-transport-layer-api-docs.netlify.app/
Author: Ivan Kondratyev (Inkaros)
"""

from .helper_modules import CRCProcessor, COBSProcessor
from .transport_layer import SerialTransportLayer

__all__ = ["SerialTransportLayer", "CRCProcessor", "COBSProcessor"]
