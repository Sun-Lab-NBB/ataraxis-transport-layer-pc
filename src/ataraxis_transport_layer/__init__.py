"""This library provides classes and methods that enable bidirectional communication between project Ataraxis systems.

See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer for more details.
API documentation: https://ataraxis-transport-layer-api-docs.netlify.app/
Author: Ivan Kondratyev (Inkaros)
"""

from .helper_modules import CRCProcessor, COBSProcessor
from .microcontroller import MicroControllerInterface
from .transport_layer import SerialTransportLayer, list_available_ports
from .custom_interfaces import TTLInterface, LickInterface, BreakInterface, ValveInterface, EncoderInterface

__all__ = [
    "BreakInterface",
    "COBSProcessor",
    "CRCProcessor",
    "EncoderInterface",
    "MicroControllerInterface",
    "LickInterface",
    "SerialTransportLayer",
    "TTLInterface",
    "ValveInterface",
    "list_available_ports",
]
