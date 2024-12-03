"""This library provides classes and methods that enable bidirectional communication between project Ataraxis systems.

See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer for more details.
API documentation: https://ataraxis-transport-layer-api-docs.netlify.app/
Author: Ivan Kondratyev (Inkaros)
"""

from .custom_modules import TTLModule, BreakModule, ValveModule, SensorModule, EncoderModule
from .helper_modules import CRCProcessor, COBSProcessor
from .microcontroller import MicroControllerInterface
from .transport_layer import SerialTransportLayer, list_available_ports

__all__ = [
    "SerialTransportLayer",
    "CRCProcessor",
    "COBSProcessor",
    "list_available_ports",
    "MicroControllerInterface",
    "TTLModule",
    "EncoderModule",
    "BreakModule",
    "SensorModule",
    "ValveModule",
]
