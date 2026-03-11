"""Provides methods for establishing and maintaining bidirectional communication with Arduino and Teensy
microcontrollers over USB and UART serial interfaces.

See the `documentation <https://ataraxis-transport-layer-pc-api-docs.netlify.app/>`_ for the description of available
assets. See the `source code repository <https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc>`_ for more details.

Authors: Ivan Kondratyev (Inkaros), Katlynn Ryu.
"""

from .helper_modules import CRCProcessor, COBSProcessor
from .transport_layer import (
    TransportLayer,
    TransportLayerStatus,
    list_available_ports,
    print_available_ports,
)

__all__ = [
    "COBSProcessor",
    "CRCProcessor",
    "TransportLayer",
    "TransportLayerStatus",
    "list_available_ports",
    "print_available_ports",
]
