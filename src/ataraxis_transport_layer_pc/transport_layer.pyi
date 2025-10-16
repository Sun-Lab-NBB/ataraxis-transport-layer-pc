from enum import IntEnum
from typing import Any

import numpy as np
from serial import Serial
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from serial.tools.list_ports_common import ListPortInfo

from .helper_modules import (
    SerialMock as SerialMock,
    CRCProcessor as CRCProcessor,
    COBSProcessor as COBSProcessor,
    _CRCProcessor as _CRCProcessor,
    _COBSProcessor as _COBSProcessor,
)

_ZERO: Incomplete
_POLYNOMIAL: Incomplete
_EMPTY_ARRAY: Incomplete
type CRCType = np.uint8 | np.uint16 | np.uint32

class TransportLayerStatus(IntEnum):
    INSUFFICIENT_BUFFER_SPACE_ERROR = -1
    MULTIDIMENSIONAL_ARRAY_ERROR = -2
    EMPTY_ARRAY_ERROR = -3
    PACKET_SIZE_UNKNOWN = 0
    PACKET_PARSED = 1
    NOT_ENOUGH_PACKET_BYTES = 2
    NOT_ENOUGH_CRC_BYTES = 3
    NO_BYTES_TO_READ = 4
    PAYLOAD_SIZE_MISMATCH = 5
    DELIMITER_FOUND_TOO_EARLY = 6
    DELIMITER_NOT_FOUND = 7

def list_available_ports() -> tuple[ListPortInfo, ...]: ...
def print_available_ports() -> None: ...

class TransportLayer:
    _accepted_numpy_scalars: tuple[
        type[np.uint8],
        type[np.uint16],
        type[np.uint32],
        type[np.uint64],
        type[np.int8],
        type[np.int16],
        type[np.int32],
        type[np.int64],
        type[np.float32],
        type[np.float64],
        type[np.bool],
    ]
    _opened: bool
    _port: SerialMock | Serial
    _crc_processor: Incomplete
    _cobs_processor: Incomplete
    _timer: Incomplete
    _start_byte: np.uint8
    _delimiter_byte: np.uint8
    _timeout: int
    _postamble_size: np.uint8
    _max_tx_payload_size: np.uint8
    _max_rx_payload_size: np.uint8
    _min_rx_payload_size: np.uint8
    _transmission_buffer: NDArray[np.uint8]
    _reception_buffer: NDArray[np.uint8]
    _minimum_packet_size: int
    _bytes_in_transmission_buffer: int
    _bytes_in_reception_buffer: int
    _consumed_bytes: int
    _leftover_bytes: bytes
    def __init__(
        self,
        port: str,
        microcontroller_serial_buffer_size: int,
        baudrate: int,
        polynomial: CRCType = ...,
        initial_crc_value: CRCType = ...,
        final_crc_xor_value: CRCType = ...,
        *,
        test_mode: bool = False,
    ) -> None: ...
    def __del__(self) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def available(self) -> bool: ...
    @property
    def transmission_buffer(self) -> NDArray[np.uint8]: ...
    @property
    def reception_buffer(self) -> NDArray[np.uint8]: ...
    @property
    def bytes_in_transmission_buffer(self) -> int: ...
    @property
    def bytes_in_reception_buffer(self) -> int: ...
    def reset_transmission_buffer(self) -> None: ...
    def reset_reception_buffer(self) -> None: ...
    def write_data(self, data_object: Any) -> None: ...
    @staticmethod
    def _write_scalar_data(target_buffer: NDArray[np.uint8], scalar_object: Any, start_index: int) -> int: ...
    @staticmethod
    def _write_array_data(target_buffer: NDArray[np.uint8], array_object: NDArray[Any], start_index: int) -> int: ...
    def read_data(self, data_object: Any) -> Any: ...
    @staticmethod
    def _read_array_data(
        source_buffer: NDArray[np.uint8], array_object: NDArray[Any], start_index: int, payload_size: int
    ) -> tuple[NDArray[Any], int]: ...
    def send_data(self) -> None: ...
    @staticmethod
    def _construct_packet(
        payload_buffer: NDArray[np.uint8],
        cobs_processor: _COBSProcessor,
        crc_processor: _CRCProcessor,
        payload_size: int,
        start_byte: np.uint8,
    ) -> NDArray[np.uint8]: ...
    def receive_data(self) -> bool: ...
    def _receive_packet(self) -> bool: ...
    def _bytes_available(self, required_bytes_count: int = 1, timeout: int = 0) -> bool: ...
    @staticmethod
    def _parse_packet(
        unparsed_bytes: NDArray[np.uint8],
        start_byte: np.uint8,
        delimiter_byte: np.uint8,
        max_payload_size: np.uint8,
        min_payload_size: np.uint8,
        postamble_size: np.uint8,
        start_found: bool = False,
        parsed_byte_count: int = 0,
        parsed_bytes: NDArray[np.uint8] = ...,
    ) -> tuple[int, int, NDArray[np.uint8], NDArray[np.uint8]]: ...
    @staticmethod
    def _process_packet(
        reception_buffer: NDArray[np.uint8],
        packet_size: int,
        cobs_processor: _COBSProcessor,
        crc_processor: _CRCProcessor,
    ) -> int: ...
