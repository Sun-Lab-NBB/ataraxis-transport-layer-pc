"""This file stores the SerializedTransferProtocol class, which provides the high-level API that encapsulates all
methods necessary to bidirectionally communicate with microcontroller devices running the C-version of this library.
Recently, the class has been updated to also support ZeroMQ-based communication with non-microcontroller devices
running the C- or Python - version of this library, making it a universal communication protocol that can connect most
devices frequently used in science applications. All features of the class are available through 4 main methods:
write_data(), send_data(), receive_data() and read_data(). See method and class docstrings for more information.
"""

from typing import Any, Type, Union, Optional
import textwrap
from dataclasses import fields, is_dataclass

from numba import njit  # type: ignore
import numpy as np
from serial import Serial
from numpy.typing import NDArray
from serial.tools import list_ports
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import console

# noinspection PyProtectedMember
from ataraxis_transport_layer.helper_modules import (
    SerialMock,
    CRCProcessor,
    COBSProcessor,
    _CRCProcessor,
    _COBSProcessor,
)


class SerialTransportLayer:
    """Provides methods to bidirectionally communicate with Microcontrollers running the C++ version of the
    TransportLayer class over the UART or USB Serial interface.

    This class functions as a central hub that calls various internal and external helper classes and fully encapsulates
    the serial port interface (via pySerial third-party library). Most of this class is hidden behind private attributes
    and methods, and any part of the class that is publicly exposed is generally safe to use and should be enough
    to realize the full functionality of the library.

    Notes:
        This class contains 4 main methods: write_data(), send_data(), receive_data() and read_data(). Write and read
        methods are used to manipulate the class-specific 'staging' buffers that aggregate the data to be sent to the
        Microcontroller and store the data received from the Microcontroller. Send and receive methods operate on the
        class buffers and trigger the sequences of steps needed to construct and send a serial packet to the controller
        or receive and decode the data sent as a packet from the controller.

        Most class inputs and arguments are configured to require a numpy scalar or array input to enforce typing,
        which is not done natively in python. Type enforcement is notably 'unpythonic', but very important for this
        library as it communicates with Microcontrollers that do use a strictly typed language (C++). Additionally,
        enforcing typing allows using efficient numpy and numba operations to optimize most of the custom library code
        to run at C speeds, which is one of the notable advantages of this library.

    Args:
        port: The name of the serial port to connect to, e.g.: 'COM3' or '/dev/ttyUSB0'. You can use the
            list_available_ports() class method to get a list of discoverable serial port names.
        baudrate: The baudrate to be used to communicate with the Microcontroller. Should match the value used by
            the microcontroller for UART ports, ignored for USB ports. Note, the appropriate baudrate for any UART-using
            controller partially depends on its CPU clock!
        polynomial: The polynomial to use for the generation of the CRC lookup table. Can be provided as a HEX
            number (e.g., 0x1021). Currently only non-reversed polynomials of numpy uint8, uint16, and uint32
            datatype are supported.
        initial_crc_value: The initial value to which the CRC checksum variable is initialized during calculation.
            This value depends on the chosen polynomial algorithm and should use the same datatype as the 'polynomial'
            argument. It can be provided as a HEX number (e.g., 0xFFFF).
        final_crc_xor_value: The final XOR value to be applied to the calculated CRC checksum value. This value
            depends on the chosen polynomial algorithm and should use the same datatype as the 'polynomial' argument.
            It can be provided as a HEX number (e.g., 0x0000).
        maximum_transmitted_payload_size: The maximum number of bytes that are expected to be transmitted to the
            Microcontroller as a single payload. This has to match the maximum_received_payload_size value used by
            the Microcontroller. Due to COBS encoding, this value has to be between 1 and 254 bytes.
        minimum_received_payload_size: The minimum number of bytes that are expected to be received from the
            Microcontroller as a single payload. This number is used to calculate the threshold for entering
            incoming data reception cycle. In turn, this is used to minimize the number of calls made to costly
            methods required to receive data. Due to COBS encoding, this value has to be between 1 and 254 bytes.
        start_byte: The value used to mark the beginning of the packet. Has to match the value used by the
            Microcontroller. Can be any value in the uint8 range (0 to 255). It is advised to use the value that is
            unlikely to occur as noise.
        delimiter_byte: The value used to denote the end of the packet. Has to match the value used by the
            Microcontroller. Due to how COBS works, it is advised to use '0' as the delimiter byte. Zero is the only
            value guaranteed to be exclusive when used as a delimiter.
        timeout: The maximum number of microseconds that can separate receiving any two consecutive bytes of the
            packet. This is used to detect and resolve stale packet reception attempts. While this defaults to 20000
            (20 ms), the library can resolve intervals in the range of ~50-100 microseconds, so the number can
            be made considerably smaller than that.
        test_mode: Determines whether the library uses a real pySerial Stream class or a StreamMock class. Only used
            during testing and should always be disabled otherwise.
        allow_start_byte_errors: Determines whether the class raises errors when it is unable to find the start value
            in the incoming byte-stream. It is advised to keep this set to False for most use cases. This is because it
            is fairly common to see noise-generated bytes inside the reception buffer. These bytes are silently cleared
            by the reception algorithm until a real packet becomes available. However, enabling this option may be
            helpful for certain debugging scenarios.

    Attributes:
        _port: Depending on the test_mode flag, stores either a SerialMock or Serial object that provides serial port
            interface.
        _crc_processor: Stores the CRCProcessor class object that provides methods for working CRC checksums.
        _cobs_processor: Stores the COBSProcessor class object that provides methods for encoding and decoding
            transmitted payloads.
        _timer: Stores the PrecisionTimer class object that provides a microsecond-precise GIL-releasing timer.
        _start_byte: Stores the byte-value that marks the beginning of transmitted and received packets.
        _delimiter_byte: Stores the byte-value that marks the end of transmitted and received packets.
        _timeout: The number of microseconds to wait between receiving any two consecutive bytes of a packet.
        _allow_start_byte_errors: Determines whether to raise errors when the start_byte value is not found among the
            available bytes during receive_data() runtime.
        _max_tx_payload_size: Stores the maximum number of bytes that can be transmitted as a single payload. This value
            cannot exceed 254 bytes due to COBS encoding.
        _max_rx_payload_size: Stores the maximum number of bytes that can be received from the microcontroller as a
            single payload. This value cannot exceed 254 bytes due to COBS encoding.
        _postamble_size: Stores the byte-size of the CRC checksum.
        _transmission_buffer: The buffer used to stage the data to be sent to the Microcontroller.
        _reception_buffer: The buffer used to store the decoded data received from the Microcontroller.
        _bytes_in_transmission_buffer: Tracks how many bytes (relative to index 0) of the _transmission_buffer are
            currently used to store the payload to be transmitted.
        _bytes_in_reception_buffer: Same as _bytes_in_transmission_buffer, but for the _reception_buffer.
        _leftover_bytes: A buffer used to preserve any 'unconsumed' bytes that were read from the serial port
            but not used to reconstruct the payload sent from the Microcontroller. This is used to minimize the number
            of calls to pySerial methods, as they are costly to run.
        _accepted_numpy_scalars: Stores numpy types (classes) that can be used as scalar inputs or as 'dtype'
            fields of the numpy arrays that are provided to class methods. Currently, these are the only types
            supported by the library.
        _minimum_packet_size: Stores the minimum number of bytes that can represent a valid packet. This value is used
            to optimize packet reception logic.

        Raises:
            TypeError: If any of the input arguments is not of the expected type.
            ValueError: If any of the input arguments have invalid values.
            SerialException: If wrapped pySerial class runs into an error.
    """

    _accepted_numpy_scalars: tuple[
        Type[np.uint8],
        Type[np.uint16],
        Type[np.uint32],
        Type[np.uint64],
        Type[np.int8],
        Type[np.int16],
        Type[np.int32],
        Type[np.int64],
        Type[np.float32],
        Type[np.float64],
        Type[np.bool],
    ] = (
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
        np.bool,
    )  # Sets up a tuple of types used to verify transmitted data

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        polynomial: Union[np.uint8, np.uint16, np.uint32] = np.uint16(0x1021),
        initial_crc_value: Union[np.uint8, np.uint16, np.uint32] = np.uint16(0xFFFF),
        final_crc_xor_value: Union[np.uint8, np.uint16, np.uint32] = np.uint16(0x0000),
        maximum_transmitted_payload_size: int = 254,
        minimum_received_payload_size: int = 1,
        start_byte: int = 129,
        delimiter_byte: int = 0,
        timeout: int = 20000,
        *,
        test_mode: bool = False,
        allow_start_byte_errors: bool = False,
    ) -> None:
        # Verifies that input arguments are valid. Does not check polynomial parameters, that is offloaded to the
        # CRCProcessor class.
        if not isinstance(port, str):
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected a string value for 'port' argument, but "
                f"encountered {port} of type {type(port).__name__}."
            )
            console.error(message=message, error=TypeError)
        if baudrate <= 0:
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected a positive integer value for 'baudrate' "
                f"argument, but encountered {baudrate} of type {type(baudrate).__name__}."
            )
            console.error(message=message, error=ValueError)
        if not 0 <= start_byte <= 255:
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected an integer value between 0 and 255 for "
                f"'start_byte' argument, but encountered {start_byte} of type {type(start_byte).__name__}."
            )
            console.error(message=message, error=ValueError)
        if not 0 <= delimiter_byte <= 255:
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected an integer value between 0 and 255 for "
                f"'delimiter_byte' argument, but encountered {delimiter_byte} of type {type(delimiter_byte).__name__}."
            )
            console.error(message=message, error=ValueError)
        if timeout < 0:
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected an integer value of 0 or above for "
                f"'timeout' argument, but encountered {timeout} of type {type(timeout).__name__}."
            )
            console.error(message=message, error=ValueError)
        if start_byte == delimiter_byte:
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected 'start_byte' and 'delimiter_byte' "
                f"arguments to have different values, but both are set to the same value ({start_byte})."
            )
            console.error(message=message, error=ValueError)
        if not 0 < maximum_transmitted_payload_size <= 254:
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected an integer value between 1 and 254 for the "
                f"'maximum_transmitted_payload_size' argument, but encountered {maximum_transmitted_payload_size} "
                f"of type {type(maximum_transmitted_payload_size).__name__}."
            )
            console.error(message=message, error=ValueError)
        if not 0 < minimum_received_payload_size <= 254:
            message = (
                f"Unable to initialize SerialTransportLayer class. Expected an integer value between 1 and 254 for the "
                f"'minimum_received_payload_size' argument, but encountered {minimum_received_payload_size} "
                f"of type {type(minimum_received_payload_size).__name__}."
            )
            console.error(message=message, error=ValueError)

        # Based on the class runtime selector, initializes a real or mock serial port manager class
        self._port: SerialMock | Serial
        if not test_mode:
            # Statically disables built-in timeout. Our jit- and c-extension classes are more optimized for this job
            # than Serial's built-in timeout.
            self._port = Serial(port, baudrate, timeout=0)
        else:
            self._port = SerialMock()

        # This verifies input polynomial parameters at class initialization time
        self._crc_processor = CRCProcessor(polynomial, initial_crc_value, final_crc_xor_value)
        self._cobs_processor = COBSProcessor()

        # On very fast CPUs, the timer can be sub-microsecond precise. On older systems, this may not necessarily hold.
        # Either way, microsecond precision is safe for most target systems.
        self._timer = PrecisionTimer("us")

        # Initializes serial packet attributes and casts all to numpy types. With the checks above, there should be
        # no overflow or casting issues.
        self._start_byte: np.uint8 = np.uint8(start_byte)
        self._delimiter_byte: np.uint8 = np.uint8(delimiter_byte)
        self._timeout: np.uint64 = np.uint64(timeout)
        self._allow_start_byte_errors: bool = allow_start_byte_errors
        self._postamble_size: np.uint8 = self._crc_processor.crc_byte_length

        # Uses payload size arguments to initialize reception and transmission buffers.
        self._max_tx_payload_size: np.uint8 = np.uint8(maximum_transmitted_payload_size)
        self._max_rx_payload_size: np.uint8 = np.uint8(254)  # Statically capped at 254 due to COBS encoding

        # Buffer sizes are up-case to uint16, as they may need to exceed the 256-size limit. They include the respective
        # payload size, the postamble size (1 to 4 bytes) and 4 static bytes for the preamble and packet metadata.
        # These 4 bytes are: start_byte, delimiter_byte, overhead_byte, and packet_size byte.
        tx_buffer_size: np.uint16 = np.uint16(self._max_tx_payload_size) + 4 + np.uint16(self._postamble_size)
        rx_buffer_size: np.uint16 = np.uint16(self._max_rx_payload_size) + 4 + np.uint16(self._postamble_size)
        self._transmission_buffer: NDArray[np.uint8] = np.zeros(shape=tx_buffer_size, dtype=np.uint8)
        self._reception_buffer: NDArray[np.uint8] = np.empty(shape=rx_buffer_size, dtype=np.uint8)

        # Based on the minimum expected payload size, calculates the minimum number of bytes that can fully represent
        # a packet. This is sued to avoid costly pySerial calls unless there is a high chance that the call will return
        # a parsable packet.
        self._minimum_packet_size: int = max(1, minimum_received_payload_size) + 4 + int(self._postamble_size)

        # Sets up various tracker and temporary storage variables that supplement class runtime.
        self._bytes_in_transmission_buffer: int = 0
        self._bytes_in_reception_buffer: int = 0
        self._leftover_bytes: bytes = bytes()  # Placeholder, this is re-initialized as needed during data reception.

        # Opens (connects to) the serial port. Cycles closing and opening to ensure the port is opened,
        # non-graciously replacing whatever is using the port at the time of instantiating SerialTransportLayer class.
        # This non-safe procedure was implemented to avoid a frequent issue with Windows taking a long time to release
        # COM ports, preventing quick connection cycling.
        self._port.close()
        self._port.open()

    def __del__(self) -> None:
        """Ensures proper resource release prior to garbage-collecting class instance."""

        # Closes the port before deleting the class instance. Not strictly required, but helpful to ensure resources
        # are released
        self._port.close()

    def __repr__(self) -> str:
        """Returns a string representation of the SerialTransportLayer class instance."""
        if isinstance(self._port, Serial):
            representation_string = (
                f"SerialTransportLayer(port='{self._port.name}', baudrate={self._port.baudrate}, polynomial="
                f"{self._crc_processor.polynomial}, start_byte={self._start_byte}, "
                f"delimiter_byte={self._delimiter_byte}, timeout={self._timeout} us, "
                f"maximum_tx_payload_size = {self._max_tx_payload_size}, "
                f"maximum_rx_payload_size={self._max_rx_payload_size})"
            )
        else:
            representation_string = (
                f"SerialTransportLayer(port & baudrate=MOCKED, polynomial={self._crc_processor.polynomial}, "
                f"start_byte={self._start_byte}, delimiter_byte={self._delimiter_byte}, timeout={self._timeout} us, "
                f"maximum_tx_payload_size = {self._max_tx_payload_size}, "
                f"maximum_rx_payload_size={self._max_rx_payload_size})"
            )
        return representation_string

    @staticmethod
    def list_available_ports() -> tuple[dict[str, int | str | Any], ...]:
        """Provides the information about each serial port addressable through the pySerial library.

        This method is intended to be used for discovering and selecting the serial port 'names' to use with this
        class.

        Returns:
            A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
            port.

        """
        # Gets the list of port objects visible to the pySerial library.
        available_ports = list_ports.comports()

        # Prints the information about each port using terminal.
        information_list = [
            {"Name": port.name, "Device": port.device, "PID": port.pid, "Description": port.description}
            for port in available_ports
        ]

        return tuple(information_list)

    @property
    def available(self) -> bool:
        """Returns True if enough bytes are available from the serial port to justify attempting to receive a packet."""

        # in_waiting is twice as fast as using the read() method. The 'true' outcome of this check is capped at the
        # minimum packet size to minimize the chance of having to call read() more than once. The method counts the
        # bytes available for reading and left over from previous packet parsing operations.
        return self._port.in_waiting + len(self._leftover_bytes) > self._minimum_packet_size

    @property
    def transmission_buffer(self) -> NDArray[np.uint8]:
        """Returns a copy of the transmission buffer numpy array.

        This buffer stores the 'staged' data to be sent to the Microcontroller. Use this method to safely access the
        contents of the buffer in a snapshot fashion.
        """
        return self._transmission_buffer.copy()

    @property
    def reception_buffer(self) -> NDArray[np.uint8]:
        """Returns a copy of the reception buffer numpy array.

        This buffer stores the decoded data received from the Microcontroller. Use this method to safely access the
        contents of the buffer in a snapshot fashion.
        """
        return self._reception_buffer.copy()

    @property
    def bytes_in_transmission_buffer(self) -> int:
        """Returns the number of payload bytes stored inside the transmission_buffer."""
        return self._bytes_in_transmission_buffer

    @property
    def bytes_in_reception_buffer(self) -> int:
        """Returns the number of payload bytes stored inside the reception_buffer."""
        return self._bytes_in_reception_buffer

    def reset_transmission_buffer(self) -> None:
        """Resets the transmission buffer bytes tracker to 0.

        This does not physically alter the buffer in any way, but makes all data inside the buffer 'invalid'. This
        approach to 'resetting' the buffer by overwriting, rather than recreation, is chosen for higher memory
        efficiency and runtime speed.
        """
        self._bytes_in_transmission_buffer = 0

    def reset_reception_buffer(self) -> None:
        """Resets the reception buffer bytes tracker to 0.

        This does not physically alter the buffer in any way, but makes all data inside the buffer 'invalid'. This
        approach to 'resetting' the buffer by overwriting, rather than recreation, is chosen for higher memory
        efficiency and runtime speed.
        """
        self._bytes_in_reception_buffer = 0

    def write_data(
        self,
        data_object: Union[
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.bool,
            NDArray[
                Union[
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.float32,
                    np.float64,
                    np.bool,
                ]
            ],
            Type,
        ],
        start_index: Optional[int] = None,
    ) -> int:
        """Writes (serializes) the input data_object to the class transmission buffer, starting at the specified
        start_index.

        If the object is of valid type and the buffer has enough space to accommodate the object data, it will be
        converted to bytes and written to the buffer at the start_index. All bytes written via this method become part
        of the payload that will be sent to the Microcontroller when send_data() method is called.

        Notes:
            At this time, the method only works with valid numpy scalars and arrays, as well as python dataclasses
            entirely made out of valid numpy types. Using numpy rather than standard python types increases runtime
            speed (when combined with other optimization steps) and enforces strict typing (critical for Microcontroller
            communication).

            The method automatically updates the _bytes_in_transmission_buffer tracker if the write operation
            increases the total number of payload bytes stored inside the buffer. If the method is used to overwrite
            previously added, it will not update the tracker variable. The only way to reset the payload size is via
            calling the appropriate buffer reset method.

            The maximum runtime speed for this method is achieved when writing data as numpy arrays, which is optimized
            to a single write operation. The minimum runtime speed is achieved by writing dataclasses, as it involves
            looping over dataclass attributes. When writing dataclasses, all attributes will be serialized and written
            as a consecutive data block to the same portion of the buffer.

        Args:
            data_object: A numpy scalar or array object or a python dataclass made entirely out of valid numpy objects.
                Supported numpy types are: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64,
                and bool. Arrays have to be 1-dimensional and not empty to be supported.
            start_index: Optional. The index inside the transmission buffer (0 to 253) at which to start writing the
                data. If set to None, the method will automatically use the _bytes_in_transmission_buffer tracker value
                to append the data to the end of the already written payload

        Returns:
            The index inside the transmission buffer that immediately follows the last index of the buffer to
            which the data was written. This index can be used as the start_index input for chained write operation
            calls to iteratively write data to the buffer.

        Raises:
            TypeError: If the input object is not a supported numpy scalar, numpy array, or python dataclass.
            ValueError: Raised if writing the input object is not possible as that would require writing outside the
                transmission buffer boundaries. Also raised when multidimensional or empty numpy arrays are
                encountered.
        """

        end_index = -10  # Initializes to a specific negative value that is not a valid index or runtime error code

        # Resolves the start_index input, ensuring it is a valid integer value if start_index is left at the default
        # None value
        if start_index is None:
            start_index = self._bytes_in_transmission_buffer

        # If the input object is a supported numpy scalar, calls the scalar data writing method.
        if isinstance(data_object, self._accepted_numpy_scalars):
            end_index = self._write_scalar_data(self._transmission_buffer, data_object, start_index)

        # If the input object is a numpy array, first ensures that it's datatype matches one of the accepted scalar
        # numpy types and, if so, calls the array data writing method.
        elif isinstance(data_object, np.ndarray) and data_object.dtype in self._accepted_numpy_scalars:
            end_index = self._write_array_data(self._transmission_buffer, data_object, start_index)

        # If the input object is a python dataclass, iteratively loops over each field of the class and recursively
        # calls write_data() to write each attribute of the class to the buffer. This should support nested dataclasses
        # if needed. This implementation supports using this function for any dataclass that stores numpy scalars or
        # arrays, replicating the behavior of the Microcontroller TransportLayer class.
        elif is_dataclass(data_object):
            # Records the initial index before looping over class attributes
            local_index = start_index

            # Loops over each field (attribute) of the dataclass and writes it to the buffer
            # noinspection PyDataclass
            for field in fields(data_object):
                # Calls the write method recursively onto the value of each field
                data_value = getattr(data_object, field.name)
                local_index = self.write_data(data_object=data_value, start_index=local_index)

                # If any such call fails for any reason (as signified by the returned index not exceeding the
                # start_index), breaks the loop to handle the error below
                if local_index < start_index:
                    break

            # Once the loop is over (due to break or having processed all class fields), sets the end_index to the
            # final recorded local_index value
            end_index = local_index

        # Unsupported input type error
        else:
            message = (
                f"Failed to write the data to the transmission buffer. Encountered an unsupported input data_object "
                f"type ({type(data_object).__name__}). At this time, only the following numpy scalar or array "
                f"types are supported: {self._accepted_numpy_scalars}. Alternatively, a dataclass with all attributes "
                f"set to supported numpy scalar or array types is also supported."
            )
            console.error(message=message, error=TypeError)

        # If the end_index exceeds the start_index, that means that an appropriate write operation was executed
        # successfully. In that case, updates the _bytes_in_transmission_buffer tracker if necessary and returns the
        # end index to caller to indicate runtime success.
        if end_index > start_index:
            # Sets the _bytes_in_transmission_buffer tracker variable to the maximum of its current value and the
            # index that immediately follows the final index of the buffer that was overwritten with he input data.
            # This only increases the tracker value if write operation increased the size of the payload.
            self._bytes_in_transmission_buffer = max(self._bytes_in_transmission_buffer, end_index)
            return end_index  # Returns the end_index to support chained overwrite operations

        # If the index is set to code 0, that indicates that the buffer does not have space to accept the written data
        # starting at the start_index.
        if end_index == 0:
            message = (
                f"Failed to write the data to the transmission buffer. The transmission buffer does not have enough "
                f"space to write the data starting at the index {start_index}. Specifically, given the data size of "
                f"{data_object.nbytes} bytes, the required buffer size is {start_index + data_object.nbytes} bytes, "
                f"but the available size is {self._transmission_buffer.size} bytes."
            )
            console.error(message=message, error=ValueError)

        # If the index is set to code -1, that indicates that a multidimensional numpy array was provided as input,
        # but only flat arrays are allowed
        if end_index == -1:
            message = (
                f"Failed to write the data to the transmission buffer. Encountered a multidimensional numpy array with "
                f"{data_object.ndim} dimensions as input data_object. At this time, only one-dimensional (flat) "
                f"arrays are supported."
            )
            console.error(message=message, error=ValueError)

        # If the index is set to code -2, that indicates that an empty numpy array was provided as input, which does
        # not make sense and, therefore, is likely an error. Also, empty arrays are explicitly not valid in C/C++, so
        # this is also against language rules to provide them with an intention to send that data to Microcontroller
        # running C.
        if end_index == -2:
            message = (
                f"Failed to write the data to the transmission buffer. Encountered an empty (size 0) numpy array as "
                f"input data_object. Writing empty arrays is not supported."
            )
            console.error(message=message, error=ValueError)

        # If the end_index is not resolved properly, catches and raises a runtime error
        message = (
            f"Failed to write the data to the transmission buffer. Encountered an unknown error code ({end_index})"
            f"returned by the writer method."
        )
        console.error(message=message, error=RuntimeError)

    @staticmethod
    @njit(nogil=True, cache=True)
    def _write_scalar_data(
        target_buffer: NDArray[np.uint8],
        scalar_object: Union[
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.bool,
        ],
        start_index: int,
    ) -> int:
        """Converts the input numpy scalar to a sequence of bytes and writes it to the transmission buffer at the
        specified start_index.

        This method is not designed to be called directly. It should always be called through the write_data() method
        of the parent class.

        Args:
            target_buffer: The buffer to which the data will be written. This should be the _transmission_buffer array
                of the caller class.
            scalar_object: The scalar numpy object to be written to the transmission buffer. Can be any supported numpy
                scalar type.
            start_index: The index inside the transmission buffer (0 to 253) at which to start writing the data.

        Returns:
            The positive index inside the transmission buffer that immediately follows the last index of the buffer to
            which the data was written. Integer code 0, if the buffer does not have enough space to accommodate the data
            written at the start_index.
        """

        # Converts the input scalar to a byte array. This is mostly so that Numba can work with the data via the
        # service method calls below. Note, despite the input being scalar, the array object may have multiple elements.
        array_object = np.frombuffer(np.array([scalar_object]), dtype=np.uint8)  # scalar → array → byte array

        # Calculates the required space inside the buffer to store the data inserted at the start_index
        data_size = array_object.size * array_object.itemsize  # Size of each element * the number of elements
        required_size = start_index + data_size

        # If the space to store the data extends outside the available transmission_buffer boundaries, returns 0.
        if required_size > target_buffer.size:
            return 0

        # Writes the data to the buffer.
        target_buffer[start_index:required_size] = array_object

        # Returns the required_size, which incidentally also matches the index that immediately follows the last index
        # of the buffer that was overwritten with the input data.
        return required_size

    @staticmethod
    @njit(nogil=True, cache=True)
    def _write_array_data(
        target_buffer: NDArray[np.uint8],
        array_object: NDArray[
            Union[
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.float32,
                np.float64,
                np.bool,
            ]
        ],
        start_index: int,
    ) -> int:
        """Converts the input numpy array to a sequence of bytes and writes it to the transmission buffer at the
        specified start_index.

        This method is not designed to be called directly. It should always be called through the write_data() method
        of the parent class.

        Args:
            target_buffer: The buffer to which the data will be written. This should be the _transmission_buffer array
                of the caller class.
            array_object: The numpy array to be written to the transmission buffer. Currently, the method is designed to
                only work with one-dimensional arrays with a minimal size of 1 element. The array should be using one
                of the supported numpy scalar datatypes.
            start_index: The index inside the transmission buffer (0 to 253) at which to start writing the data.

        Returns:
            The positive index inside the transmission buffer that immediately follows the last index of the buffer to
            which the data was written. Integer code 0, if the buffer does not have enough space to accommodate the data
            written at the start_index. Integer code -1, if the input array object is not one-dimensional.
            Integer code -2, if the input array object is empty.
        """

        if array_object.ndim != 1:
            return -1  # Returns -1 if the input array is not one-dimensional.

        if array_object.size == 0:
            return -2  # Returns -2 if the input array is empty.

        # Calculates the required space inside the buffer to store the data inserted at the start_index
        array_data = np.frombuffer(array_object, dtype=np.uint8)  # Serializes to bytes
        data_size = array_data.size * array_data.itemsize  # Size of each element * the number of elements
        required_size = start_index + data_size

        if required_size > target_buffer.size:
            return 0  # Returns 0 if the buffer does not have enough space to accommodate the data

        # Writes the array data to the buffer, starting at the start_index and ending just before required_size index
        target_buffer[start_index:required_size] = array_data

        # Returns the required_size, which incidentally also matches the index that immediately follows the last index
        # of the buffer that was overwritten with the input data.
        return required_size

    def read_data(
        self,
        data_object: Union[
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.float32,
            np.float64,
            np.bool,
            NDArray[
                Union[
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.float32,
                    np.float64,
                    np.bool,
                ]
            ],
            Type,
        ],
        start_index: int = 0,
    ) -> tuple[Any, int]:
        """Recreates the input data_object using the data read from the payload stored inside the class reception
        buffer.

        This method uses the input object as a prototype, which supplies the number of bytes to read from the decoded
        payload received from the Microcontroller and the datatype to cast the read bytes to. If the payload has
        sufficiently many bytes available from the start_index to accommodate filling the object, the object will be
        recreated using the data extracted from the payload. Calling this method does not in any way modify the
        state of the reception buffer, so the same data can be read any number of times.

        Notes:
            At this time, the method only works with valid numpy scalars and arrays as well as python dataclasses
            entirely made out of valid numpy types. Using numpy rather than standard python types increases runtime
            speed (when combined with other optimizations) and enforces strict typing (critical for Microcontroller
            communication).

            The maximum runtime speed of this method is achieved when reading data as numpy arrays, which is
            optimized to a single read operation. The minimum runtime speed is achieved by reading dataclasses, as it
            involves looping over dataclass attributes.

        Args:
            data_object: A numpy scalar or array object or a python dataclass made entirely out of valid numpy objects.
                The input object is used as a prototype to determine how many bytes to read from the reception buffer
                and has to be properly initialized. Supported numpy types are: uint8, uint16, uint32, uint64, int8,
                int16, int32, int64, float32, float64, and bool. Array prototypes have to be 1-dimensional and not
                empty to be supported.
            start_index: The index inside the reception buffer (0 to 253) from which to start reading the
                data_object bytes. Unlike for write_data() method, this value is mandatory.

        Returns:
            A tuple of 2 elements. The first element is the data_object read from the reception buffer, which is cast
            to the requested datatype. The second element is the index that immediately follows the last index that
            was read from the _reception_buffer during method runtime.

        Raises:
            TypeError: If the input object is not a supported numpy scalar, numpy array, or python dataclass.
            ValueError: If the payload stored inside the reception buffer does not have the enough bytes
                available from the start_index to fill the requested object. Also, if the input object is a
                multidimensional or empty numpy array.
        """

        end_index = -10  # Initializes to a specific negative value that is not a valid index or runtime error code

        # If the input object is a supported numpy scalar, converts it to a numpy array and calls the read method.
        # Converts the returned one-element array back to a scalar numpy type. Due to current Numba limitations, this
        # is the most efficient available method.
        if isinstance(data_object, self._accepted_numpy_scalars):
            returned_object, end_index = self._read_array_data(
                self._reception_buffer,
                np.array(data_object, dtype=data_object.dtype),
                start_index,
                self._bytes_in_reception_buffer,
            )
            out_object = returned_object[0].copy()

        # If the input object is a numpy array, first ensures that its datatype matches one of the accepted scalar
        # numpy types and, if so, calls the array data reading method.
        elif isinstance(data_object, np.ndarray):
            if data_object.dtype in self._accepted_numpy_scalars:
                out_object, end_index = self._read_array_data(
                    self._reception_buffer,
                    data_object,
                    start_index,
                    self._bytes_in_reception_buffer,
                )

        # If the input object is a python dataclass, enters a recursive loop which calls this method for each class
        # attribute. This allows retrieving and overwriting each attribute with the bytes read from the buffer,
        # similar to the Microcontroller TransportLayer class.
        elif is_dataclass(data_object):
            # Records the initial index before looping over class attributes
            local_index = start_index

            # Loops over each field of the dataclass
            # noinspection PyDataclass
            for field in fields(data_object):
                # Calls the reader function recursively onto each field of the class
                attribute_value = getattr(data_object, field.name)
                attribute_object, local_index = self.read_data(data_object=attribute_value, start_index=local_index)

                # Updates the field in the original dataclass instance with the read object
                setattr(data_object, field.name, attribute_object)

            # Once the loop is over, sets the end_index to the final recorded local_index value and out_object to the
            # data_object dataclass
            out_object = data_object
            end_index = local_index

        # If the input value is not a valid numpy scalar, an array using a valid scalar datatype or a python dataclass,
        # raises TypeError exception.
        else:
            message = (
                f"Failed to read the data from the reception buffer. Encountered an unsupported input data_object "
                f"type ({type(data_object).__name__}). At this time, only the following numpy scalar or array types "
                f"are supported: {self._accepted_numpy_scalars}. Alternatively, a dataclass with all attributes "
                f"set to supported numpy scalar or array types is also supported."
            )
            console.error(message=message, error=TypeError)

        # If end_index is different from the start_index and no error has been raised, the method runtime was
        # successful, so returns the read data_object and the end_index to caller
        if end_index > start_index:
            # Returns the object recreated using data from the buffer and the end_index to caller
            # noinspection PyUnboundLocalVariable
            return out_object, end_index

        # If the index is set to code 0, this indicates that the payload did not have sufficient data starting from the
        # start_index to recreate the object.
        elif end_index == 0:
            message = (
                f"Failed to read the data from the reception buffer. The reception buffer does not have enough "
                f"bytes available to fully fill the object starting at the index {start_index}. Specifically, given "
                f"the object size of {data_object.nbytes} bytes, the required payload size is "
                f"{start_index + data_object.nbytes} bytes, but the available size is "
                f"{self.bytes_in_reception_buffer} bytes."
            )
            console.error(message=message, error=ValueError)

        # If the index is set to code -1, that indicates that a multidimensional numpy array was provided as input,
        # but only flat arrays are allowed.
        elif end_index == -1:
            message = (
                f"Failed to read the data from the reception buffer. Encountered a multidimensional numpy array with "
                f"{data_object.ndim} dimensions as input data_object. At this time, only one-dimensional (flat) "
                f"arrays are supported."
            )
            console.error(message=message, error=ValueError)

        # If the index is set to code -2, that indicates that an empty numpy array was provided as input, which does
        # not make sense and therefore is likely an error.
        elif end_index == -2:
            message = (
                f"Failed to read the data from the reception buffer. Encountered an empty (size 0) numpy array as "
                f"input data_object. Reading empty arrays is not supported."
            )
            console.error(message=message, error=ValueError)

        # If the end_index is not resolved properly, catches and raises a runtime error. This is a static guard to
        # aid developers in discovering errors.
        message = (
            f"Failed to read the data from the reception buffer. Encountered an unknown error code ({end_index})"
            f"returned by the reader method."
        )
        console.error(message=message, error=RuntimeError)

    @staticmethod
    @njit(nogil=True, cache=True)
    def _read_array_data(
        source_buffer: NDArray[np.uint8],
        array_object: NDArray[
            Union[
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.float32,
                np.float64,
                np.bool,
            ]
        ],
        start_index: int,
        payload_size: int,
    ) -> tuple[NDArray[Any], int]:
        """Reads the requested array_object from the reception buffer of the caller class.

        Specifically, the object's data is read as bytes and is converted to an array with the appropriate datatype.
        This method is not designed to be called directly. It should always be called through the read_data() method
        of the parent class.

        Args:
            source_buffer: The buffer from which the data will be read. This should be the _reception_buffer array
                of the caller class.
            array_object: The numpy array to be read from the _reception_buffer. Currently, the method is designed to
                only work with one-dimensional arrays with a minimal size of 1 element. The array should be initialized
                and should use one of the supported datatypes. During runtime, the method reconstructs the array using
                the data read from the source_buffer.
            start_index: The index inside the reception buffer (0 to 253) at which to start reading the data.
            payload_size: The number of payload bytes stored inside the buffer. This is used to limit the read operation
                to avoid retrieving data from the uninitialized portion of the buffer. Note, this will frequently be
                different from the total buffer size.

        Returns:
            A two-element tuple. The first element is the numpy array that uses the datatype and size derived from the
            input array_object, which holds the extracted data. The second element is the index that immediately follows
            the last index that was read during method runtime to support chained read calls. If method runtime fails,
            returns an empty numpy array as the first element and a static error-code as the second element. Uses
            integer code 0 if the payload bytes available from start_index are not enough to fill the input array with
            data. Returns integer code -1 if the input array is not one-dimensional. Returns integer code -2 if the
            input array is empty.
        """

        # Calculates the end index for the read operation. This is based on how many bytes are required to represent the
        # object and the start_index for the read operation.
        required_size = start_index + array_object.nbytes

        # Prevents reading outside the payload boundaries.
        if required_size > payload_size:
            return np.empty(0, dtype=array_object.dtype), 0

        # Prevents reading multidimensional numpy arrays.
        elif array_object.ndim > 1:
            return np.empty(0, dtype=array_object.dtype), -1

        # Prevents reading empty numpy arrays
        elif array_object.size == 0:
            return np.empty(0, dtype=array_object.dtype), -2

        # Generates a new array using the input data_object datatype and a slice of the byte-buffer that corresponds to
        # the number of bytes necessary to represent the object. Uses copy to ensure the returned object is not sharing
        # the buffer with the source_buffer.
        return (
            np.frombuffer(source_buffer[start_index:required_size], dtype=array_object.dtype).copy(),
            required_size,
        )

    def send_data(self) -> bool:
        """Packages the payload stored in the transmission buffer and sends it to the connected Microcontroller over the
        serial interface.

        Overall, this method carries out two distinct steps. First, it builds a packet using the payload, which is
        very fast (~3-5 microseconds on a ~5 Ghz CPU). Next, the method sends the constructed packet over the serial
        interface managed by pySerial library, which is considerably slower (20-40 microseconds).

        Notes:
            The constructed packet being sent over the serial port has the following format:
            [START BYTE]_[OVERHEAD BYTE]_[COBS ENCODED PAYLOAD]_[DELIMITER BYTE]_[CRC CHECKSUM]

        Returns:
            True, if the dat was successfully transmitted.

        Raises:
            ValueError: If the method encounters an error during the packet construction.
        """

        # Constructs the serial packet to be sent. This is a fast inline aggregation of all packet construction steps,
        # using JIT compilation to increase runtime speed. To maximize compilation benefits, it has to access the
        # inner jitclasses instead of using the python COBS and CRC class wrappers.
        packet = self._construct_packet(
            self._transmission_buffer,
            self._cobs_processor.processor,
            self._crc_processor.processor,
            self._bytes_in_transmission_buffer,
            self._delimiter_byte,
            self._start_byte,
        )

        # A valid packet will always have a positive size. If the returned packet size is above 0, proceeds with sending
        # the packet over the serial port.
        if packet.size > 0:
            # Calls pySerial write method. This takes 80% of this method's runtime and cannot really be optimized any
            # further as its speed directly depends on how the host OS handles serial port access.
            self._port.write(packet.tobytes())

            # Resets the transmission buffer to indicate that the payload was sent and prepare for sending the next
            # payload.
            self.reset_transmission_buffer()

            # Returns True to indicate that data was successfully sent.
            return True

        # If constructor method returns an empty packet, that means one of the inner methods ran into an error.
        # Only COBS and CRC classes can run into errors during _construct_packet() runtime. When this happens, the
        # method re-runs the computations using non-jit-compiled methods that will find and resolve the error. This is
        # slow, but if errors have occurred, it is likely that speed is no longer as relevant as error resolution.
        packet = self._cobs_processor.encode_payload(
            payload=self._transmission_buffer[: self._bytes_in_transmission_buffer],
            delimiter=self._delimiter_byte,
        )
        checksum = self._crc_processor.calculate_crc_checksum(packet)
        self._crc_processor.convert_checksum_to_bytes(checksum)

        # The steps above SHOULD run into an error. If they did not, there is an unexpected error originating from the
        # _construct_packet method. In this case, raises a generic RuntimeError to prompt the user to manually
        # debug the error.
        message = (
            "Failed to send the payload data. Unexpected error encountered for _construct_packet() method. "
            "Re-running all COBS and CRC steps used for packet construction in wrapped mode did not reproduce the "
            "error. Manual error resolution required."
        )
        console.error(message=message, error=RuntimeError)

    @staticmethod
    @njit(nogil=True, cache=True)
    def _construct_packet(
        payload_buffer: NDArray[np.uint8],
        cobs_processor: _COBSProcessor,
        crc_processor: _CRCProcessor,
        payload_size: int,
        delimiter_byte: np.uint8,
        start_byte: np.uint8,
    ) -> NDArray[np.uint8]:
        """Constructs the serial packet using the payload stored inside the input buffer.

        This method inlines COBS, CRC and start_byte prepending steps that iteratively transform the payload stored
        inside the caller class transmission buffer into a serial packet that can be transmitted to the
        Microcontroller. By accessing typically hidden jit-compiled _COBSProcessor and _CRCProcessor classes, this
        method inlines and compiles all operations into a single method, achieving the highest possible execution speed.

        Notes:
            At the time of writing, given other static checks performed at class instantiation, it is nearly impossible
            for ths runtime to fail. That said, since this can potentially change in the future, the method does contain
            a full suite of error handling tools.

        Args:
            payload_buffer: The numpy array that stores the 'raw' payload bytes. This should be the transmission buffer
                of the caller class.
            cobs_processor: The inner _COBSProcessor jitclass instance. The instance can be obtained by using
                '.processor' property of the COBSProcessor wrapper class.
            crc_processor: The inner _CRCProcessor jitclass instance. The instance can be obtained by using '.processor'
                property of the RCProcessor wrapper class.
            payload_size: The number of bytes that makes up the payload. It is expected that payload only uses
                a portion of the input payload_buffer.
            delimiter_byte: The byte-value used to mark the end of each transmitted packet's payload region.
            start_byte: The byte-value used to mark the beginning of each transmitted packet.

        Returns:
            The byte numpy array containing the constructed serial packet if the method runtime was successful.
            Otherwise, returns an empty numpy array (size 0) to indicate runtime failure. To trace the source of the
            error, it may be necessary to rerun this computation using error-raising wrapper COBS and CRC classes.
        """
        # Extracts the payload from the input buffer and encodes it using COBS scheme.
        packet = cobs_processor.encode_payload(payload_buffer[:payload_size], delimiter_byte)

        # If encoding fails, escalates the error by returning an empty array.
        if packet.size == 0:
            return np.empty(0, dtype=payload_buffer.dtype)

        # Calculates the CRC checksum for the encoded payload
        checksum = crc_processor.calculate_crc_checksum(packet)

        # Checksum calculation method does not have a unique error-associated return value. If it runs into an error, it
        # returns 0, but 0 can also be returned by a successful checksum calculation. To verify that the checksum
        # calculation was successful, verifies that the processor status matches expected success status.
        if crc_processor.status != crc_processor.checksum_calculated:
            return np.empty(0, dtype=payload_buffer.dtype)

        # Converts the integer checksum to a bytes' format (to form the crc postamble)
        postamble = crc_processor.convert_checksum_to_bytes(checksum)

        # For bytes' conversion, an empty checksum array indicates failure
        if postamble.size == 0:
            return np.empty(0, dtype=payload_buffer.dtype)

        # Generates message preamble using start_byte and payload_size.
        preamble = np.array([start_byte, payload_size], dtype=np.uint8)

        # Concatenates the preamble, the encoded payload, and the checksum postamble to form the serial packet
        # and returns the constructed packet to the caller.
        combined_array = np.concatenate((preamble, packet, postamble))
        return combined_array

    def receive_data(self) -> bool:
        """If available, receives the serial packet stored inside the reception buffer of the serial port.

        This method aggregates the steps necessary to read the packet data from the serial port's reception buffer,
        verify its integrity using CRC, and decode the payload out of the received data packet using COBS. Following
        verification, the decoded payload is transferred into the _reception_buffer array. This method uses multiple
        sub-methods and attempts to intelligently minimize the number of calls to the expensive serial port buffer
        manipulation methods.

        Notes:
            Expects the received data to be organized in the following format (different from the format used for
            sending the data to the microcontroller):
            [START BYTE]_[PAYLOAD SIZE BYTE]_[OVERHEAD BYTE]_[ENCODED PAYLOAD]_[DELIMITER BYTE]_[CRC CHECKSUM]

            The method can be co-opted as the check for whether the data is present in the first place, as it returns
            'False' if called when no data can be read or when the detected data is noise.

            Since calling data parsing methods is expensive, the method only attempts to parse the data if enough
            bytes are available, which is based on the minimum_received_payload_size class argument, among
            other things. The higher the value of this argument, the less time is wasted on trying to parse incomplete
            packets.

        Returns:
            A boolean 'True' if the data was parsed and is available for reading via read-data() calls. A boolean
            'False' if the number of available bytes is not enough to justify attempting to read the data.

        Raises:
            ValueError: If the received packet fails the CRC verification check, indicating that the packet is
                corrupted.
            RuntimeError: If _receive_packet method fails. Also, if an unexpected error occurs for any of the
                methods used to receive and parse the data.
            Exception: If _validate_packet() method fails, the validation steps are re-run using slower python-wrapped
                methods. Any errors encountered by these methods (From COBS and CRC classes) are raised as their
                preferred exception types.
        """
        # Clears the reception buffer in anticipation of receiving the new packet
        self.reset_reception_buffer()

        # Attempts to receive a new packet. If successful, this returns a static integer code 1 and saves the retrieved
        # packet to the _transmission_buffer and the size of the packet to the _bytes_in_transmission_buffer tracker.
        status_code = self._receive_packet()

        # Only carries out the rest of the processing if the packet was successfully received
        if status_code == 1:
            # Validates and unpacks the payload into the reception buffer
            payload_size = self._validate_packet(
                self._reception_buffer,
                self._bytes_in_reception_buffer,
                self._cobs_processor.processor,
                self._crc_processor.processor,
                self._delimiter_byte,
                self._postamble_size,
            )

            # Payload_size will always be a positive number if verification succeeds. In this case, overwrites the
            # _bytes_in_reception_buffer tracker with the payload size and returns 'true' to indicate runtime success
            if payload_size:
                self._bytes_in_reception_buffer = payload_size
                return True

            # If payload size is 0, this indicates runtime failure. In this case, reruns the verification procedure
            # using python-wrapped methods as they will necessarily catch and raise the error that prevented validating
            # the packet. This is analogous to how it is resolved for _construct_packet() method failures.
            else:
                packet = self._reception_buffer[: self._bytes_in_reception_buffer]  # Extracts the packet

                # Resets the reception buffer to ensure intermediate data saved to the tracker is not usable for
                # data reading attempts
                self.reset_reception_buffer()

                # CRC-checks packet's integrity
                checksum = self._crc_processor.calculate_crc_checksum(buffer=packet)

                # If checksum verification (NOT calculation, that is caught by the calculator method internally) fails,
                # generates a manual error message that tells the user how the checksum failed.
                if checksum != 0:
                    # Extracts the crc checksum from the end of the packet buffer
                    byte_checksum = packet[-self._postamble_size :]

                    # Also separates the packet portion of the buffer from the checksum
                    packet = packet[: packet.size - self._postamble_size]

                    # Converts the CRC checksum extracted from the end of the packet from a byte array to an integer.
                    checksum_number = self._crc_processor.convert_bytes_to_checksum(byte_checksum)

                    # Separately, calculates the checksum for the packet
                    expected_checksum = self._crc_processor.calculate_crc_checksum(buffer=packet)

                    # Uses the checksum values calculated above to issue an informative error message to the user.
                    error_message = (
                        f"CRC checksum verification failed when receiving data. Specifically, the checksum value "
                        f"transmitted with the packet {hex(checksum_number)} did not match the value expected for the "
                        f"packet (calculated locally) {hex(expected_checksum)}. This indicates packet was corrupted "
                        f"during transmission or reception."
                    )
                    raise ValueError(
                        textwrap.fill(
                            error_message,
                            width=120,
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                    )

                # Removes the CRC bytes from the end of the packet as they are no longer necessary if the CRC check
                # passed
                packet = packet[: packet.size - self._postamble_size]

                # COBS-decodes the payload from the received packet.
                _ = self._cobs_processor.decode_payload(packet=packet, delimiter=self._delimiter_byte)

                # The steps above SHOULD run into an error. If they did not, there is an unexpected error originating
                # from the _validate_packet method. In this case, raises a generic RuntimeError to notify the user of
                # the error so that they manually discover and rectify it.
                error_message = (
                    "Unexpected error encountered for _verify_packet() method when receiving data. Re-running all "
                    "COBS and CRC steps used for packet validation in wrapped mode did not reproduce the error. Manual "
                    "error resolution required."
                )
                raise RuntimeError(
                    textwrap.fill(
                        error_message,
                        width=120,
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )

        # Handles other possible status codes, all of which necessarily mean some failure has occurred during packet
        # reception runtime.
        # Not enough bytes were available to justify attempting to receive the packet, or enough bytes
        # were available, but they were noise bytes (start byte was not found and start_byte_errors are disabled).
        elif status_code == 101:
            # In this case just returns 'False' to indicate no data was parsed.
            return False

        # There are enough bytes to read, but no start_byte is found and start_byte errors are
        # enabled.
        elif status_code == 102:
            error_message = (
                "Serial packet reception failed. Start_byte value was not found among the bytes stored inside the "
                "serial buffer when parsing incoming serial packet."
            )

        # Payload-size byte was not received in time after discovering start_byte
        elif status_code == 103:
            error_message = (
                f"Serial packet reception failed. Reception staled at payload_size byte reception. Specifically, the "
                f"payload_size was not received in time ({self._timeout} microseconds) following the reception of the "
                f"start_byte."
            )

        # Payload-size byte was set to a value that exceeds the maximum allowed received payload size.
        elif status_code == 104:
            error_message = (
                f"Serial packet reception failed. The declared size of the payload "
                f"({self._bytes_in_reception_buffer}), extracted from the received payload_size byte of the serial "
                f"packet, was above the maximum allowed size of {self._reception_buffer.size}."
            )

        # Packet bytes were not received in time (packet reception staled)
        elif status_code == 105:
            # noinspection PyUnboundLocalVariable
            error_message = (
                f"Serial packet reception failed. Reception staled at packet bytes reception. Specifically, the "
                f"byte number {self._bytes_in_reception_buffer + 1} was not received in time ({self._timeout} "
                f"microseconds) following the reception of the previous byte."
            )

        # Unknown status_code. This should not really occur, and this is a static guard to help the developers.
        else:
            error_message = (
                f"Unknown status_code value {status_code} returned by the _receive_packet() method when "
                f"receiving data."
            )

        # Regardless of the error-message, uses RuntimeError for any valid error
        raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    def _receive_packet(self) -> int:
        """Attempts to read the serialized packet from the serial interface reception buffer.

        This is a fairly complicated method that calls another jit-compiled method to parse the bytes read from
        the serial port buffer into the packet format expected by this class. The method is designed to minimize the
        number of read() and in_waiting() method calls as they are very costly. The way this method is written should
        be optimized for the vast majority of cases though.

        Notes:
            This method uses the _timeout attribute to specify the maximum delay in microseconds(!) between receiving
            any two consecutive bytes of the packet. That is, if not all bytes of the packet are available to the method
            at runtime initialization, it will wait at most _timeout microseconds for the number of available bytes to
            increase before declaring the packet stale. There are two points at which the packet can become stale: the
            very beginning (the end of the preamble reception) and the reception of the packet itself. This corresponds
            to how the microcontroller sends teh data (preamble, followed by the packet+postamble fused into one). As
            such, the separating the two breakpoints with different error codes makes sense from the algorithmic
            perspective.

            The method tries to minimize the number of read() calls it makes as these calls are costly (compared to the
            rest of the methods in this library). As such, it may occasionally read more bytes than needed to process
            the incoming packet. In this case, any 'leftover' bytes are saved to the class _leftover_bytes attribute
            and reused by the next call to _parse_packet().

            This method assumes the sender uses the same CRC type as the SerializedTransferProtocol class, as it
            directly controls the CRC checksum byte-size. Similarly, it assumes teh sender uses the same delimiter and
            start_byte values as the class instance. If any of these assumptions are violated, this method will not
            parse the packet data correctly.

            Returned static codes: 101 → no bytes to read. 102 → start byte not found error. 103 → reception staled
            at acquiring the payload_size / packet_size. 104 → payload size too large (not valid). 105 → reception
            staled at acquiring packet bytes. Also returns code 1 to indicate successful packet acquisition.

        Returns:
            A static integer code (see notes) that denotes method runtime status. Status code '1' indicates successful
            runtime, and any other code is an error to be handled by the wrapper method. If runtime is successful, the
            retrieved packet is saved to the _reception_buffer and the size of the retrieved packet is saved to the
            _bytes_in_reception_buffer tracker.
        """

        # Quick preface. This method is written with a particular focus on minimizing the number of calls to read() and
        # in_waiting() methods of the Serial class as they take a very long time to run compared to most of the
        # jit-compiled methods provided by this library. As such, if the packet can be parsed without calling these two
        # methods, that is always the priority. The trade-off is that if the packet cannot be parsed, we are losing
        # time running library methods essentially for nothing. Whether this 'gamble' works out or not heavily depends
        # on how the library is used, but it is assumed that in the vast majority of cases it WILL pay off.

        # If there are NOT enough leftover bytes to justify starting the reception procedure, checks how many bytes can
        # be obtained from the serial port
        if len(self._leftover_bytes) < self._minimum_packet_size:
            # Combines the bytes inside the serial port buffer with the leftover bytes from previous calls to this
            # method and repeats the evaluation
            available_bytes = self._port.in_waiting
            total_bytes = len(self._leftover_bytes) + available_bytes
            enough_bytes_available = total_bytes > self._minimum_packet_size

            # If enough bytes are available after factoring in the buffered bytes, reads and appends buffered bytes to
            # the end of the leftover bytes buffer
            if enough_bytes_available:
                self._leftover_bytes += self._port.read(available_bytes)

        # Otherwise, if enough bytes are available without using the read operation, statically sets the flag to true
        # and begins parsing the packet
        else:
            enough_bytes_available = True

        # If not enough bytes are available, returns the static code 101 to indicate there were not enough bytes to
        # read from the buffer
        if not enough_bytes_available:
            return 101

        # Attempts to parse the packet from read bytes. This call expects, as a minium, to find the start byte and,
        # as a maximum, to resolve the entire packet.
        status, packet_size, remaining_bytes, packet_bytes = self._parse_packet(
            self._leftover_bytes,
            self._start_byte,
            self._max_rx_payload_size,
            self._postamble_size,
            self._allow_start_byte_errors,
        )

        # Resolves parsing result:
        # Packet parsed. Saves the packet to the _reception_buffer and the packet size to the
        # _bytes_in_reception_buffer tracker.
        if status == 1:
            self._reception_buffer[:packet_size] = packet_bytes
            self._bytes_in_reception_buffer = packet_size

            # If any bytes remain unprocessed, adds them to storage until the next call to this method
            self._leftover_bytes = remaining_bytes.tobytes()
            return status

        # Status above 2 means an already resolved error or a non-error terminal status. Currently, possible causes are:
        # either the start byte was not found, or the payload_size was too large (invalid).
        if status > 2:
            # This either completely resets the leftover_bytes tracker or sets them to the number of bytes left after
            # the terminal status cause was encountered. The latter case is exclusive to code 104, as encountering
            # an invalid payload_size may have unprocessed bytes that remain at the time the error scenario is
            # encountered.
            self._leftover_bytes = remaining_bytes.tobytes()
            # Only meaningful for code 104, shares the packet size to be used in error messages via the tracker value
            self._bytes_in_reception_buffer = packet_size
            return status

        # Packet found, but not enough bytes are available to finish parsing the packet. Code 0 specifically means that
        # the parser stopped at payload_size (payload_size byte was not available). This is easily the most
        # computationally demanding case, as potentially 2 more read() calls will be needed to parse the packet.
        elif status == 0:
            # Waits for at least one more byte to become available or for the reception to timeout.
            self._timer.reset()
            available_bytes = self._port.in_waiting
            while self._timer.elapsed < self._timeout or available_bytes != 0:
                available_bytes = self._port.in_waiting

            # If no more bytes are available (only one is necessary) returns code 103: Packet reception staled at
            # payload_size byte.
            if available_bytes == 0:
                # There are no leftover bytes when code 103 is encountered, so clears the storage
                self._leftover_bytes = bytes()
                return 103

            # If more bytes are available, reads the bytes into the placeholder storage. All leftover bytes are
            # necessarily consumed if status is 0, so the original value of the storage variable is irrelevant and
            # can be discarded at this point
            self._leftover_bytes = self._port.read()

            # This time sets a boolean flag to skip looking for start byte, as start byte is already found by the
            # first parser call.
            status, packet_size, remaining_bytes, packet_bytes = self._parse_packet(
                self._leftover_bytes,
                self._start_byte,
                self._max_rx_payload_size,
                self._postamble_size,
                self._allow_start_byte_errors,
                True,
            )

            # Status 1 indicates that the packet was fully parsed. Returns the packet to caller
            if status == 1:
                self._reception_buffer[:packet_size] = packet_bytes
                self._bytes_in_reception_buffer = packet_size
                self._leftover_bytes = remaining_bytes.tobytes()
                return status

            # Status 2 indicates not all the packet was parsed, but the payload_size has been found and resolved.
            # Attempts to resolve the rest of the packet
            elif status == 2:
                # Calculates the missing number of bytes from the packet_size and the size of the packet_bytes array
                required_size = packet_size - packet_bytes.size  # Accounts for already received bytes

                # Blocks until enough bytes are available. Resets the timer every time more bytes become available
                self._timer.reset()
                available_bytes = self._port.in_waiting
                delta = required_size - available_bytes  # Used to determine when to reset the timer
                while self._timer.elapsed < self._timeout or delta > 0:
                    available_bytes = self._port.in_waiting
                    delta_new = required_size - available_bytes

                    # Compares the deltas each cycle. If new delta is different from the old one, overwrites the delta
                    # and resets the timer
                    if delta_new != delta:
                        self._timer.reset()
                        delta = delta_new

                # If the while loop is escaped due to timeout, issues code 105: Packet reception staled at receiving
                # packet bytes.
                if delta > 0:
                    # There are no leftover bytes when code 105 is encountered, so clears the storage
                    self._leftover_bytes = bytes()
                    # Saves the number of the byte at which the reception staled so that it can be used in the error
                    # message raised by the wrapper
                    self._bytes_in_reception_buffer = packet_size - delta
                    return 105

                # If the bytes were received in time, calls the parser a third time to finish packet reception. Inputs
                # the packet_size and packet_bytes returned by the last method call to automatically jump to parsing
                # the remaining packet bytes
                status, packet_size, remaining_bytes, packet_bytes = self._parse_packet(
                    self._leftover_bytes,
                    self._start_byte,
                    self._max_rx_payload_size,
                    self._postamble_size,
                    self._allow_start_byte_errors,
                    True,
                    packet_size,
                    packet_bytes,
                )

                # This is the ONLY possible outcome
                if status == 1:
                    self._reception_buffer[0:packet_size] = packet_bytes
                    self._bytes_in_reception_buffer = packet_size
                    self._leftover_bytes = remaining_bytes.tobytes()
                    return status

            # If the status is not 1 or 2, returns the (already resolved) status 104. This is currently the only
            # possibility here, but uses status value in case it ever ends up being something else as well
            else:
                self._leftover_bytes = remaining_bytes.tobytes()
                self._bytes_in_reception_buffer = packet_size  # Saves the packet size to be used in the error message
                return status

        # Same as above, but code 2 means that the payload_size was found and used to determine the packet_size, but
        # there were not enough bytes to finish parsing the packet. Attempts to wait for enough bytes to become
        # available
        elif status == 2:
            # Calculates the missing number of bytes from the packet_size and the size of the packet_bytes array
            required_size = packet_size - packet_bytes.size  # Accounts for already received bytes

            # Blocks until enough bytes are available. Resets the timer every time more bytes become available
            self._timer.reset()
            available_bytes = self._port.in_waiting
            delta = required_size - available_bytes  # Used to determine when to reset the timer
            while self._timer.elapsed < self._timeout or delta > 0:
                available_bytes = self._port.in_waiting
                delta_new = required_size - available_bytes

                # Compares the deltas each cycle. If new delta is different from the old one, overwrites the delta
                # and resets the timer
                if delta_new != delta:
                    self._timer.reset()
                    delta = delta_new

            # If the while loop is escaped due to timeout, issues code 105: Packet reception staled at receiving
            # packet bytes.
            if delta > 0:
                # There are no leftover bytes when code 105 is encountered, so clears the storage
                self._leftover_bytes = bytes()
                # Saves the number of the byte at which the reception staled so that it can be used in the error
                # message raised by the wrapper
                self._bytes_in_reception_buffer = packet_size - delta
                return 105

            # If the bytes were received in time, calls the parser a third time to finish packet reception. Inputs
            # the packet_size and packet_bytes returned by the last method call to automatically jump to parsing
            # the remaining packet bytes
            status, packet_size, remaining_bytes, packet_bytes = self._parse_packet(
                self._leftover_bytes,
                self._start_byte,
                self._max_rx_payload_size,
                self._postamble_size,
                self._allow_start_byte_errors,
                True,
                packet_size,
                packet_bytes,
            )

            # The ONLY possible outcome.
            if status == 1:
                self._reception_buffer[0:packet_size] = packet_bytes
                self._bytes_in_reception_buffer = packet_size
                self._leftover_bytes = remaining_bytes.tobytes()
                return status

        # There should not be any way to reach this guard, but it is kept here to help developers by detecting when the
        # logic of this method fails to prevent it reaching this point
        error_message = (
            f"General failure of the _receive_packet() method runtime detected. Specifically, the method reached the "
            f"static guard, which should not be possible. The last available parser status is ({status}). Manual "
            f"intervention is required to identify and resolve the error."
        )
        raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    @staticmethod
    @njit(nogil=True, cache=True)
    def _parse_packet(
        read_bytes: bytes,
        start_byte: np.uint8,
        max_payload_size: np.uint8,
        postamble_size: int | np.unsignedinteger[Any],
        allow_start_byte_errors: bool,
        start_found: bool = False,
        packet_size: int = 0,
        packet_bytes: NDArray[np.uint8] = np.empty(0, dtype=np.uint8),
    ) -> tuple[int, int, NDArray[np.uint8], NDArray[np.uint8]]:
        """Parses as much of the packet as possible using the input bytes object.

        This method contains all packet parsing logic and takes in the bytes extracted from the serial port buffer.
        Running this method may produce a number of outputs, from a fully parsed packet to an empty buffer without
        a single packet byte. This method is designed to be called repeatedly until a packet is fully parsed, or until
        an external timeout guard handled by the _receive_packet() method aborts the reception. As such, it can
        recursively work on the same packet across multiple calls. To enable proper call hierarchy it is essential that
        this method is called strictly from the _receive_packet() method.

        Notes:
            This method becomes significantly more efficient in use patterns where many bytes are allowed to aggregate
            in the serial port buffer before being evaluated. Due to JIT compilation this method is very fast, and any
            execution time loss typically comes from reading the data from the underlying serial port.

            The returns of this method are designed to support potentially iterative (not recursive) calls to this
            method. As a minium, the packet may be fully resolved (parsed or failed to be parsed) with one call, and,
            as a maximum, 3 calls may be necessary.

            The method uses static integer codes to communicate its runtime status:

            0 - Not enough bytes read to fully parse the packet. The start byte was found, but payload_size byte was not
            and needs to be read.
            1 - Packet fully parsed.
            2 - Not enough bytes read to fully parse the packet. The payload_size was resolved, but there were not
            enough bytes to fully parse the packet and more bytes need to be read.
            101 - No start byte found, interpreted as 'no bytes to read' as the class is configured to ignore start
            byte errors. Usually, this situation is caused by communication line noise generating 'noise bytes'.
            102 - No start byte found, interpreted as a 'no start byte detected' error case. This status is only
            possible when the class is configured to detect start byte errors.
            104 - Payload_size value is too big (above maximum allowed payload size) error.

            The _read_packet() method is expected to issue codes 103 and 105 if packet reception stales at
            payload_size or packet bytes reception. All error codes are converted to errors at the highest level of the
            call hierarchy, which is the receive_data() method.

        Args:
            read_bytes: A bytes() object that stores the bytes read from the serial port. If this is the first call to
                this method for a given _receive_packet() method runtime, this object may also include any bytes left
                from the previous _receive_packet() runtime.
            start_byte: The byte-value used to mark the beginning of a transmitted packet in the byte-stream. This is
                used to detect the portion of the stream that stores the data packet.
            max_payload_size: The maximum size of the payload, in bytes, that can be received. This value cannot
                exceed 254 due to COBS limitations.
            postamble_size: The number of bytes needed to store the CRC checksum. This is determined based on the type
                of the CRC polynomial used by the class.
            allow_start_byte_errors: A boolean flag that determines whether inability to find start_byte should be
                interpreted as having no bytes to read (default, code 101) or as an error (code 102).
            start_found: Iterative argument. When this method is called two or more times, this value can be provided
                to the method to skip resolving the start byte (detecting packet presence). Specifically, it is used
                when a call to this method finds the start byte, but cannot resolve the packet size. Then, during a
                second call, start_byte searching step is skipped.
            packet_size: Iterative parameter. When this method is called two or more times, this value can be provided
                to the method to skip resolving the packet size. Specifically, it is used when a call to this method
                resolves the packet size, but cannot fully resolve the packet. Then, a second call to this method is
                made to resolve the packet and the size is provided as an argument to skip already completed parsing
                steps.
            packet_bytes: Iterative parameter. If the method is able to parse some, but not all the bytes making up the
                packet, parsed bytes can be fed back into the method during a second call using this argument.
                Then, the method will automatically combine already parsed bytes with newly extracted bytes.

        Returns:
            A tuple of four elements. The first element is an integer status code that describes the runtime. The
            second element is the parsed packet_size of the packet or 0 to indicate packet_size was not parsed. The
            third element is a numpy uint8 array that stores any bytes that remain after the packet parsing has been
            terminated due to success or error. The fourth element is the uint8 array that stores the portion of the
            packet that has been parsed so far, up to the entire packet (when method succeeds).
        """

        # Converts the input 'bytes' object to a numpy array to simplify the steps below
        evaluated_bytes = np.frombuffer(read_bytes, dtype=np.uint8)
        total_bytes = evaluated_bytes.size  # Calculates the total number of available bytes.
        parsed_packet_bytes = packet_bytes.size  # Calculates the number of already parsed packet bytes.

        # Counts how many bytes have been processed during various stages of this method.
        processed_bytes = 0

        # First, resolves the start byte if it has not been found:
        if not start_found:
            # Loops over available bytes until start byte is found or the method runs out of bytes to evaluate
            for i in range(total_bytes):
                processed_bytes += 1  # Increments with each evaluated byte

                # If the start byte is found, breaks the loop
                if evaluated_bytes[processed_bytes] == start_byte:
                    start_found = True
                    break

            # If the loop above terminates without finding the start byte, ends method runtime with the appropriate
            # status code.
            if not start_found:

                # Determines the status code based on whether start byte errors are allowed.
                # If they are allowed, returns 102. Otherwise (default) returns 101.
                if allow_start_byte_errors:
                    status_code = 102
                else:
                    status_code = 101

                packet_size = 0  # If start is not found, packet size is also not known.
                remaining_bytes = np.empty(0, dtype=np.uint8)  # All input bytes were processed
                packet_bytes = np.empty(0, dtype=np.uint8)  # No packet bytes were discovered
                return status_code, packet_size, remaining_bytes, packet_bytes

            # If there are no more bytes to read after encountering the start_byte, ends method runtime with code 0.
            if processed_bytes == total_bytes:

                # Returns code 0 to indicate that the packet was detected, but not fully received. Same code is used at
                # a later stage once the payload_byte is resolved. This tells the caller to block with timeout guard
                # until more data is available. Sets packet_size to 0 to indicate it was not found.
                status_code = 0
                packet_size = 0
                remaining_bytes = np.empty(0, dtype=np.uint8)
                packet_bytes = np.empty(0, dtype=np.uint8)
                return status_code, packet_size, remaining_bytes, packet_bytes

        # If packet size is not known (is zero), enters packet_size resolution mode. Largely, this depends on parsing
        # the payload size, which always follows the start_byte.
        if packet_size == 0:
            processed_bytes += 1  # Increments the counter

            # Reads the first available unprocessed byte as the payload size and checks it for validity. Valid packets
            # should store the payload size in the byte immediately after the start_byte.
            payload_size = evaluated_bytes[processed_bytes]

            # Verifies that the payload size is within the expected payload size limits.
            if not 0 < payload_size <= max_payload_size:

                # If payload size is out of bounds, returns with status code 104: Payload size not valid.
                status_code = 104
                packet_size = int(payload_size)  # Uses packet_size to store the invalid value for error messaging

                # If more bytes are available after discovering the invalid packet size, they are returned to caller.
                # It may be tempting to discard them, but they may contain parts of the NEXT packet, so a longer
                # route of re-feeding them into the processing sequence is chosen here.
                remaining_bytes = evaluated_bytes[processed_bytes:]  # Preserves the remaining bytes
                packet_bytes = np.empty(0, dtype=np.uint8)
                return status_code, packet_size, remaining_bytes, packet_bytes

            # If payload size passed verification, calculates the packet size. This uses the payload_size as the
            # backbone and increments it with the postamble size (depends on polynomial datatype) and
            # static +2 to account for the overhead and delimiter bytes introduced by COBS-encoding the packet.
            packet_size = payload_size + 2 + postamble_size

        # Checks if enough bytes are available in the evaluated_bytes array combined with the packet_bytes input array
        # (stores already processed packet bytes) to fully parse the packet.
        unprocessed_bytes = total_bytes - processed_bytes  # Calculates how many bytes are left to process
        remaining_packet_bytes = packet_size - parsed_packet_bytes  # Calculates how many packet bytes are necessary
        if unprocessed_bytes >= remaining_packet_bytes:

            # Extracts the remaining number of bytes needed to fully parse the packet from the buffer array.
            extracted_bytes = evaluated_bytes[processed_bytes : remaining_packet_bytes + processed_bytes]

            # Appends extracted bytes to the end of the array holding already parsed bytes
            packet = np.concatenate((packet_bytes, extracted_bytes))

            # Extracts any remaining bytes so that they can be properly stored for future receive_data() calls
            remaining_bytes = evaluated_bytes[remaining_packet_bytes + processed_bytes :]

            # Uses code 1 to indicate that the packet has been fully parsed
            status_code = 1
            return status_code, packet_size, remaining_bytes, packet

        # When not all bytes of the packet are available, moves all leftover bytes to the packet array and
        # uses static code 2 to indicate the paket was not available for parsing in-full. The caller method will then
        # block in-place until enough bytes are available to guarantee success of this method runtime on the next call
        status_code = 2

        # Zero, as all leftover bytes are absorbed into packet_bytes
        remaining_bytes = np.empty(0, dtype=np.uint8)

        # Discards any processed bytes and combines all remaining bytes with packet bytes.
        packet_bytes = np.concatenate((packet_bytes, evaluated_bytes[processed_bytes:]))

        return status_code, packet_size, remaining_bytes, packet_bytes

    @staticmethod
    @njit(nogil=True, cache=True)
    def _validate_packet(
        reception_buffer: NDArray[np.uint8],
        packet_size: int,
        cobs_processor: _COBSProcessor,
        crc_processor: _CRCProcessor,
        delimiter_byte: np.uint8,
        postamble_size: int | np.unsignedinteger[Any],
    ) -> int:
        """Validates the packet using CRC checksum, decodes it using COBS-scheme, and saves it to the reception_buffer.

        Both the CRC checksum and COBS decoding act as validation steps, and they jointly make it very unlikely that
        a corrupted packet passes this step. COBS-decoding extracts the payload from the buffer, making it available
        for consumption via read_data() method calls.

        Notes:
            This method expects the packet to be stored inside the _reception_buffer and will store the decoded
            payload to the _reception_buffer if method runtime succeeds. This allows optimizing memory usage and
            reduces the overhang of passing arrays around.

            The method uses the property of CRCs that ensures running a CRC calculation on the buffer to which its CRC
            checksum is appended will always return 0. For multibyte CRCs, this may be compromised if the byte-order of
            loading the CRC bytes into the postamble is not the order expected by the receiver system. This was never an
            issue during library testing, but good to be aware that is possible (usually some of the more nuanced
            UNIX-derived systems are known to do things differently in this regard).

        Args:
            reception_buffer: The buffer to which the extracted payload data will be saved and which is expected to
                store the packet to verify. Should be set to the _reception_buffer of the class.
            packet_size: The size of the packet to be verified. Used to access the required portion of the input
                reception_buffer.
            cobs_processor: The inner _COBSProcessor jitclass instance. The instance can be obtained by using
                '.processor' property of the COBSProcessor wrapper class.
            crc_processor: The inner _CRCProcessor jitclass instance. The instance can be obtained by using '.processor'
                property of the RCProcessor wrapper class.
            delimiter_byte: The byte-value used to mark the end of each transmitted packet's payload region.
            postamble_size: The byte-size of the crc postamble.

        Returns:
             A positive (> 0) integer that represents the size of the decoded payload if the method succeeds. Static
             code 0 if the method fails.
        """
        # Extracts the packet from the reception buffer
        packet = reception_buffer[:packet_size]

        # Calculates the CRC checksum for the packet + postamble, which is expected to return 0
        checksum = crc_processor.calculate_crc_checksum(buffer=packet)

        # Verifies that the checksum calculation method ran successfully. if not, returns 0 to indicate verification
        # failure
        if crc_processor.status != crc_processor.checksum_calculated:
            return 0

        # If the checksum is not 0, but the calculator runtime was successful, this indicates that the packet was
        # corrupted, so returns code 0
        if checksum != 0:
            return 0
        else:
            # Removes the CRC bytes from the end of the packet as they are no longer necessary if the CRC check passed
            packet = packet[: packet.size - postamble_size]

        # COBS-decodes the payload from the received packet.
        payload = cobs_processor.decode_payload(packet=packet, delimiter=delimiter_byte)

        # If the returned payload is empty, returns 0 to indicate that the COBS decoding step failed. This
        # is especially important, as the COBS decoding is used as a secondary packet integrity verification mechanism
        if payload.size == 0:
            return 0

        # If decoding succeeds, copies the decoded payload over to the reception buffer and returns the positive size
        # of the payload to caller to indicate runtime success. The returned size should always be above 0 if this
        # stage is reached
        reception_buffer[: payload.size] = payload
        return payload.size
