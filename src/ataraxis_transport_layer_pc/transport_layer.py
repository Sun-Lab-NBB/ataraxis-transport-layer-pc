"""This module provides the TransportLayer class used to establish and maintain bidirectional serial communication
with Arduino and Teensy microcontrollers running the ataraxis-transport-layer-mc library over USB / UART interface.

The class is written in a way that maximizes method runtime speed and is mostly limited by the speed of pySerial
runtime. The functionality of the class is realized through 4 main methods: write_data(), send_data(), receive_data(),
and read_data(). See method and class docstrings for more information.

Additionally, this module exposes the list_available_ports() function used to discover addressable USB ports when
setting up PC-MicroController communication.
"""

from typing import Any
from dataclasses import fields, is_dataclass

from numba import njit  # type: ignore
import numpy as np
from serial import Serial
from numpy.typing import NDArray
from serial.tools import list_ports
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import console

# noinspection PyProtectedMember,PyUnresolvedReferences
from .helper_modules import (
    SerialMock,
    CRCProcessor,
    COBSProcessor,
    _CRCProcessor,
    _COBSProcessor,
)


def list_available_ports() -> tuple[dict[str, int | str | Any], ...]:  # pragma: no cover
    """Provides the information about each serial port addressable through the pySerial library.

    This function is intended to be used for discovering and selecting the serial port 'names' to use with
    TransportLayer and Communication classes.

    Returns:
        A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
        port.

    """
    # Gets the list of port objects visible to the pySerial library.
    available_ports = list_ports.comports()

    # Prints the information about each port using terminal.
    information_list = [
        {
            "Name": port.name,
            "Device": port.device,
            "PID": port.pid,
            "Description": port.description,
        }
        for port in available_ports
    ]

    return tuple(information_list)


def print_available_ports() -> None:  # pragma: no cover
    """Lists all serial ports active on the host-system with descriptive information about the device connected to
    that port.

    This command is intended to be used for discovering the USB ports that can be connected to by a TransportLayer
    class instance.
    """
    # Records the current console status and, if necessary, ensured console is enabled before running this command.
    is_enabled = True
    if not console.enabled:
        is_enabled = False
        console.enable()  # Enables console output

    # Gets a tuple that stores all active USB ports with some ID information.
    available_ports = list_available_ports()

    # Loops over all discovered ports and prints the ID information about each port to the terminal
    count = 0  # Cannot use 'enumerate' due to filtering PID==None ports.
    for port in available_ports:
        # Filters out any ports with PID == None. This is primarily used to filter out invalid ports on Linux systems.
        if port["PID"] is not None:
            count += 1  # Counts only valid ports.
            console.echo(f"{count}: {port['Device']} -> {port['Description']}")  # Removes unnecessary information.

    # If the console was enabled by this runtime, ensures it is disabled before finishing the runtime.
    if not is_enabled:
        console.disable()


class TransportLayer:
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
            print_available_ports() library method to get a list of discoverable serial port names.
        microcontroller_serial_buffer_size: The size, in bytes, of the buffer used by the target microcontroller's
            Serial buffer. Usually, this information is available from the microcontroller's manufacturer (UART / USB
            controller specification).
        baudrate: The baudrate to be used to communicate with the Microcontroller. Should match the value used by
            the microcontroller for UART ports, ignored for USB ports. Note, the appropriate baudrate for any UART-using
            controller partially depends on its CPU clock! You can use this https://wormfood.net/avrbaudcalc.php tool
            to find the best baudrate for your board.
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
            the Microcontroller. Due to COBS encoding, this value has to be between 1 and 254 bytes. When set to 0, the
            class will automatically calculate and set this argument to the highest value compatible with the
            microcontroller_serial_buffer_size argument.
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
        _opened: Tracks whether the _port has been opened.
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
        _min_rx_payload_size: Stores the minimum number of bytes that can be received from the Microcontroller as a
            single payload. This value has to be between 1 and 254 and cannot exceed the max_rx_payload_size.
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
        microcontroller_serial_buffer_size: int,
        baudrate: int,
        polynomial: np.uint8 | np.uint16 | np.uint32 = np.uint8(0x07),
        initial_crc_value: np.uint8 | np.uint16 | np.uint32 = np.uint8(0x00),
        final_crc_xor_value: np.uint8 | np.uint16 | np.uint32 = np.uint8(0x00),
        maximum_transmitted_payload_size: int = 0,
        minimum_received_payload_size: int = 1,
        start_byte: int = 129,
        delimiter_byte: int = 0,
        timeout: int = 20000,
        *,
        test_mode: bool = False,
        allow_start_byte_errors: bool = False,
    ) -> None:
        # Tracks whether the serial por t is open. This is used solely to avoid one annoying __del__ error during
        # testing
        self._opened: bool = False

        # Verifies that input arguments are valid. Does not check polynomial parameters, that is offloaded to the
        # CRCProcessor class.
        if not isinstance(port, str):
            message = (
                f"Unable to initialize TransportLayer class. Expected a string value for 'port' argument, but "
                f"encountered {port} of type {type(port).__name__}."
            )
            console.error(message=message, error=TypeError)

        if not isinstance(baudrate, int) or baudrate <= 0:
            message = (
                f"Unable to initialize TransportLayer class. Expected a positive integer value for 'baudrate' "
                f"argument, but encountered {baudrate} of type {type(baudrate).__name__}."
            )
            console.error(message=message, error=ValueError)

        if not isinstance(start_byte, int) or not 0 <= start_byte <= 255:
            message = (
                f"Unable to initialize TransportLayer class. Expected an integer value between 0 and 255 for "
                f"'start_byte' argument, but encountered {start_byte} of type {type(start_byte).__name__}."
            )
            console.error(message=message, error=ValueError)

        if not isinstance(delimiter_byte, int) or not 0 <= delimiter_byte <= 255:
            message = (
                f"Unable to initialize TransportLayer class. Expected an integer value between 0 and 255 for "
                f"'delimiter_byte' argument, but encountered {delimiter_byte} of type {type(delimiter_byte).__name__}."
            )
            console.error(message=message, error=ValueError)

        if not isinstance(timeout, int) or timeout < 0:
            message = (
                f"Unable to initialize TransportLayer class. Expected an integer value of 0 or above for "
                f"'timeout' argument, but encountered {timeout} of type {type(timeout).__name__}."
            )
            console.error(message=message, error=ValueError)

        if start_byte == delimiter_byte:
            message = (
                "Unable to initialize TransportLayer class. The 'start_byte' and 'delimiter_byte' cannot be "
                "the same."
            )
            console.error(message=message, error=ValueError)

        if not isinstance(microcontroller_serial_buffer_size, int) or microcontroller_serial_buffer_size < 1:
            message = (
                f"Unable to initialize TransportLayer class. Expected a positive integer value for "
                f"'microcontroller_serial_buffer_size' argument, but encountered {microcontroller_serial_buffer_size} "
                f"of type {type(microcontroller_serial_buffer_size).__name__}."
            )
            console.error(message=message, error=ValueError)

        if not isinstance(maximum_transmitted_payload_size, int) or not 0 <= maximum_transmitted_payload_size <= 254:
            message = (
                f"Unable to initialize TransportLayer class. Expected an integer value between 0 and 254 for "
                f"'maximum_transmitted_payload_size' argument, but encountered {maximum_transmitted_payload_size} "
                f"of type {type(maximum_transmitted_payload_size).__name__}."
            )
            console.error(message=message, error=ValueError)

        if not isinstance(minimum_received_payload_size, int) or not 1 <= minimum_received_payload_size <= 254:
            message = (
                f"Unable to initialize TransportLayer class. Expected an integer value between 1 and 254 for "
                f"'minimum_received_payload_size' argument, but encountered {minimum_received_payload_size} "
                f"of type {type(minimum_received_payload_size).__name__}."
            )
            console.error(message=message, error=ValueError)

        # If maximum_transmitted_payload_size is set to the default initialization value of 0, automatically sets it
        # to the highest valid value. The value cannot exceed 254 and has to be at least 8 bytes smaller than the
        # microcontroller_serial_buffer_size to account for packet service bytes.
        if maximum_transmitted_payload_size == 0:
            maximum_transmitted_payload_size = min((microcontroller_serial_buffer_size - 8), 254)

        # Ensures that the specified maximum transmitted payload size would fit in the microcontroller's serial
        # buffer, accounting for the maximum size of the packet service bytes that will be added to the payload.
        elif maximum_transmitted_payload_size > microcontroller_serial_buffer_size - 8:
            message = (
                f"Unable to initialize TransportLayer class. After accounting for the maximum possible size of packet "
                f"service bytes (8), transmitted packets using maximum payload size "
                f"({maximum_transmitted_payload_size}) will not fit inside the microcontroller's Serial buffer, which "
                f"only has space for {microcontroller_serial_buffer_size} bytes."
            )
            console.error(message=message, error=ValueError)

        # Based on the class runtime selector, initializes a real or mock serial port manager class
        self._port: SerialMock | Serial
        if not test_mode:
            # Statically disables built-in timeout. Our jit- and c-extension classes are more optimized for this job
            # than Serial's built-in timeout.
            self._port = Serial(port, baudrate, timeout=0)  # pragma: no cover
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
        self._timeout: int = timeout
        self._allow_start_byte_errors: bool = allow_start_byte_errors
        self._postamble_size: np.uint8 = self._crc_processor.crc_byte_length

        # Uses payload size arguments to initialize reception and transmission buffers.
        self._max_tx_payload_size: np.uint8 = np.uint8(maximum_transmitted_payload_size)
        self._max_rx_payload_size: np.uint8 = np.uint8(254)  # Statically capped at 254 due to COBS encoding
        self._min_rx_payload_size: np.uint8 = np.uint8(minimum_received_payload_size)

        # Buffer sizes are up-case to uint16, as they may need to exceed the 256-size limit. They include the respective
        # payload size, the postamble size (1 to 4 bytes) and 4 static bytes for the preamble and packet metadata.
        # These 4 bytes are: start_byte, delimiter_byte, overhead_byte, and packet_size byte.
        tx_buffer_size: np.uint16 = np.uint16(self._max_tx_payload_size) + 4 + np.uint16(self._postamble_size)
        rx_buffer_size: np.uint16 = np.uint16(self._max_rx_payload_size) + 4 + np.uint16(self._postamble_size)
        self._transmission_buffer: NDArray[np.uint8] = np.zeros(shape=tx_buffer_size, dtype=np.uint8)
        self._reception_buffer: NDArray[np.uint8] = np.empty(shape=rx_buffer_size, dtype=np.uint8)

        # Based on the minimum expected payload size, calculates the minimum number of bytes that can fully represent
        # a packet. This is used to avoid costly pySerial calls unless there is a high chance that the call will return
        # a parsable packet.
        self._minimum_packet_size: int = minimum_received_payload_size + 4 + int(self._postamble_size)

        # Sets up various tracker and temporary storage variables that supplement class runtime.
        self._bytes_in_transmission_buffer: int = 0
        self._bytes_in_reception_buffer: int = 0
        self._leftover_bytes: bytes = b""  # Placeholder, this is re-initialized as needed during data reception.

        # Opens (connects to) the serial port. Cycles closing and opening to ensure the port is opened,
        # non-graciously replacing whatever is using the port at the time of instantiating TransportLayer class.
        # This non-safe procedure was implemented to avoid a frequent issue with Windows taking a long time to release
        # COM ports, preventing quick connection cycling.
        self._port.close()
        self._port.open()
        self._opened = True

    def __del__(self) -> None:
        """Ensures proper resource release prior to garbage-collecting class instance."""
        # Closes the port before deleting the class instance. Not strictly required, but helpful to ensure resources
        # are released
        if self._opened:
            self._port.close()

    def __repr__(self) -> str:
        """Returns a string representation of the TransportLayer class instance."""
        if isinstance(self._port, Serial):  # pragma: no cover
            representation_string = (
                f"TransportLayer(port='{self._port.name}', baudrate={self._port.baudrate}, polynomial="
                f"{self._crc_processor.polynomial}, start_byte={self._start_byte}, "
                f"delimiter_byte={self._delimiter_byte}, timeout={self._timeout} us, "
                f"maximum_tx_payload_size = {self._max_tx_payload_size}, "
                f"maximum_rx_payload_size={self._max_rx_payload_size})"
            )
        else:
            representation_string = (
                f"TransportLayer(port & baudrate=MOCKED, polynomial={self._crc_processor.polynomial}, "
                f"start_byte={self._start_byte}, delimiter_byte={self._delimiter_byte}, timeout={self._timeout} us, "
                f"maximum_tx_payload_size = {self._max_tx_payload_size}, "
                f"maximum_rx_payload_size={self._max_rx_payload_size})"
            )
        return representation_string

    @property
    def available(self) -> bool:
        """Returns True if enough bytes are available from the serial port to justify attempting to receive a packet."""
        # in_waiting is twice as fast as using the read() method. The 'true' outcome of this check is capped at the
        # minimum packet size to minimize the chance of having to call read() more than once. The method counts the
        # bytes available for reading and left over from previous packet parsing operations.
        return (self._port.in_waiting + len(self._leftover_bytes)) >= self._minimum_packet_size

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
        approach to 'resetting' the buffer by overwriting, rather than recreation is chosen for higher memory
        efficiency and runtime speed.
        """
        self._bytes_in_transmission_buffer = 0

    def reset_reception_buffer(self) -> None:
        """Resets the reception buffer bytes tracker to 0.

        This does not physically alter the buffer in any way, but makes all data inside the buffer 'invalid'. This
        approach to 'resetting' the buffer by overwriting, rather than recreation is chosen for higher memory
        efficiency and runtime speed.
        """
        self._bytes_in_reception_buffer = 0

    def write_data(
        self,
        data_object: Any,
        start_index: int | None = None,
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

                # If this call fails, it will raise an error that wll terminate this loop early
                local_index = self.write_data(data_object=data_value, start_index=local_index)

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
            # index that immediately follows the final index of the buffer that was overwritten with the input data.
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
                f"{data_object.ndim} dimensions as input data_object. At this time, only one-dimensional (flat) arrays "
                f"are supported."
            )
            console.error(message=message, error=ValueError)

        # If the index is set to code -2, that indicates that an empty numpy array was provided as input, which does
        # not make sense and, therefore, is likely an error. Also, empty arrays are explicitly not valid in C/C++, so
        # this is also against language rules to provide them with an intention to send that data to Microcontroller
        # running C.
        if end_index == -2:
            message = (
                "Failed to write the data to the transmission buffer. Encountered an empty (size 0) numpy array as "
                "input data_object. Writing empty arrays is not supported."
            )
            console.error(message=message, error=ValueError)

        # If the end_index is not resolved properly, catches and raises a runtime error
        message = (
            f"Failed to write the data to the transmission buffer. Encountered an unknown error code ({end_index}) "
            f"returned by the writer method."
        )  # pragma: no cover
        console.error(message=message, error=RuntimeError)  # pragma: no cover

        # This fallback is to appease MyPy and will neve rbe reached
        raise RuntimeError(message)  # pragma: no cover

    @staticmethod
    @njit(nogil=True, cache=True)  # type: ignore # pragma: no cover
    def _write_scalar_data(
        target_buffer: NDArray[np.uint8],
        scalar_object: (
            np.uint8
            | np.uint16
            | np.uint32
            | np.uint64
            | np.int8
            | np.int16
            | np.int32
            | np.int64
            | np.float32
            | np.float64
            | np.bool
        ),
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
    @njit(nogil=True, cache=True)  # type: ignore # pragma: no cover
    def _write_array_data(
        target_buffer: NDArray[np.uint8],
        array_object: NDArray[
            np.uint8
            | np.uint16
            | np.uint32
            | np.uint64
            | np.int8
            | np.int16
            | np.int32
            | np.int64
            | np.float32
            | np.float64
            | np.bool
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
        data_object: Any,
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
        if end_index == 0:
            message = (
                f"Failed to read the data from the reception buffer. The reception buffer does not have enough "
                f"bytes available to fully fill the object starting at the index {start_index}. Specifically, given "
                f"the object size of {data_object.nbytes} bytes, the required payload size is "
                f"{start_index + data_object.nbytes} bytes, but the available size is {self.bytes_in_reception_buffer} "
                f"bytes."
            )
            console.error(message=message, error=ValueError)

        # If the index is set to code -1, that indicates that a multidimensional numpy array was provided as input,
        # but only flat arrays are allowed.
        elif end_index == -1:
            message = (
                f"Failed to read the data from the reception buffer. Encountered a multidimensional numpy array with "
                f"{data_object.ndim} dimensions as input data_object. At this time, only one-dimensional (flat) arrays "
                f"are supported."
            )
            console.error(message=message, error=ValueError)

        # If the index is set to code -2, that indicates that an empty numpy array was provided as input, which does
        # not make sense and therefore is likely an error.
        elif end_index == -2:
            message = (
                "Failed to read the data from the reception buffer. Encountered an empty (size 0) numpy array as "
                "input data_object. Reading empty arrays is not supported."
            )
            console.error(message=message, error=ValueError)

        # If the end_index is not resolved properly, catches and raises a runtime error. This is a static guard to
        # aid developers in discovering errors.
        message = (
            f"Failed to read the data from the reception buffer. Encountered an unknown error code ({end_index})"
            f"returned by the reader method."
        )  # pragma: no cover
        console.error(message=message, error=RuntimeError)  # pragma: no cover

        # Fallback to appease MyPy, will never be reached
        raise RuntimeError(message)  # pragma: no cover

    @staticmethod
    @njit(nogil=True, cache=True)  # type: ignore # pragma: no cover
    def _read_array_data(
        source_buffer: NDArray[np.uint8],
        array_object: NDArray[
            np.uint8
            | np.uint16
            | np.uint32
            | np.uint64
            | np.int8
            | np.int16
            | np.int32
            | np.int64
            | np.float32
            | np.float64
            | np.bool
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
        if array_object.ndim > 1:
            return np.empty(0, dtype=array_object.dtype), -1

        # Prevents reading empty numpy arrays
        if array_object.size == 0:
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
            True, if the data was successfully transmitted.

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
        # Due to other controls, simulating this error with tests is currently impossible. The code is kept as-is for
        # potential future relevance, however.
        checksum = self._crc_processor.calculate_crc_checksum(packet)  # pragma: no cover
        self._crc_processor.convert_checksum_to_bytes(checksum)  # pragma: no cover

        # The steps above SHOULD run into an error. If they did not, there is an unexpected error originating from the
        # _construct_packet method. In this case, raises a generic RuntimeError to prompt the user to manually
        # debug the error.
        message = (
            "Failed to send the payload data. Unexpected error encountered for _construct_packet() method. "
            "Re-running all COBS and CRC steps used for packet construction in wrapped mode did not reproduce the "
            "error. Manual error resolution required."
        )  # pragma: no cover
        console.error(message=message, error=RuntimeError)  # pragma: no cover

        # Fallback to appease MyPy, will never be reached.
        raise RuntimeError(message)  # pragma: no cover

    @staticmethod
    @njit(nogil=True, cache=True)  # type: ignore # pragma: no cover
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
        """If data bytes are available for reception, parses the serialized data packet, verified its integrity, and
        decodes the packet's payload into the class reception buffer.

        This method aggregates the steps necessary to read the packet data from the serial port's reception buffer,
        verify its integrity using CRC, and decode the payload using COBS. Following verification, the decoded payload
        is transferred into the class reception buffer. This method uses multiple sub-methods and attempts to
        intelligently minimize the number of calls to comparatively slow serial port buffer manipulation methods.

        Notes:
            Expects the received data to be organized in the following format:
            [START BYTE]_[PAYLOAD SIZE BYTE]_[OVERHEAD BYTE]_[COBS ENCODED PAYLOAD]_[DELIMITER BYTE]_[CRC CHECKSUM]

            Before doing any processing, the method checks for whether the data is present in the first place. Even if
            the class 'available' property returns True, this method can still return False. This would be the case if
            the 'data' available for reading is actually the communication line noise. Overall, there is no need to
            check the 'available' property before calling this method, as the method does this internally anyway.

            Since running this method can get comparatively costly, the method only attempts to process the data if
            it is relatively likely that the method will run successfully. To determine this likelihood, the method uses
            the minimum_received_payload_size class attribute and only processes the data if enough bytes are available
            to represent the minimum packet size.

        Returns:
            True if the packet was successfully received and its payload was decoded into the reception buffer. False,
            if there are no packet bytes to receive and process when this method is called.

        Raises:
            ValueError: If the received packet fails the CRC verification check.
            RuntimeError: If the method runs into an error while receiving the packet's data.
        """
        # Clears the reception buffer
        self.reset_reception_buffer()

        # Attempts to receive a new packet. If successful, this method saves the received packet to the
        # _transmission_buffer and the size of the packet to the _bytes_in_transmission_buffer tracker. If the method
        # runs into an error, it raises the appropriate RuntimeError.
        if not self._receive_packet():
            # If the packet parsing method does not find any packet bytes to process, it returns False. This method then
            # escalates the return to the caller.
            return False

        # If the packet is successfully parsed, validates and unpacks the payload into the class reception buffer
        payload_size = self._validate_packet(
            self._reception_buffer,
            self._bytes_in_reception_buffer,
            self._cobs_processor.processor,
            self._crc_processor.processor,
            self._delimiter_byte,
            self._postamble_size,
        )

        # Returned payload_size will always be a positive integer (>= 1) if verification succeeds. If verification
        # succeeds, overwrites the _bytes_in_reception_buffer tracker with the payload size and returns True to
        # indicate runtime success
        if payload_size:
            self._bytes_in_reception_buffer = payload_size
            return True

        # If payload size is 0, this indicates verification failure. In this case, attempts to resolve the cause of the
        # failure and raise the appropriate error:

        # Extracts the data that failed verification into a separate buffer and resets the reception buffer to
        # make it impossible to read the invalid data
        packet = self._reception_buffer[: self._bytes_in_reception_buffer].copy()
        self.reset_reception_buffer()

        # Resolves the status of the CRC checksum calculator. If verification failed due to a CRC calculation error,
        # this method will raise the appropriate error.
        # noinspection PyProtectedMember
        self._crc_processor._resolve_checksum_calculation_status(packet)

        # If CRC calculation ran successfully, resolves the status of COBS decoding (provided COBS decoder status is not
        # standby or encoding success). If verification failed due to COBS decoding error, this method will raise the
        # appropriate error.
        if (
            self._cobs_processor.processor.status != self._cobs_processor.processor.standby
            and self._cobs_processor.processor.status != self._cobs_processor.processor.payload_encoded
        ):
            # Removes the CRC bytes before running the decoder.
            # noinspection PyProtectedMember
            self._cobs_processor._resolve_decoding_status(packet=packet[: packet.size - int(self._postamble_size)])

        # If the checks above did not raise an error, the verification necessarily failed due to CRC checksum
        # verification error. This indicates that the packet was corrupted during transmission. The steps below generate
        # an informative RuntimeError message:

        # Converts the CRC checksum extracted from the end of the packet from a byte array to an integer. This uses the
        # wrapper class that raises the appropriate error if method runtime fails.
        byte_checksum = packet[-self._postamble_size :]  # Extracts the received CRC checksum as a bytes' array
        received_checksum = self._crc_processor.convert_bytes_to_checksum(byte_checksum)

        # Calculates the expected CRC checksum for the encoded payload.
        encoded_payload = packet[: packet.size - int(self._postamble_size)]  # Removes the CRC postamble bytes
        expected_checksum = self._crc_processor.calculate_crc_checksum(buffer=encoded_payload)

        # Uses the checksum values calculated above to issue an informative error message to the user.
        message = (
            f"Failed to verify the received serial packet's integrity. The checksum value transmitted with the packet "
            f"{hex(received_checksum)} did not match the expected value based on the packet data "
            f"{hex(expected_checksum)}. This indicates the packet was corrupted during transmission or reception."
        )
        console.error(message=message, error=ValueError)

        # Fallback to appease MyPy, will never be reached.
        raise ValueError(message)  # pragma: no cover

    def _receive_packet(self) -> bool:
        """Attempts to receive (parse) the serialized payload and the CRC checksum postamble from the byte-stream
        stored inside the serial interface buffer.

        This method's runtime can be broadly divided into 2 distinct steps. The first step transfers the bytes received
        by the serial interface into the buffer used by the second step. This step relies on pure python code and
        pySerial library and is comparatively slow. The second step parses the encoded payload and the CRC checksum from
        the received bytes, and it uses comparatively fast jit-compiled code. The method is designed to minimize the
        time spent in step 1 where possible, but is largely limited by the serial port interface.

        Notes:
            This method uses the class _timeout attribute to specify the maximum delay in microseconds(!) between
            receiving any two consecutive bytes of the packet. If the packet is not fully received at method runtime
            initialization, it will wait at most _timeout microseconds for the number of available bytes to
            increase before declaring the packet stale. There are two points at which the packet can become stale: the
            end of the preamble reception and the reception of the packet (payload + crc postamble). The primary
            difference between the two breakpoints is that in the latter case the exact size of the packet is known and
            in the former it is not known.

            The method tries to minimize the number of serial port interface calls, as these calls are comparatively
            costly. To do so, the method always reads all bytes available from the serial port interface, regardless of
            how many bytes it needs to resolve the packet. After resolving the packet, any 'leftover' bytes are saved
            to the class _leftover_bytes buffer and reused by the next call to _parse_packet().

            For this method to work correctly, the class configuration should exactly match the configuration of the
            TransportLayer class used by the connected Microcontroller. If the two configurations do not match, this
            method will likely fail to parse any incoming packet.

        Returns:
            True, if the method is able to successfully parse the incoming packet. In this case, the COBS-encoded
            payload and the CRC checksum will be stored inside the _reception_buffer and the size of the received
            packet will be stored in the _bytes_in_reception_buffer tracker. Returns False if there are no packet
            bytes to parse (valid non-error status).

        Raises:
            RuntimeError: If the method runs into an error while parsing the incoming packet. Broadly, this can be due
                to packet corruption, the mismatch between TransportLayer class configurations, or the packet
                transmission staling.
        """
        # Checks whether class buffers contain enough bytes to justify parsing the packet. If not, returns False to
        # indicate graceful (non-error) runtime failure.
        if not self._bytes_available(required_bytes_count=self._minimum_packet_size):
            return False

        # Pre-initializes the variables that support proper iteration of the parsing process below.
        status: int = 150  # This is not a valid status code
        parsed_bytes_count: int = 0
        parsed_bytes: NDArray[np.uint8] = np.empty(shape=0, dtype=np.uint8)
        start_found: bool = False

        # Enters the packet parsing loop. Due to the parsing implementation, the packet can be resolved over at most
        # three iterations of the parsing method. Therefore, this loop is statically capped at 3 iterations.
        for call_count in range(3):
            # Converts leftover_bytes (bytes) to a numpy uint8 array for compatibility with _parse_packet
            remaining_bytes = np.frombuffer(self._leftover_bytes, dtype=np.uint8).copy()

            # Calls the packet parsing method. The method reuses some iterative outputs as arguments for later
            # calls.
            status, parsed_bytes_count, remaining_bytes, parsed_bytes = self._parse_packet(
                remaining_bytes,
                self._start_byte,
                self._delimiter_byte,
                self._max_rx_payload_size,
                self._min_rx_payload_size,
                self._postamble_size,
                self._allow_start_byte_errors,
                start_found,
                parsed_bytes_count,
                parsed_bytes,
            )

            # Convert remaining_bytes_np (numpy array) back to bytes after function runtime
            self._leftover_bytes = remaining_bytes.tobytes()
            # Resolves parsing result:
            # Packet parsed. Saves the packet to the _reception_buffer and the packet size to the
            # _bytes_in_reception_buffer tracker.
            if status == 1:
                self._reception_buffer[: parsed_bytes.size] = parsed_bytes
                self._bytes_in_reception_buffer = parsed_bytes.size  # Includes encoded payload + CRC postamble!
                return True  # Success code

            # Partial success status. The method was able to resolve the start_byte, but not the payload_size. This
            # means that the method does not know the exact number of bytes needed to fully resolve the packet. The
            # expectation is that the next byte after the start_byte is the payload_size byte. Therefore, technically,
            # only one additional byte needs to be available to justify the next iteration of packet parsing. However,
            # to minimize the number of serial interface calls, _bytes_available() blocks until there are enough bytes
            # to fully cover the minimum packet size -1 (-1 is to account for already processed start_byte). This
            # maximizes the chances of successfully parsing the full packet during iteration 2. That said, since the
            # exact size of the packet is not known, iteration 3 may be necessary.
            if status == 0 and not self._bytes_available(
                required_bytes_count=self._minimum_packet_size - 1, timeout=self._timeout
            ):
                # The only way for _bytes_available() to return False is due to timeout guard aborting additional bytes'
                # reception.
                message = (
                    f"Failed to parse the size of the incoming serial packet. The packet size byte was not received in "
                    f"time ({self._timeout} microseconds), following the reception of the START byte."
                )
                console.error(message=message, error=RuntimeError)

                # This explicit fallback terminator is here to appease Mypy and will never be reached.
                raise RuntimeError(message)  # pragma: no cover

            # Partial success status. This is generally similar to status 0 with one notable exception. Status 2 means
            # that the payload size was parsed and, therefore, the exact number of bytes making up the processed packet
            # is known. This method, therefore, blocks until the class is able to receive enough bytes to fully
            # represent the packet or until the reception timeout.
            if status == 2 and not self._bytes_available(
                required_bytes_count=parsed_bytes.size - parsed_bytes_count, timeout=self._timeout
            ):
                # The only way for _bytes_available() to return False is due to timeout guard aborting additional bytes'
                # reception.
                message = (
                    f"Failed to parse the incoming serial packet data. The byte number {parsed_bytes_count + 1} "
                    f"out of {parsed_bytes.size} was not received in time ({self._timeout} microseconds), "
                    f"following the reception of the previous byte. Packet reception staled."
                )
                console.error(message=message, error=RuntimeError)

                # This explicit fallback terminator is here to appease Mypy and will never be reached.
                raise RuntimeError(message)  # pragma: no cover

            # A separate error message that specifically detects status 3: Not enough bytes to resolve the CRC
            # postamble. Technically, this error should not be possible (it is the terminal runtime status for the
            # packet parsing method). However, it is implemented to avoid confusion with status 2 and 0.
            if status == 3 and not self._bytes_available(
                required_bytes_count=parsed_bytes.size - parsed_bytes_count, timeout=self._timeout
            ):
                # The only way for _bytes_available() to return False is due to timeout guard aborting additional bytes'
                # reception.
                message = (
                    f"Failed to parse the incoming serial packet's CRC postamble. The byte number "
                    f"{parsed_bytes_count + 1} out of {parsed_bytes.size} was not received in time "
                    f"({self._timeout} microseconds), following the reception of the previous byte. Packet reception "
                    f"staled."
                )  # pragma: no cover
                console.error(message=message, error=RuntimeError)  # pragma: no cover

                # This explicit fallback terminator is here to appease Mypy and will never be reached.
                raise RuntimeError(message)  # pragma: no cover

            # If _bytes_available() method returned true for status codes 1 to 3, that means that additional bytes were
            # received in time and the loop has to be cycled again to process newly received bytes.
            if status <= 3:
                continue

            # Any code other than partial or full success code is interpreted as the terminal code. All codes other
            # than 101 are error codes. Code 101 is a non-error non-success terminal code. This clause also contains
            # the resolution for unexpected status codes.

            # No packet to receive. This is a non-error terminal status.
            if status == 101:
                return False  # Non-error, non-success return code

            # Start byte was not discovered among the available bytes.
            if status == 102:
                message = (
                    f"Failed to parse the incoming serial packet data. Unable to find the start_byte "
                    f"({self._start_byte}) value among the bytes stored inside the serial buffer."
                )

            # Parsed payload size is not within the boundaries specified by the minimum and maximum payload sizes.
            elif status == 103:
                message = (
                    f"Failed to parse the incoming serial packet data. The parsed size of the COBS-encoded payload "
                    f"({parsed_bytes.size}), is outside the expected boundaries "
                    f"({self._min_rx_payload_size} to {self._max_rx_payload_size}). This likely indicates a "
                    f"mismatch in the transmission parameters between this system and the Microcontroller."
                )

            # Delimiter byte value was encountered before reaching the end of the COBS-encoded payload data region.
            # 'expected number' is calculated like this: parsed_bytes has space for the encoded packet + CRC. So, to get
            # the expected delimiter byte number, we just subtract the CRC size from the parsed_bytes size.
            elif status == 104:
                message = (
                    f"Failed to parse the incoming serial packet data. Delimiter byte value ({self._delimiter_byte}) "
                    f"encountered at payload byte number {parsed_bytes_count}, instead of the expected byte number "
                    f"{parsed_bytes.size - int(self._postamble_size)}. This likely indicates packet corruption or "
                    f"mismatch in the transmission parameters between this system and the Microcontroller."
                )

            # The last COBS-encoded payload (encoded packet's) data value does not match the expected delimiter byte
            # value.
            elif status == 105:
                message = (
                    f"Failed to parse the incoming serial packet data. Delimiter byte value ({self._delimiter_byte}) "
                    f"expected as the last encoded packet byte ({parsed_bytes.size - int(self._postamble_size)}), but "
                    f"instead encountered {parsed_bytes[parsed_bytes_count-1]}. This likely indicates packet "
                    f"corruption or mismatch in the transmission parameters between this system and the "
                    f"Microcontroller."
                )

            # Unknown status_code. Reaching this clause should not be possible. This is a static guard to help
            # developers during future codebase updates.
            else:  # pragma: no cover
                break  # Breaks the loop, which issues the 'unknown status code' message

            # Raises the resolved error message as RuntimeError.
            console.error(message=message, error=RuntimeError)

        # The static guard for unknown status code. This is moved to the end of the message to appease MyPy.
        message = (
            f"Failed to parse the incoming serial packet data. Encountered an unknown status value "
            f"{status}, returned by the _receive_packet() method. Manual user intervention is required to "
            f"resolve the issue."
        )  # pragma: no cover
        # Raises the resolved error message as RuntimeError.
        console.error(message=message, error=RuntimeError)  # pragma: no cover

        # This explicit fallback terminator is here to appease Mypy and will never be reached.
        raise RuntimeError(message)  # pragma: no cover

    def _bytes_available(self, required_bytes_count: int = 1, timeout: int = 0) -> bool:
        """Determines if the required number of bytes is available across all class buffers that store unprocessed
        bytes.

        Specifically, this method first checks whether the leftover_bytes buffer of the class contains enough bytes.
        If not, the method checks how many bytes can be extracted from the serial interface buffer, combines these bytes
        with the leftover bytes, and repeats the check. If the check succeeds, the method reads all available bytes from
        the serial port and stacks them with the bytes already stored inside the leftover_bytes buffer before returning.
        Optionally, the method can block in-place while the serial port continuously receives new bytes, until the
        required number of bytes becomes available.

        Notes:
            This method is primarily designed to optimize packet processing speed by minimizing the number of calls to
            the serial interface methods. This is because these methods take a comparatively long time to execute.

        Args:
            required_bytes_count: The number of bytes that needs to be available across all class buffers that store
                unprocessed bytes for this method to return True.
            timeout: The maximum number of microseconds that can pass between the serial port receiving any two
                consecutive bytes. Using a non-zero timeout allows the method to briefly block and wait for the
                required number of bytes to become available, as long as the serial port keeps receiving new bytes.

        Returns:
            True if enough bytes are available at the end of this method runtime to justify parsing the packet.
        """
        # Tracks the number of bytes available from the leftover_bytes buffer
        available_bytes = len(self._leftover_bytes)

        # If the requested number of bytes is already available from the leftover_bytes buffer, returns True.
        if available_bytes >= required_bytes_count:
            return True

        # If there are not enough leftover bytes to satisfy the requirement, enters a timed loop that waits for
        # the serial port to receive additional bytes. The serial port has its own buffer, and it takes a
        # comparatively long time to view and access that buffer. Hence, this is a 'fallback' procedure.
        self._timer.reset()  # Resets the timer before entering the loop
        previous_additional_bytes = 0  # Tracks how many bytes were available during the previous iteration of the loop
        once = True  # Allows the loop below to run once even if timeout is 0
        while self._timer.elapsed < timeout or once:
            # Deactivates the 'once' condition to make future loop iterations correctly depend on timeout
            if once:
                once = False

            additional_bytes = self._port.in_waiting  # Returns the number of bytes that can be read from serial port.
            total_bytes = available_bytes + additional_bytes  # Combines leftover and serial port bytes.

            # If the combined total matches the required number of bytes, reads additional bytes into the leftover_bytes
            # buffer and returns True.
            if total_bytes >= required_bytes_count:
                self._leftover_bytes += self._port.read(
                    additional_bytes
                )  # This takes twice as long as 'available' check
                return True

            # If the total number of bytes was not enough, checks whether serial port has received any additional bytes
            # since the last loop iteration. This is primarily used to reset the timer upon new bytes' reception.
            if previous_additional_bytes < additional_bytes:  # pragma: no cover
                previous_additional_bytes = additional_bytes  # Updates the byte tracker, if necessary
                self._timer.reset()  # Resets the timeout timer as long as the port receives additional bytes

        # If there are not enough bytes across both buffers, returns False.
        return False

    @staticmethod
    @njit(nogil=True, cache=True)  # type: ignore # pragma: no cover # pragma: no cover
    def _parse_packet(
        unparsed_bytes: NDArray[np.uint8],
        start_byte: np.uint8,
        delimiter_byte: np.uint8,
        max_payload_size: np.uint8,
        min_payload_size: np.uint8,
        postamble_size: np.uint8,
        allow_start_byte_errors: bool,
        start_found: bool = False,
        parsed_byte_count: int = 0,
        parsed_bytes: NDArray[np.uint8] = np.empty(0, dtype=np.uint8),
    ) -> tuple[int, int, NDArray[np.uint8], NDArray[np.uint8]]:
        """Parses as much of the packet data as possible using the input unparsed_bytes object.

        This method contains all packet parsing logic, split into 4 distinct stages: resolving the start_byte, resolving
        the packet_size, resolving the encoded payload, and resolving the CRC postamble. It is common for the method to
        not advance through all stages during a single call, requiring multiple calls to fully parse the packet.
        The method is written in a way that supports iterative calls to work on the same packet. This method is not
        intended to be used standalone and should always be called through the _receive_packet() method.

        Notes:
            For this method, the 'packet' refers to the COBS encoded payload + the CRC checksum postamble. While each
            received byte stream also necessarily includes the metadata preamble, the preamble data is used and
            discarded during this method's runtime.

            This method becomes significantly more efficient in use patterns where many bytes are allowed to aggregate
            in the serial port buffer before being evaluated. Due to JIT compilation this method is very fast, and any
            major execution delay typically comes from reading the data from the underlying serial port.

            The returns of this method are designed to support iterative calls to this method. As a minium, the packet
            may be fully resolved (parsed or failed to be parsed) with one call, and, as a maximum, 3 calls may be
            necessary.

            The method uses static integer codes to communicate its runtime status:

            0 - Not enough bytes read to fully parse the packet. The start byte was found, but packet size has not
            been resolved and, therefore, not known.
            1 - Packet fully parsed.
            2 - Not enough bytes read to fully parse the packet. The packet size was resolved, but there were not
            enough bytes to fully parse the packet (encoded payload + crc postamble).
            3 - Not enough bytes read to fully parse the packet. The packet payload was successfully parsed, but there
            were not enough bytes to fully parse the CRC postamble.
            101 - No start byte found, which is interpreted as 'no bytes to read,' as the class is configured to
            ignore start byte errors. Usually, this situation is caused by communication line noise generating
            'noise bytes'.
            102 - No start byte found, which is interpreted as a 'no start byte detected' error case. This status is
            only possible when the class is configured to detect start byte errors.
            103 - Parsed payload_size value is either less than the minimum expected value or above the maximum value.
            This likely indicates packet corruption or communication parameter mismatch between this class and the
            connected Microcontroller.
            104 - Delimiter byte value encountered before reaching the end of the encoded payload data block. It is
            expected that the last byte of the encoded payload is set to the delimiter value and that the value is not
            present anywhere else inside the encoded payload region. Encountering the delimiter early indicates packet
            corruption.
            105 - Delimiter byte value not encountered at the end of the encoded payload data block. See code 104
            description for more details, but this code also indicates packet corruption.

        Args:
            unparsed_bytes: A bytes() object that stores the bytes read from the serial port. If this is the first call
                to this method for a given _receive_packet() method runtime, this object may also include any bytes left
                from the previous _receive_packet() runtime.
            start_byte: The byte-value used to mark the beginning of a transmitted packet in the byte-stream. This is
                used to detect the portion of the stream that stores the data packet.
            delimiter_byte: The byte-value used to mark the end of a transmitted packet in the byte-stream. This is
                used to detect the portion of the stream that stores the data packet and the portion that stores the
                CRC postamble.
            max_payload_size: The maximum size of the payload, in bytes, that can be received. This value cannot
                exceed 254 due to COBS limitations.
            min_payload_size: The minimum size of the payload, in bytes, that can be received. This value cannot be
                less than 1 or above 254 (see above) and cannot exceed the max_payload_size.
            postamble_size: The number of bytes needed to store the CRC checksum. This is determined based on the type
                of the CRC polynomial used by the class.
            allow_start_byte_errors: A boolean flag that determines whether inability to find start_byte should be
                interpreted as having no bytes to read (default, code 101) or as an error (code 102).
            start_found: Iterative argument. When this method is called two or more times, this value can be provided
                to the method to skip resolving the start byte (detecting packet presence). Specifically, it is used
                when a call to this method finds the start byte, but cannot resolve the packet size. Then, during a
                second call, start_byte searching step is skipped.
            parsed_byte_count: Iterative parameter. When this method is called multiple times, this value communicates
                how many bytes out of the expected byte number have been parsed by the previous method runtime.
            parsed_bytes: Iterative parameter. This object is initialized to the expected packet size once it is parsed.
                Multiple method runtimes may be necessary to fully fill the object with parsed data bytes.

        Returns:
            A tuple of four elements. The first element is an integer status code that describes the runtime. The
            second element is the number of packet's bytes processed during method runtime. The third element is a
            bytes' object that stores any unprocessed bytes that remain after method runtime. The fourth element
            is the uint8 array that stores some or all of the packet's bytes.
        """
        # Converts the input 'bytes' object to a numpy array to optimize further buffer manipulations
        total_bytes = unparsed_bytes.size  # Calculates the total number of bytes available for parsing
        processed_bytes = 0  # Tracks how many input bytes are processed during method runtime

        # Stage 1: Resolves the start_byte. Detecting the start byte tells the method the processed byte-stream contains
        # a packet that needs to be parsed.
        if not start_found:
            # Loops over available bytes until start byte is found or the method runs out of bytes to evaluate
            for i in range(total_bytes):
                processed_bytes += 1  # Increments the counter for each evaluated byte

                # If the start byte is found, breaks the loop and sets the start byte acquisition flag to True
                if unparsed_bytes[i] == start_byte:
                    start_found = True
                    break

            # If the loop above terminates without finding the start byte, ends method runtime with the appropriate
            # status code.
            if not start_found:
                # Determines the status code based on whether start byte errors are allowed.
                # If they are allowed, returns 102. Otherwise (default) returns 101.
                if allow_start_byte_errors:
                    status_code = 102  # This will terminate packet reception with an error
                else:
                    status_code = 101  # This will terminate packet reception without an error

                remaining_bytes = np.empty(0, dtype=np.uint8)  # The loop above used all unprocessed bytes
                return status_code, parsed_byte_count, remaining_bytes, parsed_bytes

            # If this stage uses up all unprocessed bytes, ends method runtime with partial success code (0)
            if processed_bytes == total_bytes:
                remaining_bytes = np.empty(0, dtype=np.uint8)  # The loop above used all unprocessed bytes
                return 0, parsed_byte_count, remaining_bytes, parsed_bytes

        # Calculates the size of the COBS-encoded payload (data packet) from the total size of the parsed_bytes
        # array and the crc_postamble. Ensures the value is always non-negative. Relies on the fact that stage 2
        # initializes the parsed_bytes array to have enough space for the COBS-encoded payload and the crc_postamble.
        # Assumes that the default parsed_bytes array is an empty (size 0) array.
        packet_size = max(parsed_bytes.size - int(postamble_size), 0)

        # Stage 2: Resolves the packet_size. Packet size is essential for knowing how many bytes need to be read to
        # fully parse the packet. Additionally, this is used to infer the packet layout, which is critical for the
        # following stages.
        if packet_size == 0:
            # Reads the first available unprocessed byte and checks it for validity. This relies on the fact that
            # valid packets store the payload_size byte immediately after the start_byte.
            payload_size = unparsed_bytes[processed_bytes]

            processed_bytes += 1  # Increments the counter. Has to be done after reading the byte above.

            # Verifies that the payload size is within the expected payload size limits. If payload size is out of
            # bounds, returns with status code 103: Payload size not valid.
            if not min_payload_size <= payload_size <= max_payload_size:
                remaining_bytes = unparsed_bytes[processed_bytes:].copy()  # Returns any remaining unprocessed bytes
                parsed_bytes = np.empty(payload_size, dtype=np.uint8)  # Uses invalid size for the array shape anyway
                return 103, parsed_byte_count, remaining_bytes, parsed_bytes

            # If payload size passed verification, calculates the number of bytes occupied by the COBS-encoded payload
            # and the CRC postamble. Specifically, uses the payload_size and increments it with +2 to account for the
            # overhead and delimiter bytes introduced by COBS-encoding the packet. Also adds the size of the CRC
            # postamble.
            remaining_size = int(payload_size) + 2 + int(postamble_size)

            # Uses the calculated size to pre-initialize the parsed_bytes array to accommodate the encoded payload and
            # the CRC postamble. Subsequently, the size of the array will be used to infer the size of the encoded
            # payload.
            parsed_bytes = np.empty(shape=remaining_size, dtype=np.uint8)

            # If this stage uses up all unprocessed bytes, ends method runtime with partial success code (2)
            if processed_bytes == total_bytes:
                remaining_bytes = np.empty(0, dtype=np.uint8)  # The loop above used all unprocessed bytes
                return 2, parsed_byte_count, remaining_bytes, parsed_bytes
            # Recalculates the packet size to match the size of the expanded array. Otherwise, if all stages are
            # resolved as part of the same cycle, the code below will continue working with the assumption that the
            # packet size is 0.
            packet_size = max(parsed_bytes.size - int(postamble_size), 0)

        # Based on the size of the packet and the number of already parsed packet bytes, calculates the remaining
        # number of bytes. Ensures the resultant value is always non-negative. If this value is 0, stage 3 is skipped.
        remaining_packet_bytes = max((packet_size - parsed_byte_count), 0)

        # Stage 3: Resolves the COBS-encoded payload. This is the variably sized portion of the stream that contains
        # communicated data with some service values.
        if remaining_packet_bytes != 0:
            # Adjusts loop indices to account for bytes that might have been processed prior to this step
            for i in range(processed_bytes, total_bytes):
                # Transfers the evaluated byte from the unparsed buffer into the parsed buffer.
                # Uses parsed_byte_count as writing index to sequentially fill the array with data over potentially
                # multiple iterations of this method
                parsed_bytes[parsed_byte_count] = unparsed_bytes[i]

                processed_bytes += 1  # Increments the processed bytes counter
                parsed_byte_count += 1  # Unlike processed_bytes, this tracker is shared by multiple method calls.
                remaining_packet_bytes -= 1  # Decrements remaining packet bytes counter with each processed byte

                # If the evaluated byte matches the delimiter byte value and this is not the last byte of the encoded
                # payload, the packet is likely corrupted. Returns with error code 104: Delimiter byte encountered too
                # early.
                if unparsed_bytes[i] == delimiter_byte and remaining_packet_bytes != 0:
                    remaining_bytes = unparsed_bytes[processed_bytes:].copy()  # Returns any remaining unprocessed bytes
                    return 104, parsed_byte_count, remaining_bytes, parsed_bytes

                # If the evaluated byte is a delimiter byte value and this is the last byte of the encoded payload, the
                # payload is fully parsed. Gracefully breaks the loop and advances to the CRC postamble parsing stage.
                if unparsed_bytes[i] == delimiter_byte and remaining_packet_bytes == 0:
                    break

                # If the last evaluated payload byte is not a delimiter byte value, this also indicates that the
                # packet is likely corrupted. Returns with code 105: Delimiter byte not found.
                if remaining_packet_bytes == 0 and unparsed_bytes[i] != delimiter_byte:
                    remaining_bytes = unparsed_bytes[processed_bytes:].copy()  # Returns any remaining unprocessed bytes
                    return 105, parsed_byte_count, remaining_bytes, parsed_bytes

            # If this stage uses up all unprocessed bytes, ends method runtime with partial success code (2)
            if total_bytes - processed_bytes == 0:
                remaining_bytes = np.empty(0, dtype=np.uint8)  # The loop above used all unprocessed bytes
                return 2, parsed_byte_count, remaining_bytes, parsed_bytes

        # If the packet is fully resolved at this point, terminates the runtime before advancing to stage 4. While this
        # is likely not possible, this guard would catch a case where the CRC payload is fully resolved when the
        # execution reaches this point.
        if parsed_bytes.size == parsed_byte_count:
            remaining_bytes = unparsed_bytes[processed_bytes:].copy()
            return 1, parsed_byte_count, remaining_bytes, parsed_bytes
        # Otherwise, determines how many CRC bytes are left to parse
        remaining_crc_bytes = parsed_bytes.size - parsed_byte_count

        # Stage 4: Resolves the CRC checksum postamble. This is the static portion of the stream that follows the
        # encoded payload. This is used for payload data integrity verification.
        for i in range(processed_bytes, total_bytes):
            # Transfers the evaluated byte from the unparsed buffer into the parsed buffer
            parsed_bytes[parsed_byte_count] = unparsed_bytes[i]

            processed_bytes += 1  # Increments the processed bytes counter
            parsed_byte_count += 1  # Increments the parsed packet and postamble byte tracker
            remaining_crc_bytes -= 1  # Decrements remaining CRC bytes counter with each processed byte

            # If all crc bytes have been parsed, the packet is also fully parsed. Returns with success code 1.
            if remaining_crc_bytes == 0:
                remaining_bytes = unparsed_bytes[processed_bytes:].copy()
                return 1, parsed_byte_count, remaining_bytes, parsed_bytes

        # The only way to reach this point is when the CRC parsing loop above escapes due to running out of bytes to
        # process without fully parsing the postamble. Returns with partial success code (2)
        remaining_bytes = np.empty(0, dtype=np.uint8)  # The loop above used all unprocessed bytes
        return 3, parsed_byte_count, remaining_bytes, parsed_bytes

    @staticmethod
    @njit(nogil=True, cache=True)  # type: ignore # pragma: no cover
    # pragma: no cover
    def _validate_packet(
        reception_buffer: NDArray[np.uint8],
        packet_size: int,
        cobs_processor: _COBSProcessor,
        crc_processor: _CRCProcessor,
        delimiter_byte: np.uint8,
        postamble_size: np.uint8,
    ) -> int:
        """Validates the packet by passing it through a CRC checksum calculator, decodes the COBS-encoded payload, and
        saves it back to the input reception_buffer.

        Both the CRC checksum calculation and COBS decoding act as data integrity verification steps. Jointly, they
        make it very unlikely that a corrupted packet advances to further data processing steps. If this method runs
        successfully, the payload will be available for consumption via read_data() method.

        Notes:
            This method expects the packet to be stored inside the _reception_buffer and will write the decoded
            payload back to the _reception_buffer if method runtime succeeds. This allows optimizing memory access and
            method runtime speed.

            The method uses the following CRC property: running a CRC calculation on the data with appended CRC
            checksum (for that data) will always return 0. This stems from the fact that CRC checksum is the remainder
            of dividing the data by the CRC polynomial. For checksums that use multiple bytes, it is essential that the
            receiver and the sender use the same order of bytes when serializing and deserializing the CRC postamble,
            for this method to run as expected.

        Args:
            reception_buffer: The buffer that stores the packet to be verified and decoded. If method runtime is
                successful, a portion of the buffer will be overwritten to store the decoded payload. This should be
                the reception buffer of the caller class.
            packet_size: The number of bytes that makes up the packet to be verified. It is expected that payload only
                uses a portion of the input reception_buffer.
            cobs_processor: The inner _COBSProcessor jitclass instance. The instance can be obtained by using
                '.processor' property of the COBSProcessor wrapper class.
            crc_processor: The inner _CRCProcessor jitclass instance. The instance can be obtained by using '.processor'
                property of the RCProcessor wrapper class.
            delimiter_byte: The byte-value used to mark the end of each received packet's payload region.
            postamble_size: The CRC postamble byte-size for the processed packet.

        Returns:
             A positive integer (>= 1) that stores the size of the decoded payload, if the method succeeds. Integer
             error code 0, if the method fails.
        """
        # Extracts the packet from the reception buffer. The methods below assume the entirety of the input buffer
        # stores the data to be processed, which is likely not true for the input reception buffer. The reception buffer
        # is statically initialized to have enough space to store the maximum supported payload size.
        packet = reception_buffer[:packet_size]

        # Calculates the CRC checksum for the packet. Since the packet includes the CRC checksum postamble, running the
        # CRC calculation on the data + checksum should always return 0.
        checksum = crc_processor.calculate_crc_checksum(packet)

        # Verifies that the checksum calculation method ran successfully. There are two distinct failure cases here.
        # The first is an error during the CRC calculator method runtime (unlikely), indicated by the crc_processor
        # status. The second is packet corruption inferred from the calculated checksum not being 0. In either case,
        # returns error code 0.
        if crc_processor.status != crc_processor.checksum_calculated or checksum != 0:
            return 0

        # Removes the CRC bytes from the end of the packet, as they are no longer necessary after the CRC verification
        packet = packet[: packet.size - int(postamble_size)]

        # Decodes the COBS-encoded payload from the packet
        payload = cobs_processor.decode_payload(packet=packet, delimiter=delimiter_byte)

        # If the returned payload is an empty array, returns 0 to indicate that the COBS decoding step failed.
        if payload.size == 0:
            return 0

        # If decoding succeeds, copies the decoded payload over to the reception buffer and returns the positive size
        # of the payload to caller to indicate runtime success. The returned size should always be above 0 if this
        # stage is reached
        reception_buffer[: payload.size] = payload
        return payload.size
