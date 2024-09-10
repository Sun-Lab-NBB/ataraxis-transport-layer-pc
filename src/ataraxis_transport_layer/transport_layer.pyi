from typing import Any

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

from ataraxis_transport_layer.helper_modules import (
    SerialMock as SerialMock,
    CRCProcessor as CRCProcessor,
    COBSProcessor as COBSProcessor,
    _CRCProcessor as _CRCProcessor,
    _COBSProcessor as _COBSProcessor,
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
    _port: Incomplete
    _crc_processor: Incomplete
    _cobs_processor: Incomplete
    _timer: Incomplete
    _start_byte: Incomplete
    _delimiter_byte: Incomplete
    _timeout: Incomplete
    _allow_start_byte_errors: Incomplete
    _postamble_size: Incomplete
    _max_tx_payload_size: Incomplete
    _max_rx_payload_size: Incomplete
    _transmission_buffer: Incomplete
    _reception_buffer: Incomplete
    _minimum_packet_size: Incomplete
    _bytes_in_transmission_buffer: int
    _bytes_in_reception_buffer: int
    _leftover_bytes: Incomplete
    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        polynomial: np.uint8 | np.uint16 | np.uint32 = ...,
        initial_crc_value: np.uint8 | np.uint16 | np.uint32 = ...,
        final_crc_xor_value: np.uint8 | np.uint16 | np.uint32 = ...,
        maximum_transmitted_payload_size: int = 254,
        minimum_received_payload_size: int = 1,
        start_byte: int = 129,
        delimiter_byte: int = 0,
        timeout: int = 20000,
        *,
        test_mode: bool = False,
        allow_start_byte_errors: bool = False,
    ) -> None: ...
    def __del__(self) -> None:
        """Ensures proper resource release prior to garbage-collecting class instance."""
    def __repr__(self) -> str:
        """Returns a string representation of the SerialTransportLayer class instance."""
    @staticmethod
    def list_available_ports() -> tuple[dict[str, int | str | Any], ...]:
        """Provides the information about each serial port addressable through the pySerial library.

        This method is intended to be used for discovering and selecting the serial port 'names' to use with this
        class.

        Returns:
            A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
            port.

        """
    @property
    def available(self) -> bool:
        """Returns True if enough bytes are available from the serial port to justify attempting to receive a packet."""
    @property
    def transmission_buffer(self) -> NDArray[np.uint8]:
        """Returns a copy of the transmission buffer numpy array.

        This buffer stores the 'staged' data to be sent to the Microcontroller. Use this method to safely access the
        contents of the buffer in a snapshot fashion.
        """
    @property
    def reception_buffer(self) -> NDArray[np.uint8]:
        """Returns a copy of the reception buffer numpy array.

        This buffer stores the decoded data received from the Microcontroller. Use this method to safely access the
        contents of the buffer in a snapshot fashion.
        """
    @property
    def bytes_in_transmission_buffer(self) -> int:
        """Returns the number of payload bytes stored inside the transmission_buffer."""
    @property
    def bytes_in_reception_buffer(self) -> int:
        """Returns the number of payload bytes stored inside the reception_buffer."""
    def reset_transmission_buffer(self) -> None:
        """Resets the transmission buffer bytes tracker to 0.

        This does not physically alter the buffer in any way, but makes all data inside the buffer 'invalid'. This
        approach to 'resetting' the buffer by overwriting, rather than recreation, is chosen for higher memory
        efficiency and runtime speed.
        """
    def reset_reception_buffer(self) -> None:
        """Resets the reception buffer bytes tracker to 0.

        This does not physically alter the buffer in any way, but makes all data inside the buffer 'invalid'. This
        approach to 'resetting' the buffer by overwriting, rather than recreation, is chosen for higher memory
        efficiency and runtime speed.
        """
    def write_data(
        self,
        data_object: np.uint8
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
        | NDArray[
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
        ]
        | type,
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
    @staticmethod
    def _write_scalar_data(
        target_buffer: NDArray[np.uint8],
        scalar_object: np.uint8
        | np.uint16
        | np.uint32
        | np.uint64
        | np.int8
        | np.int16
        | np.int32
        | np.int64
        | np.float32
        | np.float64
        | np.bool,
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
    @staticmethod
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
    def read_data(
        self,
        data_object: np.uint8
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
        | NDArray[
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
        ]
        | type,
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
    @staticmethod
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
    @staticmethod
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
    @staticmethod
    def _parse_packet(
        read_bytes: bytes,
        start_byte: np.uint8,
        max_payload_size: np.uint8,
        postamble_size: int | np.unsignedinteger[Any],
        allow_start_byte_errors: bool,
        start_found: bool = False,
        packet_size: int = 0,
        packet_bytes: NDArray[np.uint8] = ...,
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
    @staticmethod
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
