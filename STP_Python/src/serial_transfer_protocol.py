"""This file stores the SerializedTransferProtocol class, which provides the high-level API that encapsulates all
methods necessary to bidirectionally communicate with microcontroller devices running the C-version of this library.
Recently, teh class has been updated to also support ZeroMQ-based communication with non-microcontroller devices
running the C- or Python - version of this library, making it a universal communication protocol that can connect most
devices frequently used in science applications. All features of the class are available through 4 main methods:
write_data(), send_data(), receive_data() and read_data(). See method and class docstrings for more information.
"""

import textwrap
from dataclasses import fields, is_dataclass
from typing import Optional, Type, Union

import numpy as np
import serial
from numba import njit
from serial.tools import list_ports

from src.helper_modules import COBSProcessor, CRCProcessor, ElapsedTimer, SerialMock


class SerializedTransferProtocol:
    """Provides the high-level API that encapsulates all methods necessary to bidirectionally communicate with
    microcontrollers running the C-version of the SerializedTransferProtocol library.

    This class functions as a central hub that calls various internal and external helper methods and fully encapsulates
    the serial port interface (via pySerial third-party library). Most of this class is hidden behind private attributes
    and methods, and any part of the class that is publicly exposed is generally safe to use and should be sufficient
    to realize the full functionality of the library.

    Notes:
        This class contains 4 main methods: write_data(), send_data(), receive_data() and read_data(). Write and read
        methods are used to manipulate the class-specific 'staging' buffers that aggregate the data to be sent to the
        microcontroller and store the data received from the microcontroller. Send and receive methods operate on the
        class buffers and trigger the sequences of steps needed to construct and send a serial packet to the controller
        or receive and decode the data sent as a packet from the controller. See method descriptions for more details.

        Most class inputs and arguments are configured to require a numpy scalar or array input to enforce typing,
        which is not done natively in python. Type enforcement is notably 'unpythonic', but very important for this
        library as it communicates with microcontrollers that do use a strictly typed ecosystem (C++). Additionally,
        enforcing typing allows using efficient numpy and numba operations to optimize most of the custom library code
        to run at C++ speeds, which is one of the notable advantages of this library.

        All class attributes are private by design and documented primarily for the developers willing to contribute to
        the library or use library assets in their own projects. Users can safely ignore that entire section.

    Attributes:
        __port: Depending on the test_mode flag, this is either a SerialMock object or a pySerial Serial object. During
            'production' runtime, this object provides low-level access to USB / UART ports using OS-specific APIs.
        __crc_processor: CRCProcessor class object that provides the methods to manipulate (calculate and serialize)
            CRC checksums for the transmitted and received data packets.
        __cobs_processor: COBSProcessor class object that provides the methods to encode and decode the payloads to and
            from the transmitted and received packets.
        __timer: ElapsedTimer class object that provides an easy-to-use interface to measure time intervals, most
            notably used to detect and escape staled packet reception sequences.
        __start_byte: The value that marks the beginning of all transmitted and received packets. Encountering this
            value is the only condition for entering packet reception mode.
        __delimiter_byte: The value that marks the end of all transmitted and received packets. This value is necessary
            for the proper functioning of the microcontroller's packet reception sequence, and it's presence at the end
            of the packet is used as the packet integrity marker.
        __timeout: The number of microseconds to wait between receiving any two consecutive bytes of the packet. Note,
            this value specifically concerns the interval between two bytes of the packet, not the whole packet.
        __allow_start_byte_errors: A boolean flag that determines whether to raise errors when the start byte is not
            found among the available bytes when receive_data() method is called. The default behavior is to ignore
            such errors as communication line noise can create 'fake' data bytes that are routinely cleared by the
            reception method. However, enabling this option may be helpful in certain debugging scenarios.
        __max_tx_payload_size: The maximum number of bytes that are expected to be transmitted as a single payload. This
            number is statically capped at 254 bytes due to COBS encoding. The value of this attribute should always
            match the value of the similar attribute used in the microcontroller library, which is why it is
            user-addressable through initialization arguments.
        __max_rx_payload_size: The maximum number of bytes expected to be received from the microcontroller as a single
            payload. Similar restrictions and considerations apply as with the __max_tx_payload_size. Since PCs do not
            have the same memory restrictions as controllers, this is always statically set to 254 (maximum possible
            size).
        __postamble_size: The byte-size of the CRC checksum. Calculated automatically based on the number of bytes used
            by the polynomial argument. This attribute is used to optimize packet reception and data buffering and
            de-buffering.
        __transmission_buffer: The private buffer used to stage the data to be sent to the microcontroller.
            Specifically, the write_data() method adds input data to this buffer as a sequence of bytes. When
            send_data() method is called, the contents of the buffer are packaged and sent to the microcontroller.
            The buffer is statically allocated at instantiation and relies on the __bytes_in_transmission_buffer tracker
            to specify which portion of the buffer is filled with payload bytes any given time.
        __reception_buffer: The private buffer that stores the decoded data received from the microcontroller. The
            buffer is filled by calling receive_data() method, and the received payload can then be read into objects by
            calling read_data() method. Like __transmission_buffer, the __reception buffer is statically allocated at
            instantiation and relies on the __bytes_in_reception_buffer tracker to specify which portion of the buffer
            is used at any given time to store the received payload.
        __bytes_in_transmission_buffer: Tracks how many bytes (relative to index 0) of the __transmission_buffer are
            currently used to store the payload to be transmitted. This allows efficient data buffering by overwriting
            and working with the minimal number of bytes necessary to store the payload. When the __transmission_buffer
            is 'reset' the buffer itself is not changed in any way, but this tracker is set to 0. The tracker is
            conditionally incremented by each write_data() method call and reset by send_data() method calls. This
            attribute can also be reset using reset_transmission_buffer() method.
        __bytes_in_reception_buffer: Same as __bytes_in_transmission_buffer, but for the __reception_buffer. However,
            read_data() method calls do not modify the value of this attribute. Only receive_data() or
            reset_reception_buffer() methods can modify this attribute.
        __leftover_bytes: A private buffer used to preserve any 'unconsumed' bytes that were read from the serial port
            but not used to reconstruct the payload sent from the microcontroller. Since pySerial read() method calls
            are very costly, to improve runtime speed, this class does everything it can to minimize the number of
            such calls. Part of this strategy involves always reading as many bytes as available and 'stashing' any
            'excessive' bytes to be re-used by the next receive_data() calls.
        __accepted_numpy_scalars: A tuple of numpy types (classes) that can be used as scalar inputs or as 'dtype'
            fields of the numpy arrays that are provided to the read_data() and write_data() methods. This class only
            works with these datatypes and will raise errors if it encounters any other input. The only exceptions to
            this rule are dataclasses made up entirely of supported numpy scalars or arrays (Python's equivalent for
            C-structures), such dataclasses are also valid inputs and will be processed correctly.
        __minimum_packet_size: The minimum number of bytes that can represent a valid packet. This number is statically
            determined at class initialization and is subsequently used to optimize packet reception logic (minimizes
            wasting time by calling 'expensive' methods that evaluate the state of the communication hardware (reception
            buffers, etc.). The user can optionally overwrite this number (via minimum_received_payload_size argument)
            instead of using the baseline calculation to further minimize the time wasted on running futile serial port
            operations, but the minimum value this can take is 5 + postamble_size (1 payload byte, 2 preamble bytes,
            2 COBS bytes and however many bytes are needed to store the CRC postamble). Note, when user overwrites the
            minimum_received_payload_size, specifically the '1' payload value is changed, keeping the other values the
            same.

        Args:
            port: The name of the serial port to connect to, e.g.: 'COM3' or '/dev/ttyUSB0'. You can use the
                list_available_ports() class method to obtain a list of discoverable serial port names.
            baudrate: The baudrate to be used to communicate with the microcontroller. Should match the value used by
                the microcontroller for UART ports, ignored for USB ports. Defaults to 115200, which is
                a fairly fast rate supported by most microcontroller boards (many can use faster rates!).
            polynomial: The polynomial to use for the generation of the CRC lookup table. Can be provided as a HEX
                number (e.g., 0x1021). Currently only non-reversed polynomials of numpy uint8, uint16 and uint32
                datatype are supported. Defaults to 0x1021 (CRC-16 CCITT-FALSE).
            initial_crc_value: The initial value to which the CRC checksum variable is initialized during calculation.
                This value depends on the chosen polynomial algorithm ('polynomial' argument) and should use the same
                datatype as the polynomial argument. It can be provided as a HEX number (e.g., 0xFFFF).
                Defaults to 0xFFFF (CRC-16 CCITT-FALSE).
            final_crc_xor_value: The final XOR value to be applied to the calculated CRC checksum value. This value
                depends on the chosen polynomial algorithm ('polynomial' argument) and should use the same datatype as
                the polynomial argument. It can be provided as an appropriately sized HEX number (e.g., 0x0000).
                Defaults to 0x0000 (CRC-16 CCITT-FALSE).
            maximum_transmitted_payload_size: The maximum number of bytes that are expected to be transmitted to the
                microcontroller as a single payload. This HAS to match the maximum_received_payload_size value used by
                the microcontroller to ensure that it has the buffer space to accept all payloads. In fact, the only
                reason this parameter is used-addressable is precisely so that the user ensures it matches the
                microcontroller settings. Defaults to 254.
            minimum_received_payload_size: The minimum number of bytes that can be expected to be received from the
                microcontroller as a single payload. This number is used to calculate the threshold for entering
                incoming data reception cycle to minimize the number of calls made to costly operations required to
                receive data. Cannot exceed 254 bytes and cannot be less than 1 byte.
            start_byte: The value used to mark the beginning of the packet. Has to match the value used by the
                microcontroller. Can be any value in the uint8 range (0 to 255). It is advised to use the value that is
                unlikely to occur as noise. Defaults to 129.
            delimiter_byte: The value used to denote the end of the packet. Has to match the value used by the
                microcontroller. Due to how COBS works, it is advised to use '0' as the delimiter byte. Zero is the only
                value guaranteed to be exclusive when used as a delimiter. Defaults to 0.
            timeout: The maximum number of microseconds(!) that can separate receiving any two consecutive bytes of the
                packet. This is used to detect and resolve stale packet reception attempts. While this defaults to 20000
                (20 ms), the library can resolve intervals in the range of dozen(s) of microseconds, so the number can
                be made considerably smaller than that. Defaults to 20000.
            test_mode: The boolean flag that determines whether the library uses a real pySerial Stream class or a
                helper StreamMock class. Only used during testing and should always be disabled otherwise. Defaults to
                False.
            allow_start_byte_errors: The boolean flag that determines whether the class raises errors when it is unable
                to find the start value in the incoming byte-stream. It is advised to keep this set to False for most
                use cases. This is because it is fairly common to see noise-generated bytes inside the reception buffer
                that are then silently cleared by the algorithm until a real packet becomes available. However, enabling
                this option may be helpful for certain debugging scenarios. Defaults to False.

        Raises:
            ValueError: If maximum_transmitted_payload_size argument exceeds 254 (this is the hard limit due to COBS
                encoding). If the minimum_received_payload_size is below 1 or above 254. Also, if the start_byte is set
                to the same value as the delimiter_byte.
            SerialException: Originate from the pySerial class used to connect to the USB / UART port interface.
                Primarily, these exceptions will be raised when the requested port does not exist or the user lacks
                permissions to overtake whatever uses the port at the time (or access it at all). See pySerial
                documentation if you need help understanding what the exceptions mean.
    """

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
            test_mode: bool = False,
            allow_start_byte_errors: bool = False,
    ) -> None:
        # Ensures that the class is initialized properly by catching possible class configuration argument errors.
        if start_byte == delimiter_byte:
            error_message = (
                f"Unable to initialize SerializedTransferProtocol class. 'start_byte' and 'delimiter_byte' arguments "
                f"cannot be set to the same value ({start_byte})."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif maximum_transmitted_payload_size > 254:
            error_message = (
                f"Unable to initialize SerializedTransferProtocol class. 'maximum_transmitted_payload_size' argument "
                f"value ({maximum_transmitted_payload_size}) cannot exceed 254."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif not (1 <= minimum_received_payload_size <= 254):
            error_message = (
                f"Unable to initialize SerializedTransferProtocol class. 'minimum_received_payload_size' argument "
                f"value ({minimum_received_payload_size}) must be between 1 and 254 (inclusive)."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Initializes subclasses
        if not test_mode:
            self.__port = serial.Serial(port, baudrate, timeout=0)
        else:
            self.__port = SerialMock()
        self.__crc_processor = CRCProcessor(polynomial, initial_crc_value, final_crc_xor_value)
        self.__cobs_processor = COBSProcessor()
        self.__timer = ElapsedTimer("us")  # Times various packet reception steps to timeout if packet reception stales

        # Initializes user-defined attributes
        self.__start_byte = np.uint8(start_byte)
        self.__delimiter_byte = np.uint8(delimiter_byte)
        self.__timeout = np.uint64(timeout)
        self.__allow_start_byte_errors = allow_start_byte_errors

        # Initializes automatically calculated and static attributes
        self.__max_tx_payload_size = maximum_transmitted_payload_size  # Capped at 254 due to COBS encoding
        self.__max_rx_payload_size = 254  # Capped at 254 due to COBS encoding
        self.__postamble_size = self.__crc_processor.crc_byte_length
        # The buffers are always set to maximum payload size + 2 + postamble to be large enough to store packets. This
        # is needed to support __parse_packet() method functioning.
        self.__transmission_buffer = np.empty(self.__max_tx_payload_size + 2 + self.__postamble_size, dtype=np.uint8)
        self.__reception_buffer = np.empty(self.__max_rx_payload_size + 2 + self.__postamble_size, dtype=np.uint8)
        self.__bytes_in_transmission_buffer = 0
        self.__bytes_in_reception_buffer = 0
        self.__leftover_bytes = bytes()  # Placeholder
        self.__accepted_numpy_scalars = (
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
            np.bool_,
        )  # Used to verify scalar and numpy array input datatypes and for error messages
        self.__minimum_packet_size = max(1, minimum_received_payload_size) + 4 + self.__postamble_size

        # Opens (connects to) the serial port. Cycles closing and opening to ensure the port is opened if it exists at
        # the cost of potentially non-graciously replacing whatever is using the port at the time of instantiating
        # SerializedTransferProtocol class. This may not be required on all platforms, but the original developer used
        # the combination fo platformio and Windows which was notoriously bad at releasing the port ownership after
        # uploading controller firmware.
        self.__port.close()
        self.__port.open()

    def __del__(self) -> None:
        # Closes the port before deleting the class instance. Not strictly required, but seeing how unreliable Windows
        # is about releasing port handles it doesn't hurt, to say the least.
        self.__port.close()

    def __repr__(self) -> str:
        if isinstance(self.__port, serial.Serial):
            repr_message = (
                f"SerializedTransferProtocol(port='{self.__port.name}', baudrate={self.__port.baudrate}, polynomial="
                f"{self.__crc_processor.polynomial}, initial_crc_value={self.__crc_processor.initial_crc_value}, "
                f"final_xor_value={self.__crc_processor.final_xor_value}, crc_byte_length="
                f"{self.__crc_processor.crc_byte_length} start_byte={self.__start_byte}, delimiter_byte="
                f"{self.__delimiter_byte}, timeout={self.__timeout} us, allow_start_byte_errors="
                f"{self.__allow_start_byte_errors}, maximum_tx_payload_size = {self.__max_tx_payload_size}, "
                f"maximum_rx_payload_size={self.__max_rx_payload_size})"
            )
        else:
            repr_message = (
                f"SerializedTransferProtocol(port & baudrate=MOCKED, polynomial={self.__crc_processor.polynomial}, "
                f"initial_crc_value={self.__crc_processor.initial_crc_value}, final_xor_value="
                f"{self.__crc_processor.final_xor_value}, crc_byte_length={self.__crc_processor.crc_byte_length}, "
                f"start_byte={self.__start_byte}, delimiter_byte={self.__delimiter_byte}, timeout={self.__timeout} us, "
                f"allow_start_byte_errors={self.__allow_start_byte_errors}, maximum_tx_payload_size="
                f"{self.__max_tx_payload_size}, maximum_rx_payload_size={self.__max_rx_payload_size})"
            )
        return textwrap.fill(repr_message, width=120, break_long_words=False, break_on_hyphens=False)

    @staticmethod
    def list_available_ports() -> None:
        """Surveys and prints the information about each serial port discoverable using pySerial library.

        Notes:
            You can use the 'name' of any such port as the 'port' argument for the initializer function of this class.
        """
        # Obtains the list of port objects visible to the pySerial library.
        available_ports = list_ports.comports()

        # Prints the information about each port using terminal.
        print(f"Available serial ports:")
        for num, port in enumerate(available_ports, start=1):
            print(
                f"{num}. Name: '{port.name}', Device: '{port.device}', PID:{port.pid}, Description: "
                f"'{port.description}'"
            )

    @property
    def available(self) -> bool:
        """Returns True if enough bytes are available from the serial port to justify attempting to receive a packet."""

        # in_waiting is twice as fast as using the read() method. The outcome of this check is capped at the minimum
        # packet size to minimize the chance of having to call read() more than once. It also factors in any
        # leftover bytes to ensure there is no stalling when more data than necessary, so this will reliably return
        # 'true' when there is a high-0enough chance to receive a packet.
        return self.__port.in_waiting + len(self.__leftover_bytes) > self.__minimum_packet_size

    @property
    def transmission_buffer(self) -> np.ndarray:
        """Returns a copy of the private __transmission_buffer numpy array."""
        return self.__transmission_buffer.copy()

    @property
    def reception_buffer(self) -> np.ndarray:
        """Returns a copy of the private __reception_buffer numpy array."""
        return self.__reception_buffer.copy()

    @property
    def bytes_in_transmission_buffer(self) -> int:
        """Returns the number of payload bytes stored inside the private __transmission_buffer array."""
        return self.__bytes_in_transmission_buffer

    @property
    def bytes_in_reception_buffer(self) -> int:
        """Returns the number of payload bytes stored inside the private __reception_buffer array."""
        return self.__bytes_in_reception_buffer

    def reset_transmission_buffer(self):
        """Resets the __bytes_in_transmission_buffer to 0. Note, does not physically alter the buffer in any way and
        only resets the __bytes_in_transmission_buffer tracker."""
        self.__bytes_in_transmission_buffer = 0

    def reset_reception_buffer(self):
        """Resets the __bytes_in_reception_buffer to 0. Note, does not physically alter the buffer in any way and
        only resets the __bytes_in_reception_buffer tracker."""
        self.__bytes_in_reception_buffer = 0

    def write_data(
            self,
            data_object: Union[np.unsignedinteger, np.signedinteger, np.floating, np.bool_, np.ndarray, Type],
            start_index: Optional[int] = None,
    ) -> int:
        """Writes the input data_object to the __transmission_buffer, starting at the specified start_index.

        This method acts as a wrapper for specific private methods that are called depending on the input data_object
        type. If the object is valid and the buffer has enough space to accommodate the object, it will be translated to
        bytes and written to the buffer at the start_index. All bytes written via this method become part of the payload
        that will be sent to the microcontroller when send_data() method is called.

        Notes:
            At this time, the method only works with valid numpy scalars and arrays as well as python dataclasses
            entirely made out of valid numpy types. Using numpy rather than standard python types increases runtime
            speed (when combined with other optimization steps) and enforces strict typing (critical for microcontroller
            communication, as most controllers use strictly typed C++ / C languages).

            The method automatically updates the local __bytes_in_transmission_buffer tracker if the write operation
            increases the total number of payload bytes stored inside the buffer. If the method is used (via a specific
            start_index input) to overwrite already counted data, it will not update the tracker variable. The only way
            to reduce the value of the tracker is to call the reset_transmission_buffer() method to reset it to 0 or to
            call the send_data() method to send the data to the microcontroller, automatically resetting the tracker in
            the process.

            The maximum runtime speed of this method is achieved when writing data as numpy arrays, which is optimized
            to a single write operation. The minimum runtime speed is achieved by writing dataclasses, as it involves
            slow python looping over dataclass attributes. Choose the input format based on your speed and convenience
            requirements. When writing dataclasses, all attributes will be serialized and written as a consecutive data
            block to the same portion of the buffer (which mimics how this is done in the microcontroller library
            version for C++ structures).

        Args:
            data_object: A numpy scalar or array object or a python dataclass made entirely out of valid numpy objects.
                Supported numpy types are: uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64,
                and bool_. Additionally, arrays have to be 1-dimensional and not empty to be supported.
            start_index: Optional. The index inside the __transmission_buffer (0 to 253) at which to start writing the
                data. If set to None, the method will automatically use the __bytes_in_transmission_buffer tracker value
                to append the data to the end of the already written payload. Defaults to None.

        Returns:
            The index inside the __transmission buffer that immediately follows the last index of the buffer to
            which the data was written. This index can be used as the start_index input for chained write operation
            calls to iteratively and continuously write data to the buffer.

        Raises:
            TypeError: If the input object is not a supported numpy scalar, numpy array, or python dataclass.
            ValueError: Raised if writing the input object is not possible as that would require writing outside the
                __transmission_buffer boundaries. Also raised when multidimensional or empty numpy arrays are
                encountered.
            RuntimeError: If the error-resolving mechanism based on the value of the end_index is not able to
                resolve the error code. This should not really occur, so this is more of a static guard to aid
                developers.

        """

        # Pre-initializes the end index tracker.
        end_index = -10  # Initializes to a specific negative value that is not a valid index or runtime error code

        # Resolves the start_index input, ensuring it is a valid integer value if start_index is left at the default
        # None value
        if start_index is None:
            start_index = self.__bytes_in_transmission_buffer

        # If the input object is a supported numpy scalar, calls the scalar data writing method.
        if isinstance(data_object, self.__accepted_numpy_scalars):
            end_index = self.__write_scalar_data(
                self.__transmission_buffer, data_object, data_object.nbytes, start_index
            )

        # If the input object is a numpy array, first ensures that it's datatype matches one of the accepted scalar
        # numpy types and, if so, calls the array data writing method.
        elif isinstance(data_object, np.ndarray):
            if data_object.dtype in self.__accepted_numpy_scalars:
                end_index = self.__write_array_data(self.__transmission_buffer, data_object, start_index)

        # If the input object is a python dataclass, iteratively loops over each field of the class and recursively
        # calls write_data() to write each attribute of the class to the buffer. This should support nested dataclasses
        # if needed. This implementation supports using this function for any dataclass that stores numpy scalars or
        # arrays, replicating the handling of structures as done in the microcontroller version of this library.
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
                # start_index), breaks the loop to handle the error at the bottom of this method
                if local_index < start_index:
                    break

            # Once the loop is over (due to break or having processed all class fields), sets the end_index to the
            # final recorded local_index value
            end_index = local_index

        # Unsupported input error
        else:
            error_message = (
                f"Unsupported input data_object type ({type(data_object)}) encountered when writing data "
                f"to __transmission_buffer. At this time, only the following numpy scalar or array types are "
                f"supported: {self.__accepted_numpy_scalars}. Alternatively, a dataclass with all attributes set to "
                f"supported numpy scalar or array types is also supported."
            )

            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If the end_index exceeds the start_index, that means that an appropriate write operation was executed
        # successfully. In that case, updates the __bytes_in_transmission_buffer tracker if necessary and returns the
        # end index to caller to indicate runtime success.
        if end_index > start_index:
            # Sets the __bytes_in_transmission_buffer tracker variable to the maximum of its current value and the
            # index that immediately follows the final index of the buffer that was overwritten with he input data.
            # This only increases the tracker value if write operation increased the size of the payload and also
            # prevents the tracker value from decreasing.
            self.__bytes_in_transmission_buffer = max(self.__bytes_in_transmission_buffer, end_index)

            return end_index  # Returns the end_index and not the payload size to support chained overwrite operations

        # If the index is set to code 0, that indicates that the buffer does not have space to accept the written data
        # starting at the start_index.
        elif end_index == 0:
            error_message = (
                f"Insufficient buffer space to write the data to the __transmission_buffer starting at the index "
                f"'{start_index}'. Specifically, given the data size of '{data_object.nbytes}' bytes, the required "
                f"buffer size is '{start_index + data_object.nbytes}' bytes, but the available size is "
                f"'{self.__transmission_buffer.size}' bytes."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If the index is set to code -1, that indicates that a multidimensional numpy array was provided as input,
        # but only flat arrays are allowed
        elif end_index == -1:
            error_message = (
                f"A multidimensional numpy array with {data_object.ndim} dimensions encountered when writing "
                f"data to __transmission_buffer. At this time, only one-dimensional (flat) arrays are supported."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If the index is set to code -2, that indicates that an empty numpy array was provided as input, which does
        # not make sense and therefore is likely an error. Also, empty arrays are explicitly not valid in C/C++, so
        # this is also against language rules to provide them with an intention to send that data to microcontroller
        # running C.
        elif end_index == -2:
            error_message = (
                f"An empty (size 0) numpy array encountered when writing data to __transmission_buffer. Writing empty "
                f"arrays is not supported."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If the end_index is not resolved properly, catches and raises a runtime error
        else:
            error_message = (
                f"Unknown end_index-communicated error code ({end_index}) encountered when writing data "
                f"to __transmission_buffer."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    @staticmethod
    @njit
    def __write_scalar_data(
            target_buffer: np.ndarray,
            data_object: Union[np.unsignedinteger, np.signedinteger, np.floating, np.bool_],
            object_size: int,
            start_index: int,
    ) -> int:
        """Converts input numpy scalars to byte-sequences and writes them to the __transmission_buffer.

        Notes:
            Uses static integer codes to indicate runtime errors.

        Args:
            target_buffer: The buffer to which the data will be written. This should be the class-specific private
                __transmission_buffer array.
            data_object: The scalar numpy object to be written to the __transmission_buffer. Can be any supported numpy
                scalar type.
            object_size: The byte-size of the data object to be written. It should be calculated by the wrapper as numba
                does not know how to use .size attributes of numpy scalar types (see write_data() documentation for more
                details).
            start_index: The index inside the __transmission_buffer (0 to 253) at which to start writing the data.

        Returns:
            The positive index inside the __transmission_buffer that immediately follows the last index of the buffer to
            which the data was written. Returns 0 if the buffer does not have enough space to accommodate the data
            written at the start_index.
        """

        # Calculates the required space inside the buffer to store the data inserted at the start_index
        required_size = start_index + object_size

        # If the space to store the data extends outside the available transmission_buffer boundaries, returns 0.
        if required_size > target_buffer.size:
            return 0

        # Writes the scalar data to the buffer. Uses a two-step conversion where the scalar is first cast as an array,
        # and then the array is used as a buffer from which the bytes representing the data are fed into the
        # transmission buffer. This allows properly handling any supported numpy scalar byte-size.
        target_buffer[start_index:required_size] = np.frombuffer(np.array([data_object]), dtype=np.uint8)

        # Returns the required_size, which incidentally also matches the index that immediately follows the last index
        # of the buffer that was overwritten with the input data.
        return required_size

    @staticmethod
    @njit
    def __write_array_data(
            target_buffer: np.ndarray,
            array_object: np.ndarray,
            start_index: int,
    ) -> int:
        """Converts input numpy arrays to byte-sequences and writes them to the __transmission_buffer.

        Notes:
            Uses static integer codes to indicate runtime errors.

        Args:
            target_buffer: The buffer to which the data will be written. This should be the class-specific
                __transmission_buffer array.
            array_object: The numpy array to be written to the transmission buffer. Currently, the method is designed to
                only work with one-dimensional arrays with a minimal size of 1 element. The array should be using one
                of the supported numpy scalar datatypes (see write_data() documentation for more details).
            start_index: The index inside the __transmission_buffer (0 to 253) at which to start writing the data.

        Returns:
            The positive index inside the __transmission_buffer that immediately follows the last index of the buffer to
            which the data was written. Returns 0 if the buffer does not have enough space to accommodate the data
            written at the start_index. Returns -1 if the input array is not one-dimensional. Returns -2 if the input
            array is empty.
        """

        if array_object.size == 0:
            return -2  # Returns -2 if the input array is empty.

        elif array_object.ndim != 1:
            return -1  # Returns -1 if the input array is not one-dimensional.

        # Calculates the required space inside the buffer to store the data inserted at the start_index
        data_size = array_object.size * array_object.itemsize  # Size of each element * the number of elements
        required_size = start_index + data_size

        if required_size > target_buffer.size:
            return 0  # Returns 0 if the buffer does not have enough space to accommodate the data

        # Writes the array data to the buffer, starting at the start_index and ending just before required_size index
        target_buffer[start_index:required_size] = np.frombuffer(array_object, dtype=np.uint8)

        # Returns the required_size, which incidentally also matches the index that immediately follows the last index
        # of the buffer that was overwritten with the input data.
        return required_size

    def read_data(
            self,
            data_object: Union[np.unsignedinteger, np.signedinteger, np.floating, np.bool_, np.ndarray, Type],
            start_index: int = 0,
    ) -> tuple[Union[np.unsignedinteger, np.signedinteger, np.floating, np.bool_, np.ndarray, Type], int]:
        """Recreates the input data_object using the data read from the payloads stored inside __reception_buffer.

        This method acts as a wrapper for the private jit-compiled method called to actually read the data. This method
        uses the input object as a prototype, which supplies the number of bytes to read from the received payload and
        the datatype to cast the read bytes to. If the payload has sufficiently many bytes available from the
        start_index to accommodate filling the object, the object will be recreated using the data extracted from the
        payload.

        Notes:
            At this time, the method only works with valid numpy scalars and arrays as well as python dataclasses
            entirely made out of valid numpy types. Using numpy rather than standard python types increases runtime
            speed (when combined with other optimizations) and enforces strict typing (critical for microcontroller
            communication, as most controllers use strictly typed C++ / C languages).

            The method does not change the value of the __bytes_in_reception_buffer tracker, as reading from buffer does
            not in any way modify the data stored inside the buffer. The only way to change the value of the tracker is
            to call the reset_reception_buffer() method or receive_data() method.

            Note, the maximum runtime speed of this method is achieved when reading data as numpy arrays, which is
            optimized to a single read operation. The minimum runtime speed is achieved by reading dataclasses, as it
            involves slow python looping over dataclass attributes. Choose the read format based on your speed and
            convenience requirements.

            Also, internally, this method converts scalars to a one-element numpy array, as it is faster to use
            jit-compiled array-based method. This makes arrays the most time-efficient inputs, as it does not
            involve running scalar-array-scalar conversions.

        Args:
            data_object: A numpy scalar or array object or a python dataclass made entirely out of valid numpy objects.
                The input object is used as a prototype to determine how many bytes to read from the __reception_buffer
                and have to be properly initialized. Supported numpy types are: uint8, uint16, uint32, uint64, int8,
                int16, int32, int64, float32, float64 and bool_. Additionally, arrays have to be 1-dimensional and not
                empty to be supported.
            start_index: The index inside the __reception_buffer (0 to 253) from which to start reading the
                data_object bytes. Unlike for write_data() method, this value is mandatory.

        Returns:
            A tuple of 2 elements. The first element is the data_object read from the __reception_buffer, which is cast
            to the appropriate numpy type. When the data_object is a dataclass, the returned object will be the same
            dataclass instance with all attributes overwritten with read numpy values. The second element is the index
            that immediately follows the last index that was read from the __reception_buffer during method runtime.

        Raises:
            TypeError: If the input object is not a supported numpy scalar, numpy array, or python dataclass.
            ValueError: If the payload stored inside the __reception_buffer does not have the sufficient number of bytes
                available from the start_index to fill the requested object. Also, if the input object is a
                multidimensional or empty numpy array.
            RuntimeError: If the error-resolving mechanism based on the value of the end_index is not able to
                resolve the error code. This should not really occur, so this is more of a static guard to aid
                developers.
        """

        # Pre-initializes the end index tracker.
        end_index = -10  # Initializes to a specific negative value that is not a valid index or runtime error code

        # If the input object is a supported numpy scalar, converts it to a numpy array and calls the read method.
        # Converts the returned one-element array back to a scalar numpy type. Due to numba limitations (or, rather,
        # the unorthodox way it is used here), this is the most efficient available method.
        if isinstance(data_object, self.__accepted_numpy_scalars):
            out_object, end_index = self.__read_array_data(
                self.__reception_buffer,
                np.array(data_object, dtype=data_object.dtype),
                start_index,
                self.__bytes_in_reception_buffer,
            )
            out_object = np.dtype(data_object.dtype).type(out_object.item())

        # If the input object is a numpy array, first ensures that it's datatype matches one of the accepted scalar
        # numpy types and, if so, calls the array data reading method.
        elif isinstance(data_object, np.ndarray):
            if data_object.dtype in self.__accepted_numpy_scalars:
                out_object, end_index = self.__read_array_data(
                    self.__reception_buffer,
                    data_object,
                    start_index,
                    self.__bytes_in_reception_buffer,
                )

        # If the input object is a python dataclass, enters a recursive loop which calls this method for each class
        # attribute. This allows retrieving and overwriting each attribute with the bytes read from the buffer,
        # simulating how the microcontroller version of this library works for C / C++ structures.
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

        # If the input value is not a valid numpy scalar, array using a valid scalar datatype or a python dataclass,
        # raises TypeError exception.
        else:
            error_message = (
                f"Unsupported input data_object type ({type(data_object)}) encountered when reading data "
                f"from __reception_buffer. At this time, only the following numpy scalar or array types are supported: "
                f"{self.__accepted_numpy_scalars}. Alternatively, a dataclass with all attributes set to supported "
                f"numpy scalar or array types is also supported."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If end_index is different from the start_index and no error has been raised, the method runtime was
        # successful, so returns the read data_object and the end_index to caller
        if end_index > start_index:
            # Returns the object recreated using data from the buffer and the end_index to caller
            # noinspection PyUnboundLocalVariable
            return out_object, end_index

        # If the index is set to code 0, this indicates that the payload did not have sufficient data starting from the
        # start_index to recreate the object.
        elif end_index == 0:
            error_message = (
                f"Insufficient payload size to read the data from the __reception_buffer starting at the index "
                f"'{start_index}'. Specifically, given the object size of '{data_object.nbytes}' bytes, the required "
                f"payload size is '{start_index + data_object.nbytes}' bytes, but the available size is "
                f"'{self.bytes_in_reception_buffer}' bytes."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If the index is set to code -1, that indicates that a multidimensional numpy array was provided as input,
        # but only flat arrays are allowed.
        elif end_index == -1:
            error_message = (
                f"A multidimensional numpy array with {data_object.ndim} dimensions requested when reading "
                f"data from __reception_buffer. At this time, only one-dimensional (flat) arrays are supported."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If the index is set to code -2, that indicates that an empty numpy array was provided as input, which does
        # not make sense and therefore is likely an error.
        elif end_index == -2:
            error_message = (
                f"Am empty (size 0) numpy array requested when reading data from __reception_buffer. Reading empty "
                f"arrays is currently not supported."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # If the end_index is not resolved properly, catches and raises a runtime error. This is a static guard to
        # aid developers in discovering errors.
        else:
            error_message = (
                f"Unknown end_index-communicated error code ({end_index}) encountered when reading data "
                f"from __reception_buffer."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    @staticmethod
    @njit
    def __read_array_data(
            source_buffer: np.ndarray,
            array_object: Union[np.unsignedinteger, np.signedinteger, np.floating, np.bool_, np.ndarray],
            start_index: int,
            payload_size: int,
    ) -> tuple[np.ndarray, int]:
        """Reads the requested array_object from the __reception_buffer.

        Notes:
            Uses static integer codes to indicate runtime errors.

        Args:
            source_buffer: The buffer from which the data will be read. This should be the class-specific
                __reception_buffer array.
            array_object: The numpy array to be read from the __reception_buffer. Currently, the method is designed to
                only work with one-dimensional arrays with a minimal size of 1 element. The array should be initialized
                and should use one of the supported datatypes. It is used as a prototype reconstructed using the data
                stored inside the buffer.
            start_index: The index inside the __reception_buffer (0 to 253) at which to start reading the data.
            payload_size: The number of payload bytes stored inside the buffer. This is used to limit the read operation
                to avoid retrieving data from the uninitialized portion of the buffer.

        Returns:
            A two-element tuple. The first element is the numpy array that uses the datatype and size derived from the
            input array_object and value(s) reconstructed from the source_buffer-read data. The second element is the
            index that immediately follows the last index that was read during method runtime to support chained read
            calls. If method runtime fails, returns an empty numpy array as the first element and a static error-code.
            Uses 0 for error code if the payload does not have enough bytes from the start_index to the end of the
            payload to fill the array with data. Returns -1 if the input array is not one-dimensional. Returns -2 if the
            input array is empty.
        """

        # Calculates the end index for the read operation (this is based on how many bytes are required to represent the
        # object and the start_index for the read operation).
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
        # the number of bytes necessary to represent the object (also derived from the data_object via .nbytes).
        return np.frombuffer(source_buffer[start_index:required_size], dtype=array_object.dtype).copy(), required_size

    def send_data(self) -> bool:
        """Packages the payload stored in the __transmission_buffer and sends it over the serial port.

        This is the central wrapper method that calls various sub-methods as needed. It carries out two distinct steps:
        Builds a transmission packet using the payload (fast) and writes it to the serial port (slow).

        Notes:
            The constructed packet being sent over the serial port has the following format:
            [START BYTE]_[OVERHEAD BYTE]_[COBS ENCODED PAYLOAD]_[DELIMITER BYTE]_[CRC CHECKSUM]

            Packet construction takes between 3 and 5 us, writing the packet takes 24-40 us and cannot really be
            optimized any further beyond rewriting pySerial and, probably, Windows serial handlers. In the future, other
            transmission backends, such as MQTT, may be added as this byte-stream format is actually fairy universal.

        Returns:
            True to indicate that the data was sent.

        Raises:
            Exception: Uses bundled COBS and CRC classes to resolve any errors and raise appropriate exceptions if this
                method runs into an error during its runtime.
            RuntimeError: If an unexpected error that is not caught by the automated error-resolution mechanism is
                encountered. Such cases are very unlikely, but the static guard is kept around in case they do occur.
        """

        # Constructs the serial packet ot be sent. This is a fast inline aggregation of all packet construction steps
        # and to increase its runtime speed it uses JIT compilation and, therefore, has to access the inner jitclasses
        # instead of suing the python COBS and CRC class wrappers.
        packet = self.__construct_packet(
            self.__transmission_buffer,
            self.__cobs_processor.processor,
            self.__crc_processor.processor,
            self.__bytes_in_transmission_buffer,
            self.__delimiter_byte,
            self.__start_byte,
        )

        # A valid packet will always have a positive size. If the returned packet size is above 0, proceeds with sending
        # the packet over the serial port.
        if packet.size > 0:
            # Calls pySerial write method. This takes 80% of this method's runtime and cannot really be optimized any
            # further as its speed directly depends on how Windows API handles serial port access.
            self.__port.write(packet.tobytes())

            # Resets the transmission buffer to indicate that the payload was sent and prepare for sending the next
            # payload.
            self.reset_transmission_buffer()

            # Returns True to indicate that data was successfully sent.
            return True

        # If constructor method returns an empty packet, that means one of the inner methods ran into an error.
        # Currently, only COBS and CRC classes can run into errors during __construct_packet() runtime (in theory, right
        # now errors are more or less not possible at all). When that happens, the method re-runs the computations using
        # non-jit-compiled methods that will find and resolve the error. This is rather slow, but it is not meant to be
        # executed in the first place, so is considered acceptable to use as a static fallback.
        packet = self.__cobs_processor.encode_payload(
            payload=self.__transmission_buffer[: self.__bytes_in_transmission_buffer], delimiter=self.__delimiter_byte
        )
        checksum = self.__crc_processor.calculate_packet_crc_checksum(packet)
        self.__crc_processor.convert_crc_checksum_to_bytes(checksum)

        # The steps above SHOULD run into an error. If they did not, there is an unexpected error originating from the
        # __construct_packet method. In this case, raises a generic RuntimeError to notify the user of the error so that
        # they manually discover and rectify it.
        error_message = (
            "Unexpected error encountered for __construct_packet() method when sending payload data. Re-running all "
            "COBS and CRC steps used for packet construction in wrapped mode did not reproduce the error. Manual "
            "error resolution required."
        )
        raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    @staticmethod
    @njit
    def __construct_packet(
            payload_buffer: np.ndarray,
            cobs_processor: COBSProcessor.processor,
            crc_processor: CRCProcessor.processor,
            payload_size: int,
            delimiter_byte: np.uint8,
            start_byte: np.uint8,
    ) -> np.ndarray:
        """Constructs the serial packet using the payload stored inside the input buffer.

        This method inlines COBS, CRC and start_byte prepending steps that iteratively transform the payload stored
        inside the class __transmission_buffer into a serial packet that can be transmitted to the microcontroller.
        By accessing typically hidden jit-compiled _COBSProcessor and _CRCProcessor classes, this method inlines and
        compiles all operations into a single method, achieving the highest possible execution speed.

        Notes:
            At the time of writing, there is more or less no way this runtime can fail. The static guards evaluated at
            class initialization and the way other class methods are implemented basically ensure the method will
            always work as expected. That said, since this can potentially change in the future, the method does contain
            a full suite of error handling tools.

        Args:
            payload_buffer: The numpy array that stores the 'raw' payload bytes. This should be automatically set to the
                __transmission_buffer by the wrapper method.
            cobs_processor: The inner _COBSProcessor jitclass instance. The instance can be obtained by using
                '.processor' property of the COBSProcessor wrapper class.
            crc_processor: The inner _CRCProcessor jitclass instance. The instance can be obtained by using '.processor'
                property of the RCProcessor wrapper class.
            payload_size: The number of bytes that actually makes up the payload (it is expected that payload only uses
                a portion of the input payload_buffer).
            delimiter_byte: The byte-value used to mark the end of each transmitted packet's payload region.
            start_byte: The byte-value used to mark the beginning of each transmitted packet.

        Returns:
            A numpy byte array containing the constructed serial packet if the method runtime is successful. Otherwise,
            returns an empty numpy byte array (size 0) to indicate runtime failure. In this case, it is advised to use
            the verification methods available through the COBS and CRC processor classes to raise the proper exception
            based on the error encountered. Note, the verification methods are private and have to be accessed directly.

        """
        # Extracts the payload from the input buffer and encodes it using COBS scheme.
        packet = cobs_processor.encode_payload(payload_buffer[:payload_size], delimiter_byte)

        # If encoding fails, escalates the error by returning an empty array. For both encoding methods, the presence of
        # a failure is easily discerned by evaluating whether the packet is an empty array, as all encoding methods
        # return empty arrays as a sign of failure.
        if packet.size == 0:
            return np.empty(0, dtype=payload_buffer.dtype)

        # Calculates the CRC checksum for the encoded payload and converts it to a bytes' numpy array
        checksum = crc_processor.calculate_packet_crc_checksum(packet)

        # Checksum calculation method does not have a unique error-associated return value. If it runs into an error, it
        # returns 0, but 0 can also be returned by a successful checksum calculation. To verify that the checksum
        # calculation was successful, verifies that the processor status matches expected success status.
        if crc_processor.status != crc_processor.checksum_calculated:
            return np.empty(0, dtype=payload_buffer.dtype)

        # Converts the integer checksum to a bytes' format (form the crc postamble)
        postamble = crc_processor.convert_crc_checksum_to_bytes(checksum)

        # For bytes' conversion, an empty checksum array indicates failure
        if postamble.size == 0:
            return np.empty(0, dtype=payload_buffer.dtype)

        # Converts the start_byte to a preamble array. Then, concatenates the preamble, the encoded payload and the
        # checksum postamble to form the serial packet and returns the constructed packet to the caller.
        preamble = np.array([start_byte], dtype=np.uint8)
        combined_array = np.concatenate((preamble, packet, postamble))
        return combined_array

    def receive_data(self) -> bool:
        """If available, receives the serial packet stored inside the reception buffer of the serial port.

        This method aggregates the steps necessary to read the packet data from the serial port's reception buffer,
        verify its integrity using CRC and decode the payload out of the received data packet using COBS. Following
        verification, the decoded payload is transferred into the __reception_buffer array. This method uses multiple
        sub-methods and attempts to intelligently minimize the number of calls to the expensive serial port buffer
        manipulation methods.

        Notes:
            Expects the received data to be organized in the following format (different from the format used for
            sending the data to the microcontroller):
            [START BYTE]_[PAYLOAD SIZE BYTE]_[OVERHEAD BYTE]_[ENCODED PAYLOAD]_[DELIMITER BYTE]_[CRC CHECKSUM]

            The method can be co-opted as the check for whether the data is present in the first place, as it returns
            'False' if called when no data can be read or when the detected data is noise.

            Since calling data parsing methods is expensive, the method only attempts to parse the data if sufficient
            number of bytes is available, which is based on the minimum_received_payload_size class argument, among
            other things. The higher the value of this argument, the less time is wasted on trying to parse incomplete
            packets.

        Returns:
            A boolean 'True' if the data was parsed and is available for reading via read-data() calls. A boolean
            'False' if the number of available bytes is not sufficient to justify attempting to read the data.

        Raises:
            ValueError: If the received packet fails the CRC verification check, indicating that the packet is
                corrupted.
            RuntimeError: If __receive_packet method fails. Also, if an unexpected error occurs for any of the
                methods used to receive and parse the data.
            Exception: If __validate_packet() method fails, the validation steps are re-run using slower python-wrapped
                methods. Any errors encountered by these methods (From COBS and CRC classes) are raised as their
                preferred exception types.
        """
        # Clears the reception buffer in anticipation of receiving the new packet
        self.reset_reception_buffer()

        # Attempts to receive a new packet. If successful, this returns a static integer code 1 and saves the retrieved
        # packet to the __transmission_buffer and the size of the packet to the __bytes_in_transmission_buffer tracker.
        status_code = self.__receive_packet()

        # Only carries out the rest of the processing if the packet was successfully received
        if status_code == 1:
            # Validates and unpacks the payload into the reception buffer
            payload_size = self.__validate_packet(
                self.__reception_buffer,
                self.__bytes_in_reception_buffer,
                self.__cobs_processor.processor,
                self.__crc_processor.processor,
                self.__delimiter_byte,
                self.__postamble_size,
            )

            # Payload_size will always be a positive number if verification succeeds. In this case, overwrites the
            # __bytes_in_reception_buffer tracker with the payload size and returns 'true' to indicate runtime success
            if payload_size:
                self.__bytes_in_reception_buffer = payload_size
                return True

            # If payload size is 0, this indicates runtime failure. In this case, reruns the verification procedure
            # using python-wrapped methods as they will necessarily catch and raise the error that prevented validating
            # the packet. This is analogous to how it is resolved for __construct_packet() method failures.
            else:
                packet = self.__reception_buffer[: self.__bytes_in_reception_buffer]  # Extracts the packet

                # Resets the reception buffer to ensure intermediate data saved to the tracker is not usable for
                # data reading attempts
                self.reset_reception_buffer()

                # CRC-checks packet's integrity
                checksum = self.__crc_processor.calculate_packet_crc_checksum(buffer=packet)

                # If checksum verification (NOT calculation, that is caught by the calculator method internally) fails,
                # generates a manual error message that tells the user how the checksum failed.
                if checksum != 0:
                    # Extracts the crc checksum from the end of the packet buffer
                    byte_checksum = packet[-self.__postamble_size:]

                    # Also separates the packet portion of the buffer from the checksum
                    packet = packet[: packet.size - self.__postamble_size]

                    # Converts the CRC checksum extracted from the end of the packet from a byte array to an integer.
                    checksum_number = self.__crc_processor.convert_crc_checksum_to_integer(byte_checksum)

                    # Separately, calculates the checksum for the packet
                    expected_checksum = self.__crc_processor.calculate_packet_crc_checksum(buffer=packet)

                    # Uses the checksum values calculated above to issue an informative error message to the user.
                    error_message = (
                        f"CRC checksum verification failed when receiving data. Specifically, the checksum value "
                        f"transmitted with the packet {hex(checksum_number)} did not match the value expected for the "
                        f"packet (calculated locally) {hex(expected_checksum)}. This indicates packet was corrupted "
                        f"during transmission or reception."
                    )
                    raise ValueError(
                        textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)
                    )

                # Removes the CRC bytes from the end of the packet as they are no longer needed if the CRC check passed
                packet = packet[: packet.size - self.__postamble_size]

                # COBS-decodes the payload from the received packet.
                _ = self.__cobs_processor.decode_payload(packet=packet, delimiter=self.__delimiter_byte)

                # The steps above SHOULD run into an error. If they did not, there is an unexpected error originating
                # from the __validate_packet method. In this case, raises a generic RuntimeError to notify the user of
                # the error so that they manually discover and rectify it.
                error_message = (
                    "Unexpected error encountered for __verify_packet() method when receiving data. Re-running all "
                    "COBS and CRC steps used for packet validation in wrapped mode did not reproduce the error. Manual "
                    "error resolution required."
                )
                raise RuntimeError(
                    textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)
                )

        # Handles other possible status codes, all of which necessarily mean some failure has occurred during packet
        # reception runtime.
        # Not enough bytes were available to justify attempting to receive the packet or enough bytes
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
                f"payload_size was not received in time ({self.__timeout} microseconds) following the reception of the "
                f"start_byte."
            )

        # Payload-size byte was set to a value that exceeds the maximum allowed received payload size.
        elif status_code == 104:
            error_message = (
                f"Serial packet reception failed. The declared size of the payload "
                f"({self.__bytes_in_reception_buffer}), extracted from the received payload_size byte of the serial "
                f"packet, was above the maximum allowed size of {self.__reception_buffer.size}."
            )

        # Packet bytes were not received in time (packet staled)
        elif status_code == 105:
            # noinspection PyUnboundLocalVariable
            error_message = (
                f"Serial packet reception failed. Reception staled at packet bytes reception. Specifically, the "
                f"byte number {self.__bytes_in_reception_buffer + 1} was not received in time ({self.__timeout} "
                f"microseconds) following the reception of the previous byte."
            )

        # Unknown status_code. This should not really occur, and this is a static guard to help the developers.
        else:
            error_message = (
                f"Unknown status_code value {status_code} returned by the __receive_packet() method when "
                f"receiving data."
            )

        # Regardless of the error-message, uses RuntimeError for any valid error
        raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    def __receive_packet(self) -> int:
        """Attempts to read the serialized packet from the serial port's reception buffer.

        This is a fairly involved method that calls a jit-compiled parser method to actually parse the bytes read from
        the serial port buffer into the packet format to be processed further. The method is designed in a way to
        minimize the number of read() and in_waiting() method calls as they are very costly. The way this method is
        written should be optimized for the vast majority of cases though.

        Notes:
            This method uses the __timeout attribute to specify the maximum delay in microseconds(!) between receiving
            any two consecutive bytes of the packet. That is, if not all bytes of the packet are available to the method
            at runtime initialization, it will wait at most __timeout microseconds for the number of available bytes to
            increase before declaring the packet stale. There are two points at which the packet can become stale: the
            very beginning (the end of the preamble reception) and the reception of the packet itself. This corresponds
            to how the microcontroller sends teh data (preamble, followed by the packet+postamble fused into one). As
            such, the separating the two breakpoints with different error codes makes sense from the algorithmic
            perspective.

            The method tries to minimize the number of read() calls it makes as these calls are costly (compared to the
            rest of the methods in this library). As such, it may occasionally read more bytes than needed to process
            the incoming packet. In this case, any 'leftover' bytes are saved to the class __leftover_bytes attribute
            and reused by the next call to __parse_packet().

            This method assumes the sender uses the same CRC type as the SerializedTransferProtocol class, as it
            directly controls the CRC checksum byte-size. Similarly, it assumes teh sender uses the same delimiter and
            start_byte values as the class instance. If any of these assumptions are violated, this method will not
            parse the packet data correctly.

            Returned static codes: 101 -> no bytes to read. 102 -> start byte not found error. 103 -> reception staled
            at acquiring the payload_size / packet_size. 104 -> payload size too large (not valid). 105 -> reception
            staled at acquiring packet bytes. Also returns code 1 to indicate successful packet acquisition.

        Returns:
            A static integer code (see notes) that denotes method runtime status. Status code '1' indicates successful
            runtime, and any other code is an error to be handled by the wrapper method. If runtime is successful, the
            retrieved packet is saved to the __reception_buffer and the size of the retrieved packet is saved to the
            __bytes_in_reception_buffer tracker.
        """

        # Quick preface. This method is written with a particular focus on minimizing the number of calls to read() and
        # in_waiting() methods of the Serial class as they take a very long time to run compared to most of the
        # jit-compiled methods provided by this library. As such, if the packet can be parsed without calling these two
        # methods, that is always the priority. The trade-off is that if the packet cannot be parsed, we are losing
        # time running library methods essentially for nothing. Whether this 'gamble' works out or not heavily depends
        # on how the library is used, but it is assumed that in the vast majority of cases it WILL pay off.

        # If there are NOT enough leftover bytes to justify starting the reception procedure, checks how many bytes can
        # be obtained from the serial port
        if len(self.__leftover_bytes) < self.__minimum_packet_size:
            # Combines the bytes inside the serial port buffer with the leftover bytes from previous calls to this
            # method and repeats the evaluation
            available_bytes = self.__port.in_waiting
            total_bytes = len(self.__leftover_bytes) + available_bytes
            enough_bytes_available = total_bytes > self.__minimum_packet_size

            # If the enough bytes are available if buffer bytes are included, reads and appends them to the end of the
            # leftover bytes
            if enough_bytes_available:
                self.__leftover_bytes += self.__port.read(available_bytes)

        # Otherwise, if enough bytes are available without using the read operation, statically sets the flag to true
        # and begins parsing the packet
        else:
            enough_bytes_available = True

        # If not enough bytes are available, returns the static code 101 to indicate there was an insufficient number of
        # bytes to read from the buffer
        if not enough_bytes_available:
            return 101

        # The first call to the parses method, expect to at the very least find the start byte and at best to resolve
        # the entire packet
        status, packet_size, remaining_bytes, packet_bytes = self.__parse_packet(
            self.__leftover_bytes,
            self.__start_byte,
            self.__max_rx_payload_size,
            self.__postamble_size,
            self.__allow_start_byte_errors,
        )

        # Resolves parsing pass outcomes.
        # Packet parsed. Saves the packet to the __reception_buffer and the packet size to the
        # __bytes_in_reception_buffer tracker (for now)
        if status == 1:
            self.__reception_buffer[:packet_size] = packet_bytes
            self.__bytes_in_reception_buffer = packet_size

            # If any bytes remain unprocessed, adds them to storage until the next call to this method
            self.__leftover_bytes = remaining_bytes.tobytes()
            return status

        # Status above 2 means an already resolved error or a non-error terminal status. Either the start byte was not
        # found, or the payload_size was too large (invalid).
        elif status > 2:
            # This either completely resets the leftover_bytes tracker or sets them to the number of bytes left after
            # the terminator code situation was encountered. THis latter case is exclusive to code 104, as encountering
            # an invalid payload_size may have unprocessed bytes that remain at the time the error scenario is
            # encountered.
            self.__leftover_bytes = remaining_bytes.tobytes()
            # Only meaningful for code 104, shares the packet size to be used in error messages via the tracker value
            self.__bytes_in_reception_buffer = packet_size
            return status

        # Packet found, but not enough bytes are available to finish parsing the packet. Code 0 specifically means that
        # the parser stopped at payload_size (payload_size byte was not available). This is easily the most
        # computationally demanding case, as potentially 2 more read() calls will be needed to parse the packet.
        elif status == 0:
            # Waits for at least one more byte to become available or for the reception to timeout.
            self.__timer.reset()
            available_bytes = self.__port.in_waiting
            while self.__timer.elapsed < self.__timeout or available_bytes != 0:
                available_bytes = self.__port.in_waiting

            # If no more bytes are available (only one is needed) returns code 103: Packet reception staled at
            # payload_size byte.
            if available_bytes == 0:
                # There are no leftover bytes when code 103 is encountered, so clears the storage
                self.__leftover_bytes = bytes()
                return 103

            # If more bytes are available, reads the bytes into the placeholder storage. All leftover bytes are
            # necessarily consumed if status is 0, so the original value of the storage variable is irrelevant and
            # can be discarded at this point
            self.__leftover_bytes = self.__port.read()

            # This time sets a boolean flag to skip looking for start byte, as start byte is already found by the
            # first parser call.
            status, packet_size, remaining_bytes, packet_bytes = self.__parse_packet(
                self.__leftover_bytes,
                self.__start_byte,
                self.__max_rx_payload_size,
                self.__postamble_size,
                self.__allow_start_byte_errors,
                True,
            )

            # Status 1 indicates that the packet was fully parsed. Returns the packet to caller
            if status == 1:
                self.__reception_buffer[:packet_size] = packet_bytes
                self.__bytes_in_reception_buffer = packet_size
                self.__leftover_bytes = remaining_bytes.tobytes()
                return status

            # Status 2 indicates not all the packet was parsed, but the payload_size has been found and resolved.
            # Attempts to resolve the rest of the packet
            elif status == 2:
                # Calculates the missing number of bytes from the packet_size and the size of the packet_bytes array
                required_size = packet_size - packet_bytes.size  # Accounts for already received bytes

                # Blocks until enough bytes are available. Resets the timer every time more bytes become available
                self.__timer.reset()
                available_bytes = self.__port.in_waiting
                delta = required_size - available_bytes  # Used to determine when to reset the timer
                while self.__timer.elapsed < self.__timeout or delta > 0:
                    available_bytes = self.__port.in_waiting
                    delta_new = required_size - available_bytes

                    # Compares the deltas each cycle. If new delta is different from the old one, overwrites the delta
                    # and resets the timer
                    if delta_new != delta:
                        self.__timer.reset()
                        delta = delta_new

                # If the while loop is escaped due to timeout, issues code 105: Packet reception staled at receiving
                # packet bytes.
                if delta > 0:
                    # There are no leftover bytes when code 105 is encountered, so clears the storage
                    self.__leftover_bytes = bytes()
                    # Saves the number of the byte at which the reception staled so that it can be used in the error
                    # message raised by the wrapper
                    self.__bytes_in_reception_buffer = packet_size - delta
                    return 105

                # If the bytes were received in time, calls the parser a third time to finish packet reception. Inputs
                # the packet_size and packet_bytes returned by the last method call to automatically jump to parsing
                # the remaining packet bytes
                status, packet_size, remaining_bytes, packet_bytes = self.__parse_packet(
                    self.__leftover_bytes,
                    self.__start_byte,
                    self.__max_rx_payload_size,
                    self.__postamble_size,
                    self.__allow_start_byte_errors,
                    True,
                    packet_size,
                    packet_bytes,
                )

                # This is the ONLY possible outcome
                if status == 1:
                    self.__reception_buffer[0:packet_size] = packet_bytes
                    self.__bytes_in_reception_buffer = packet_size
                    self.__leftover_bytes = remaining_bytes.tobytes()
                    return status

            # If the status is not 1 or 2, returns the (already resolved) status 104. This is currently the only
            # possibility here, but uses status value in case it ever ends up being something else as well
            else:
                self.__leftover_bytes = remaining_bytes.tobytes()
                self.__bytes_in_reception_buffer = packet_size  # Saves the packet size to be used in the error message
                return status

        # Same as above, but code 2 means that the payload_size was found and used to determine the packet_size, but
        # there were not enough bytes to finish parsing the packet. Attempts to wait for enough bytes to become
        # available
        elif status == 2:
            # Calculates the missing number of bytes from the packet_size and the size of the packet_bytes array
            required_size = packet_size - packet_bytes.size  # Accounts for already received bytes

            # Blocks until enough bytes are available. Resets the timer every time more bytes become available
            self.__timer.reset()
            available_bytes = self.__port.in_waiting
            delta = required_size - available_bytes  # Used to determine when to reset the timer
            while self.__timer.elapsed < self.__timeout or delta > 0:
                available_bytes = self.__port.in_waiting
                delta_new = required_size - available_bytes

                # Compares the deltas each cycle. If new delta is different from the old one, overwrites the delta
                # and resets the timer
                if delta_new != delta:
                    self.__timer.reset()
                    delta = delta_new

            # If the while loop is escaped due to timeout, issues code 105: Packet reception staled at receiving
            # packet bytes.
            if delta > 0:
                # There are no leftover bytes when code 105 is encountered, so clears the storage
                self.__leftover_bytes = bytes()
                # Saves the number of the byte at which the reception staled so that it can be used in the error
                # message raised by the wrapper
                self.__bytes_in_reception_buffer = packet_size - delta
                return 105

            # If the bytes were received in time, calls the parser a third time to finish packet reception. Inputs
            # the packet_size and packet_bytes returned by the last method call to automatically jump to parsing
            # the remaining packet bytes
            status, packet_size, remaining_bytes, packet_bytes = self.__parse_packet(
                self.__leftover_bytes,
                self.__start_byte,
                self.__max_rx_payload_size,
                self.__postamble_size,
                self.__allow_start_byte_errors,
                True,
                packet_size,
                packet_bytes,
            )

            # The ONLY possible outcome.
            if status == 1:
                self.__reception_buffer[0:packet_size] = packet_bytes
                self.__bytes_in_reception_buffer = packet_size
                self.__leftover_bytes = remaining_bytes.tobytes()
                return status

        # There should not be any way to reach this guard, but it is kept here to help developers by detecting when the
        # logic of this method fails to prevent it reaching this point
        error_message = (
            f"General failure of the __receive_packet() method runtime detected. Specifically, the method reached the "
            f"static guard, which should not be possible. The last available parser status is ({status}). Manual "
            f"intervention is required to identify and resolve the error."
        )
        raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    @staticmethod
    @njit
    def __parse_packet(
            read_bytes: bytes,
            start_byte: np.uint8,
            max_payload_size: int,
            postamble_size: int,
            allow_start_byte_errors: bool,
            start_found: bool = False,
            packet_size: int = 0,
            packet_bytes: np.ndarray = np.empty(0, dtype=np.uint8),
    ) -> tuple[int, int, np.ndarray, np.ndarray]:
        """Parses as much of the packet as possible using the input bytes stream.

        This method contains all packets parsing logic and takes in the bytes extracted from the serial port buffer and,
        as a best case scenario, returns the extracted packet ready for CRC verification and COBS decoding. This method
        is designed to be called repeatedly until a packet is fully parsed or until an external timeout guard handled
        by the __receive_packet() method kicks in to abort the reception. As such, it can recursively work on the same
        packet across multiple calls. To enable proper call, hierarchy it is essential that this method is called
        strictly from the __receive_packet() method.

        Notes:
            This method becomes increasingly helpful in use patterns where many bytes are allowed to aggregate in the
            serial port before being evaluated. Due to JIT compilation this method is very fast, and any execution time
            loss typically comes from reading the data from the underlying serial port. That step is optimized to read
            as much data as possible with each read() call, so the more data aggregates before being read the more
            efficient is each call to the major receive_data() method.

            The returns of this method are designed to support potentially iterative (not recursive) calls to this
            method. As a minium, the packet may be fully parsed (or definitively fail to be parsed, that is a valid
            outcome too) with one call, and as a maximum, 3 calls may be needed.

            The method uses static integer codes to communicate its runtimes:

            0 - Not enough bytes read to fully parse the packet. The start byte was found, but payload_size byte was not
            and needs to be read.
            1 - Packet fully parsed.
            2 - Not enough bytes read to fully parse the packet. The payload_size was resolved, but there were not
            enough bytes to fully parse the packet and more bytes need to be read.
            101 - No start byte found, interpreted as 'no bytes to read' as the class is configured to ignore start
            byte errors.
            102 - No start byte found, interpreted as a 'no start byte detected' error case.
            104 - Payload_size value is too big (above maximum allowed payload size) error.

            The wrapper method is expected to issue codes 103 and 105 if packet reception stales at payload_size or
            packet_bytes. All error codes are converted to errors at the highest level, which is the receive_data()
            method call (just how it is does for the rest of the methods).

        Args:
            read_bytes: A bytes() object that stores the bytes read from the serial port. If this is the first call to
                this method for a given _receive_packet() method runtime, this object would also include any bytes left
                from the previous _receive_packet() runtime.
            start_byte: The byte-value sued to mark the beginning of a transmitted packet in the byte-stream. This is
                used to detect the portion of the stream that encodes the packet.
            max_payload_size: The maximum size of the payload that can be received. This is set automatically by the
                parent class and, for PCs, this is capped at 254 bytes.
            postamble_size: The number of bytes needed to store the CRC checksum. This is determined based on the type
                of the CRC polynomial and provided by the parent class.
            allow_start_byte_errors: A boolean flag that determines whether inability to find start_byte should be
                interpreted as a natural case of having no bytes to read (default, code 101) or as an error (code 102)
                to be raised to the user. This is also derived from the parent class flag.
            start_found: Iterative parameter. When this method is called two or more times, this value can be provided
                to the method to skip resolving the start byte (detecting packet presence). Specifically, it is used
                when a call to this method finds the packet, but cannot resolve the packet size. Then, during a second
                call, start_byte searching step is skipped.
            packet_size: Iterative parameter. When this method is called two or more times, this value can be provided
                to the method to skip resolving the packet size. Specifically, it is used when a call to this method
                resolves the packet size, but cannot resolve the packet. Then, a second call to this method is made to
                resolve the packet and the size is provided as an argument to skip no longer needed parsing steps.
            packet_bytes: Iterative parameter. If the method is able to parse some, but not all the packet's bytes,
                the already parsed bytes can be fed back into the method during a second call using this argument.
                Then, the method will automatically combine already parsed bytes with any additionally extracted bytes.

        Returns:
            A tuple of four elements. The first element is a static integer code that described the outcome of method
            runtime. The second element is the parsed packet_size of the packet or 0 to indicate packet_size was not
            parsed or provided as a method argument. The third element is a numpy uint8 array that stores any bytes that
            remain after the packet has been fully parsed or parsing ran into code 104. The fourth element is the uint8
            array that stores the portion of the packet that has been parsed so far, which may be the entire packet or
            only part of the packet.
        """

        # Converts the input 'bytes' object to a numpy array to simplify processing operations below
        evaluated_bytes = np.frombuffer(read_bytes, dtype=np.uint8)

        # Loops bytes to be evaluated until all are processed or an exit condition is encountered
        for i in range(evaluated_bytes.size):
            # Counts processed bytes, always based on 'i'. Allows simplifying index increments carried out in response
            # to parsing various parts of the packet
            processed_bytes = i

            # Starts by looking for the start byte. Advances into the next stage either if a start_byte value is
            # encountered OR when resuming the processing from a previous call (indicated by the boolean start_found
            # flag).
            if evaluated_bytes[processed_bytes] == start_byte or start_found:
                # If the packet size is not known (is zero), enters packet_size resolution mode
                if packet_size == 0:
                    # Increments bytes counter to account for 'processing' start byte and advancing to the next byte
                    processed_bytes += 1

                    # If there are no more bytes to read after encountering the start_byte, this means that the
                    # payload_size byte was not received.
                    if evaluated_bytes.size - processed_bytes == 0:
                        # If the payload_size byte is not found, returns code 0 to indicate not enough bytes were
                        # available to parse the packet (same code is used at a later stage once the payload_byte is
                        # resolved). This basically tells the wrapper to block with timeout guard until more data is
                        # available. Here, returns packet_size set to 0 to indicate it was not found.
                        status_code = 0
                        packet_size = 0
                        remaining_bytes = np.empty(0, dtype=np.uint8)
                        packet_bytes = np.empty(0, dtype=np.uint8)

                        # Packages and returns output values as a tuple
                        return status_code, packet_size, remaining_bytes, packet_bytes

                    # If there are more bytes available after resolving the start_byte, reads the following byte as the
                    # payload size and checks it for validity
                    payload_size = evaluated_bytes[processed_bytes]
                    processed_bytes += 1  # Increments bytes counter to account for processing the payload_size byte

                    # Verifies that the payload size is within the allowed payload size limits. The PC statically sets
                    # this limit to the maximum valid value of 254 bytes, but a variable is used to check it nonetheless
                    # for better maintainability.
                    if payload_size > max_payload_size:
                        # If payload size is out of bounds, returns with status code 104: Payload size too big. This
                        # is a static integer code (like the rest of the code here!)
                        status_code = 104
                        packet_size = int(payload_size)  # Returns invalid payload size to be used in the error message

                        # In case there are more bytes after discovering the invalid packet size, they are returned.
                        # it may be tempting to discard them, but they may contain parts of the NEXT packet, so a longer
                        # route of re-feeding them into the processing sequence is chosen here. This does not matter
                        # unless error 104 is suppressed at some level (not default behavior) and the algorithm is
                        # allowed to just keep going... This may escalate to a cascade of parsing errors, but this is
                        # the problem for the developer that disables error 104 to solve. The author suggests using
                        # time-based data discarding approach to avoid error cascades in this case.
                        remaining_bytes = evaluated_bytes[processed_bytes:]  # Discards processed bytes
                        packet_bytes = np.empty(0, dtype=np.uint8)

                        # Packages and returns output values as a tuple
                        return status_code, packet_size, remaining_bytes, packet_bytes

                    # If payload size passed verification, calculates the packet size. This uses the payload_size as the
                    # backbone and increments it with the dynamic postamble size (depends on polynomial datatype) and
                    # static +2 to account for the overhead and delimiter bytes introduced by COBS-encoding the packet.
                    packet_size = payload_size + 2 + postamble_size

                # This step is reached either by parsing the start_byte and/or packet_size or by having the packet_size
                # already provided at call time (when resuming parsing packets that could not be completed during
                # previous method call). Either way, checks if enough bytes are available in the evaluated_bytes array
                # combined with the packet_bytes input array to fully parse the packet.
                if evaluated_bytes.size - processed_bytes >= packet_size - packet_bytes.size:
                    # Extracts the remaining number of bytes needed to fully parse the packet from the processed bytes
                    # array. Also, automatically 'discards' any processed bytes
                    extracted_bytes = evaluated_bytes[
                                      processed_bytes: packet_size - packet_bytes.size + processed_bytes
                                      ]

                    # Appends extracted bytes to the end of the array holding already parsed bytes
                    packet = np.concatenate((packet_bytes, extracted_bytes))

                    # Extracts any remaining bytes so that they can be properly stored for future receive_data() calls
                    remaining_bytes = evaluated_bytes[packet_size - packet_bytes.size + processed_bytes:]

                    # Sets the status to static code 1: Packet fully parsed code
                    status_code = 1

                    # Packages and returns the output data
                    return status_code, packet_size, remaining_bytes, packet

                # When not all bytes of the packet are available, moves all leftover bytes to the packet array and
                # uses static code 2 to indicate the paket was not available for parsing in-full, but that the
                # packet_size has been resolved. The wrapper method will then block in-place until enough bytes are
                # available to guarantee success of this method runtime on the next call
                else:
                    status_code = 2
                    # Zero, as all leftover bytes are absorbed into packet_bytes
                    remaining_bytes = np.empty(0, dtype=np.uint8)
                    # Discards any processed bytes and combines all evaluated bytes with the packet bytes. Generally,
                    # it should be impossible to reach this outcome and have any packet_bytes, but better to handle it
                    # safely.
                    packet_bytes = np.concatenate((packet_bytes, evaluated_bytes[processed_bytes:]))

                    # Note, packet_size is the same value it was provided to this method or calculated by this method
                    return status_code, packet_size, remaining_bytes, packet_bytes

        # If this point is reached, that means that the method was not able to resolve the start byte. Determines the
        # status code based on whether start byte errors are allowed or not. If they are allowed, returns 102,
        # otherwise (default) returns 101.
        if allow_start_byte_errors:
            status_code = 102
        else:
            status_code = 101

        # Also resolves the rest of the output
        packet_size = 0
        remaining_bytes = np.empty(0, dtype=np.uint8)
        packet_bytes = np.empty(0, dtype=np.uint8)

        # Packages and returns output values as a tuple
        return status_code, packet_size, remaining_bytes, packet_bytes

    @staticmethod
    @njit
    def __validate_packet(
            reception_buffer: np.ndarray,
            packet_size: int,
            cobs_processor: COBSProcessor.processor,
            crc_processor: CRCProcessor.processor,
            delimiter_byte: np.uint8,
            postamble_size: int,
    ) -> int:
        """Validates the packet using CRC checksum, decodes it using COBS-scheme and saves it to the reception_buffer.

        Both the CRC checksum and COBS decoding act as validation steps, and they jointly make it very unlikely that
        a corrupted packet passes this step. COBS-decoding extracts the payload from the buffer, making it available
        for consumption via read_data() method calls.

        Notes:
            This method expects the packet to be stored inside the __reception_buffer and will store the decoded
            payload to the __reception_buffer if method runtime succeeds. This allows optimizing memory usage and
            reduces the overhang of passing arrays around.

            The method uses the property of CRCs that ensures running a CRC calculation on the buffer to which its CRC
            checksum is appended will always return 0. For multibyte CRCs, this may be compromised if the byte-order of
            loading the CRC bytes into the postamble is not the order expected by the receiver system. This was never an
            issue during library testing, but good to be aware that is possible (usually some of the more nuanced
            UNIX-derived systems are known to do things differently in this regard).

        Args:
            reception_buffer: The buffer to which the extracted payload data will be saved and which is expected to
                store the packet to verify. Should be set to the __reception_buffer of the class.
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
        checksum = crc_processor.calculate_packet_crc_checksum(buffer=packet)

        # Verifies that the checksum calculation method ran successfully. if not, returns 0 to indicate verification
        # failure
        if crc_processor.status != crc_processor.checksum_calculated:
            return 0

        # If the checksum is not 0, but the calculator runtime was successful, this indicates that the packet was
        # corrupted, so returns code 0
        if checksum != 0:
            return 0
        else:
            # Removes the CRC bytes from the end of the packet as they are no longer needed if the CRC check passed
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
