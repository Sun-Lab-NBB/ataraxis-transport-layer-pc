"""This module contains the low-level helper classes that support the runtime of TransportLayer class methods."""

from typing import Any

from numba import uint8, uint16, uint32  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray
from numba.experimental import jitclass  # type: ignore[import-untyped]
from ataraxis_base_utilities import console

# Defines constants that are frequently reused in this module
_ZERO = np.uint8(0)
_ONE_BYTE = 1
_TWO_BYTE = 2
_BYTE_SIZE = 8

# Defines the collection of NumPy types used by the CRCProcessor class to represent valid input arguments and output
# values.
type CRCType = np.uint8 | np.uint16 | np.uint32


class _COBSProcessor:  # pragma: no cover
    """Provides methods for encoding and decoding data using the Consistent Overhead Byte Stuffing (COBS) scheme.

    Notes:
        This class is intended to be initialized through Numba's 'jitclass' function.

        See the original paper for the details on COBS methodology and specific data packet layouts:
        S. Cheshire and M. Baker, "Consistent overhead byte stuffing," in IEEE/ACM Transactions on Networking, vol. 7,
        no. 2, pp. 159-172, April 1999, doi: 10.1109/90.769765.

    Attributes:
        maximum_payload_size: The maximum size of the payload, in bytes. Due to COBS, cannot exceed 254 bytes.
        minimum_payload_size: The minimum size of the payload, in bytes.
        maximum_packet_size: The maximum size of the packet, in bytes. Due to COBS, it cannot exceed 256 bytes
            (254 payload bytes + 1 overhead + 1 delimiter byte).
        minimum_packet_size: The minimum size of the packet, in bytes. Due to COBS cannot be below 3 bytes.
    """

    def __init__(self) -> None:
        # Constant class parameters
        self.maximum_payload_size: int = 254
        self.minimum_payload_size: int = 1
        self.maximum_packet_size: int = 256
        self.minimum_packet_size: int = 3
        self.delimiter: int = 0

    def encode_payload(self, payload: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Encodes the input payload into a transmittable packet using the COBS scheme.

        Args:
            payload: The payload to be encoded using COBS scheme.

        Returns:
            The packet encoded using the COBS scheme.
        """
        # Saves payload size to a separate variable
        size = payload.size

        # Initializes the output array, uses payload size + 2 as size to make space for the overhead and
        # delimiter bytes (see COBS scheme for more details on why this is necessary).
        packet = np.empty(size + 2, dtype=payload.dtype)
        packet[-1] = self.delimiter  # Sets the last byte of the packet to the delimiter byte value
        packet[1:-1] = payload  # Copies input payload into the packet array, leaving spaces for overhead and delimiter.

        # A tracker variable that is used to calculate the distance to the next delimiter value when an
        # unencoded delimiter is required.
        next_delimiter_position = packet.size - 1  # Initializes to the index of the delimiter value added above

        # Iterates over the payload in reverse and replaces every instance of the delimiter value inside the
        # payload with the distance to the next delimiter value (or the value added to the end of the payload).
        # This process ensures that the delimiter value is only found at the end of the packet and, if the delimiter
        # is not 0, potentially also as the overhead byte value. This encodes the payload using the COBS scheme.
        for i in range(size - 1, -1, -1):  # Loops over every index of the payload
            if payload[i] == self.delimiter:
                # If any of the payload values match the delimiter value, replaces that value in the packet with
                # the distance to the next_delimiter_position. This is either the distance to the next encoded
                # value or the distance to the delimiter value located at the end of the packet.
                packet[i + 1] = next_delimiter_position - (i + 1)  # +1 is to translate from payload to packet index

                # Overwrites the next_delimiter_position with the index of the encoded value
                next_delimiter_position = i + 1  # +1 is to translate for payload to packet index

        # Once the runtime above is complete, sets the overhead byte to the value of the
        # next_delimiter_position. As a worst-case scenario, that would be the index of the delimiter byte
        # written to the end of the packet, which at maximum can be 255. Otherwise, that would be the distance
        # to the first encoded delimiter value inside the payload. It is now possible to start with the overhead
        # byte and 'jump' through all encoded values all the way to the end of the packet, where the only
        # unencoded delimiter is found.
        packet[0] = next_delimiter_position

        # Returns the encoded packet array to caller
        return packet

    def decode_payload(self, packet: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decodes the COBS-encoded payload from the input packet.

        Args:
            packet: The COBS-encoded packet from which to decode the payload.

        Returns:
            The payload decoded from the packet or an empty uninitialized numpy array if the method fails to decode the
            payload.
        """
        # noinspection DuplicatedCode
        size = packet.size  # Extracts packet size for the checks below

        # This is necessary due to how this method is used by the main class, where the input to this method
        # happens to be a 'readonly' array. Copying the array removes the readonly flag.
        packet = packet.copy()

        # Tracks the currently evaluated variable's index in the packet array. Initializes to 0 (overhead byte
        # index).
        read_index = 0

        # Tracks the distance to the next index to evaluate, relative to the read_index value
        next_index = packet[read_index]  # Reads the distance stored in the overhead byte into the next_index

        # Loops over the payload and iteratively jumps over all encoded values, restoring (decoding) them back
        # to the delimiter value in the process. Carries on with the process until it reaches the end of the
        # packet or until it encounters an unencoded delimiter value. These two conditions should coincide for
        # each well-formed packet.
        while (read_index + next_index) < size:
            # Increments the read_index via aggregation for each iteration of the loop
            read_index += next_index

            # If the value inside the packet array pointed by read_index is an unencoded delimiter, evaluates
            # whether the delimiter is encountered at the end of the packet
            if packet[read_index] == self.delimiter:
                if read_index == size - 1:
                    # If the delimiter is found at the end of the packet, extracts and returns the decoded
                    # packet to the caller.
                    return packet[1:-1]

                # If the delimiter is encountered before reaching the end of the packet, this indicates that
                # the packet was corrupted during transmission and the CRC-check failed to recognize the
                # data corruption. In this case, returns an empty array to indicate the error.
                return np.empty(0, dtype=packet.dtype)

            # If the read_index pointed value is not an unencoded delimiter, first extracts the value and saves
            # it to the next_index, as the value is the distance to the next encoded value or the unencoded
            # delimiter.
            next_index = packet[read_index]

            # Decodes the extracted value by overwriting it with the delimiter value
            packet[read_index] = self.delimiter

        # If this point is reached, that means that the method did not encounter an unencoded delimiter before
        # reaching the end of the packet. While the reasons for this are numerous, overall that means that the
        # packet is malformed and the data is corrupted, returns an empty array to indicate the error.
        return np.empty(0, dtype=packet.dtype)


class COBSProcessor:
    """Exposes the API for encoding and decoding data using the Consistent Overhead Byte Stuffing (COBS) scheme.

    This class wraps a JIT-compiled COBS processor implementation, combining the convenience of a pure-python API with
    the speed of the C-compiled processing code.

    Notes:
        This class is intended to be used by the TransportLayer class and should not be used directly by the
        end-users. It makes specific assumptions about the layout and contents of the processed data buffers that are
        not verified during runtime and must be enforced through the use of the TransportLayer class.

    Attributes:
        _processor: Stores the jit-compiled _COBSProcessor instance, which carries out all computations.
    """

    def __init__(self) -> None:
        # The template for the numba compiler to assign specific datatypes to variables used by the COBSProcessor class.
        # This is necessary for Numba to properly compile the class to C. Has to be defined before the class is
        # instantiated with the jitclass function.
        cobs_spec = [
            ("status", uint8),
            ("maximum_payload_size", uint8),
            ("minimum_payload_size", uint8),
            ("maximum_packet_size", uint16),
            ("minimum_packet_size", uint8),
            ("delimiter", uint8),
        ]

        # Instantiates the jit class and saves it to the wrapper class attribute. Developer hint: when used as a
        # function, jitclass returns an uninitialized compiled object, so initializing is crucial here.
        self._processor: _COBSProcessor = jitclass(cls_or_spec=_COBSProcessor, spec=cobs_spec)()

    def __repr__(self) -> str:
        """Returns a string representation of the COBSProcessor class instance."""
        return (
            f"COBSProcessor(maximum_payload_size={self._processor.maximum_payload_size}, "
            f"minimum_payload_size={self._processor.minimum_payload_size}, "
            f"maximum_packet_size={self._processor.maximum_packet_size}, "
            f"minimum_packet_size={self._processor.minimum_packet_size}, "
            f"delimiter={self._processor.delimiter})"
        )

    def encode_payload(self, payload: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Encodes the input payload into a transmittable packet using COBS scheme.

        The encoding produces the following packet structure: [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            payload: The payload to be encoded using the COBS scheme.

        Returns:
            The serialized packet encoded using the COBS scheme.
        """
        # Encodes the payload. The current class version assumes that the encoding cannot fail due to the safety checks
        # enforced by the TransportLayer class.
        return self._processor.encode_payload(payload)

    def decode_payload(self, packet: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Decodes the COBS-encoded payload from the input packet.

        Expects the input packets to adhere to the following structure:
        [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            packet: The COBS-encoded packet from which to decode the payload.

        Returns:
            The payload decoded from the packet.

        Raises:
            ValueError: If the decoding fails, indicating uncaught packet corruption.
        """
        # Calls decoding method
        payload = self._processor.decode_payload(packet)

        if payload.size == 0:
            message = (
                "Failed to decode the payload using the COBS scheme as the decoder did not find an unencoded delimiter"
                "at the expected location during the decoding process. Packet is likely corrupted."
            )
            console.error(message, error=ValueError)

        # Returns the decoded payload to caller if verification was successful
        return payload

    @property
    def processor(self) -> _COBSProcessor:
        """Returns the jit-compiled COBS processor class instance.

        This accessor allows external methods to directly interface with the JIT-compiled class, bypassing the Python
        wrapper.
        """
        return self._processor


class _CRCProcessor:  # pragma: no cover
    """Provides methods for working with Cyclic Redundancy Check (CRC) checksums used to verify the integrity of
    transferred data packets.

    Notes:
        This class is intended to be initialized through Numba's 'jitclass' function.

        For more information on how the CRC checksum works, see the original paper:
        W. W. Peterson and D. T. Brown, "Cyclic Codes for Error Detection," in Proceedings of the IRE, vol. 49, no. 1,
        pp. 228-235, Jan. 1961, doi: 10.1109/JRPROC.1961.287814.

        To increase runtime speed, this class generates a static CRC lookup table using the input polynomial, which is
        subsequently used to calculate CRC checksums.

    Attributes:
        polynomial: Stores the polynomial used for the CRC checksum calculation.
        initial_crc_value: Stores the initial value used for the CRC checksum calculation.
        final_xor_value: Stores the final XOR value used for the CRC checksum calculation.
        crc_byte_length: Stores the length of the CRC polynomial in bytes.
        crc_table: The array that stores the CRC lookup table.

    Args:
        polynomial: The polynomial used to generate the CRC lookup table.
        initial_crc_value: The initial value to which the CRC checksum variable is initialized during calculation.
        final_xor_value: The final XOR value to be applied to the calculated CRC checksum value.
    """

    def __init__(
        self,
        polynomial: CRCType,
        initial_crc_value: CRCType,
        final_xor_value: CRCType,
    ) -> None:
        # Resolves the crc_type and polynomial size based on the input polynomial. Makes use of the recently added
        # dtype comparison support
        crc_type: type[np.unsignedinteger[Any]]
        # noinspection PyTypeChecker
        if isinstance(polynomial, uint8):
            crc_type = np.uint8
            polynomial_size = np.uint8(1)
        elif isinstance(polynomial, uint16):
            crc_type = np.uint16
            polynomial_size = np.uint8(2)
        else:
            crc_type = np.uint32
            polynomial_size = np.uint8(4)

        # Local variables
        self.polynomial: CRCType = polynomial
        self.initial_crc_value: CRCType = initial_crc_value
        self.final_xor_value: CRCType = final_xor_value
        self.crc_byte_length: np.uint8 = polynomial_size
        self.crc_table = np.empty(256, dtype=crc_type)  # Initializes to empty for efficiency

        # Generates the lookup table based on the target polynomial parameters and iteratively sets each variable
        # inside the crc_table placeholder to the calculated values.
        self._generate_crc_table(polynomial=polynomial)

    # noinspection PyTypeHints
    def calculate_checksum(self, buffer: NDArray[np.uint8], check: bool = False) -> np.uint16:
        """Calculates the checksum for the data stored in the input buffer.

        Depending on configuration, this method can be used to either generate and write the CRC checksum to the end
        of the packet or to verify the integrity of the incoming packet using its checksum postamble.

        Args:
            buffer: The buffer that contains the COBS-encoded packet for which to resolve the checksum. The buffer must
                include the space for the CRC checksum at the end of the packet.
            check: Determines whether the method is called to verify the incoming packet's data integrity or to
                generate and write the CRC checksum to the outgoing packet's postamble section.

        Returns:
            The size of the buffer occupied by the packet's data and the appended CRC checksum if the method is called
            to calculate the new CRC checksum. The value '1' if the method is configured to verify the packet's data
            integrity and the data is intact and '0' otherwise.
        """
        # Intelligently determines the packet size based on buffer size and CRC checksum length. Also determines
        # noinspection PyTypeChecker
        packet_size = len(buffer) - self.crc_byte_length

        # Initializes the checksum
        crc_checksum = self.initial_crc_value

        # Calculates the checksum for the packet
        for i in range(packet_size):
            table_index = (crc_checksum >> (8 * (self.crc_byte_length - 1))) ^ buffer[i]
            crc_checksum = self._make_polynomial_type((crc_checksum << 8) ^ self.crc_table[table_index])

        # If the method is called to verify the incoming packet's integrity, includes the CRC checksum postamble in
        # the calculation.
        if check:
            # noinspection PyTypeChecker
            for i in range(packet_size, packet_size + self.crc_byte_length):
                table_index = (crc_checksum >> (8 * (self.crc_byte_length - 1))) ^ buffer[i]
                crc_checksum = self._make_polynomial_type((crc_checksum << 8) ^ self.crc_table[table_index])

        # Applies the final XOR
        crc_checksum ^= self.final_xor_value

        # If the method is called to generate and write a new checksum, adds the calculated checksum to the end of the
        # buffer.
        if not check:
            for i in range(self.crc_byte_length):
                buffer[packet_size + i] = (crc_checksum >> (8 * (self.crc_byte_length - i - 1))) & 0xFF

            # Returns the total size of the buffer with the post-pended checksum to indicate that the method ran as
            # expected.
            return np.uint16(len(buffer))

        # If the method is called to verify the data integrity, returns 1 if it succeeds and 0 otherwise.
        # Running the CRC calculation on the data with post-pended checksum should always return 0 for valid
        # data packets
        if crc_checksum == 0:
            return np.uint16(1)
        # Otherwise, the data is corrupted.
        return np.uint16(0)

    def _generate_crc_table(self, polynomial: CRCType) -> None:
        """Uses the input polynomial to compute the CRC checksums for each possible uint8 (byte) value.

        The method updates the precompiled empty crc_table with polynomial-derived CRC values. This method is only
        intended to be called by the class initialization method. Do not use this method outside the class
        initialization context.

        Notes:
            Due to the intricacies of JIT compilation and type-inferencing, the polynomial must be provided as an
            argument, rather than as an instance attribute.

        Args:
            polynomial: The polynomial to use for the generation of the CRC lookup table.
        """
        # Determines the number of bits in the CRC datatype
        crc_bits = np.uint8(self.crc_byte_length * 8)

        # Determines the Most Significant Bit (MSB) mask based on the CRC type
        msb_mask = self._make_polynomial_type(np.left_shift(1, crc_bits - 1))

        # Iterates over each possible value of a byte variable
        for byte in np.arange(256, dtype=np.uint8):
            # Casts crc to the appropriate type based on the polynomial type
            crc = self._make_polynomial_type(byte)

            # Shifts the CRC value left by the appropriate number of bits based on the CRC type to align the
            # initial value to the highest byte of the CRC variable.
            if crc_bits > _BYTE_SIZE:
                crc <<= crc_bits - _BYTE_SIZE

            # Loops over each of the 8 bits making up the byte-value being processed
            for _ in range(_BYTE_SIZE):
                # Checks if the top bit (MSB) is set
                if crc & msb_mask:
                    # If the top bit is set, shifts the crc value left to bring the next bit into the top
                    # position, then XORs it with the polynomial. This simulates polynomial division where bits
                    # are checked from top to bottom.
                    crc = self._make_polynomial_type((crc << 1) ^ polynomial)
                else:
                    # If the top bit is not set, simply shifts the crc value left. This moves to the next bit
                    # without changing the current crc value, as division by polynomial wouldn't modify it.
                    crc <<= np.uint8(1)

            # Adds the calculated CRC value for the byte to the storage table using byte-value as the key
            # (index). This value is the remainder of the polynomial division of the byte (treated as a
            # CRC-sized number), by the CRC polynomial.
            self.crc_table[byte] = crc

    def _make_polynomial_type(self, value: Any) -> CRCType:
        """Converts the input value to the appropriate numpy unsigned integer type based on the class instance
        polynomial datatype.

        This is a minor helper method designed to be used exclusively by other class methods. It allows
        resolving typing issues originating from the fact that, at the time of writing, numba is unable to use
        '.itemsize' and other properties of scalar numpy types.

        Notes:
            The datatype of the polynomial is inferred based on the byte-length of the polynomial as either
            uint8, uint16, or uint32 (uses 'crc_byte_length' attribute of the class).

        Args:
            value: The value to convert to the polynomial type.

        Returns:
            The value converted to the requested numpy unsigned integer datatype
        """
        # CRC-8
        if self.crc_byte_length == _ONE_BYTE:
            return np.uint8(value)

        # CRC-16
        if self.crc_byte_length == _TWO_BYTE:
            return np.uint16(value)

        # CRC-32. Since there are no plans to support CRC-64, this is the only remaining option
        return np.uint32(value)


class CRCProcessor:
    """Exposes the API for working with Cyclic Redundancy Check (CRC) checksums used to verify the integrity
    of transferred data packets.

    This class wraps a JIT-compiled CRC processor implementation, combining the convenience of a pure-python API with
    the speed of the C-compiled processing code.

    Notes:
        This class is intended to be used by the TransportLayer class and should not be used directly by the
        end-users. It makes specific assumptions about the layout and contents of the processed data buffers that are
        not verified during runtime and must be enforced through the use of the TransportLayer class.

    Attributes:
        _processor: Stores the jit-compiled _CRCProcessor instance, which carries out all computations.

    Args:
        polynomial: The polynomial to use for the generation of the CRC lookup table. The polynomial must be standard
            (non-reflected / non-reversed).
        initial_crc_value: The value to which the CRC checksum is initialized before calculation.
        final_xor_value: The value with which the CRC checksum is XORed after calculation.

    Raises:
        TypeError: If class initialization arguments are not of the valid type.
    """

    def __init__(
        self,
        polynomial: CRCType,
        initial_crc_value: CRCType,
        final_xor_value: CRCType,
    ) -> None:
        # Converts the input polynomial type from numpy to numba format so that it can be used in the spec list below
        if polynomial.dtype is np.dtype(np.uint8):
            crc_type = uint8
        elif polynomial.dtype is np.dtype(np.uint16):
            crc_type = uint16
        else:
            crc_type = uint32

        # The template for the numba compiler to assign specific datatypes to variables used by CRCProcessor class.
        crc_spec = [
            ("polynomial", crc_type),
            ("initial_crc_value", crc_type),
            ("final_xor_value", crc_type),
            ("crc_byte_length", uint8),
            ("crc_table", crc_type[:]),
        ]

        # Initializes and compiles the internal _CRCProcessor class. This automatically generates the static CRC lookup
        # table
        self._processor: _CRCProcessor = jitclass(cls_or_spec=_CRCProcessor, spec=crc_spec)(
            polynomial=polynomial,
            initial_crc_value=initial_crc_value,
            final_xor_value=final_xor_value,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the CRCProcessor object."""
        return (
            f"CRCProcessor(polynomial={hex(self._processor.polynomial)}, "
            f"initial_crc_value={hex(self._processor.initial_crc_value)}, "
            f"final_xor_value={hex(self._processor.final_xor_value)}, "
            f"crc_byte_length={self._processor.crc_byte_length})"
        )

    def calculate_checksum(self, buffer: NDArray[np.uint8], check: bool) -> np.uint16:
        """Calculates the checksum for the data stored in the input buffer.

        Depending on configuration, this method can be used to either generate and write the CRC checksum to the end
        of the packet or to verify the integrity of the incoming packet using its checksum postamble.

        Args:
            buffer: The buffer that contains the COBS-encoded packet for which to resolve the checksum. The buffer must
                include the space for the CRC checksum at the end of the packet.
            check: Determines whether the method is called to verify the incoming packet's data integrity or to
                generate and write the CRC checksum to the outgoing packet's postamble section.

        Returns:
            The calculated numpy uint8, uint16, or uint32 integer CRC checksum value.

        Raises:
            ValueError: If the method is unable to verify the incoming packet's data integrity.
        """
        # Runs the CRC checksum calculation. If the method is called in the check mode and returns 0, this indicates
        # that the CRC computation failed.
        result = self._processor.calculate_checksum(buffer, check)

        if result == 0:
            message = "CRC verification: Failed. The input data packet was corrupted in transmission."
            console.error(
                message=message,
                error=ValueError,
            )

            # Fallback to appease MyPy.
            raise ValueError(message)  # pragma: no cover

        return result

    @property
    def crc_byte_length(self) -> np.uint8:
        """Returns the byte-size used by the CRC checksums."""
        return self._processor.crc_byte_length

    # noinspection PyTypeHints
    @property
    def crc_table(self) -> NDArray[CRCType]:
        """Returns the CRC checksum lookup table."""
        return self._processor.crc_table

    @property
    def processor(self) -> _CRCProcessor:
        """Returns the jit-compiled CRC processor class instance.

        This accessor allows external methods to directly interface with the JIT-compiled class, bypassing the Python
        wrapper.
        """
        return self._processor

    @property
    def polynomial(self) -> CRCType:
        """Returns the polynomial used for checksum calculation."""
        return self._processor.polynomial

    @property
    def initial_crc_value(self) -> CRCType:
        """Returns the initial value used for checksum calculation."""
        return self._processor.initial_crc_value

    @property
    def final_xor_value(self) -> CRCType:
        """Returns the final XOR value used for checksum calculation."""
        return self._processor.final_xor_value


class SerialMock:
    """Mocks the behavior of the PySerial's `Serial` class for testing purposes.

    This class provides a mock implementation of the `Serial` class, enabling unit tests for the TransportLayer class
    without a hardware connection. It replicates the core functionalities of the PySerial's `Serial` class that are
    relevant for testing, such as reading and writing data.

    Attributes:
        is_open: A flag indicating if the mock serial port is open.
        tx_buffer: A byte buffer that stores transmitted data.
        rx_buffer: A byte buffer that stores received data.
    """

    def __init__(self) -> None:
        self.is_open: bool = False
        self.tx_buffer: bytes = b""
        self.rx_buffer: bytes = b""

    def __repr__(self) -> str:
        """Returns a string representation of the SerialMock object."""
        return f"SerialMock(open={self.is_open})"

    def open(self) -> None:
        """Opens the mock serial port, setting `is_open` to True."""
        if not self.is_open:
            self.is_open = True

    def close(self) -> None:
        """Closes the mock serial port, setting `is_open` to False."""
        if self.is_open:
            self.is_open = False

    def write(self, data: bytes) -> None:
        """Writes data to the `tx_buffer`.

        Args:
            data: The serialized data to be written to the output buffer.

        Raises:
            TypeError: If `data` is not a bytes' object.
            RuntimeError: If the mock serial port is not open.
        """
        if self.is_open:
            if isinstance(data, bytes):
                self.tx_buffer += data
            else:
                message = "Data must be a 'bytes' object"
                raise TypeError(message)
        else:
            message = "Mock serial port is not open"
            raise RuntimeError(message)

    def read(self, size: int = 1) -> bytes:
        """Reads a specified number of bytes from the `rx_buffer`.

        Args:
            size: The number of bytes to read from the input buffer.

        Returns:
            A bytes' object containing the requested data from the `rx_buffer`.

        Raises:
            RuntimeError: If the mock serial port is not open.
        """
        if self.is_open:
            data = self.rx_buffer[:size]
            self.rx_buffer = self.rx_buffer[size:]
            return data
        message = "Mock serial port is not open"
        raise RuntimeError(message)

    def reset_input_buffer(self) -> None:
        """Clears the `rx_buffer` attribute.

        Raises:
            RuntimeError: If the mock serial port is not open.
        """
        if self.is_open:
            self.rx_buffer = b""
        else:
            message = "Mock serial port is not open"
            raise RuntimeError(message)

    def reset_output_buffer(self) -> None:
        """Clears the `tx_buffer` attribute.

        Raises:
            RuntimeError: If the mock serial port is not open.
        """
        if self.is_open:
            self.tx_buffer = b""
        else:
            message = "Mock serial port is not open"
            raise RuntimeError(message)

    @property
    def in_waiting(self) -> int:
        """Returns the number of bytes stored in the `rx_buffer`."""
        return len(self.rx_buffer)

    @property
    def out_waiting(self) -> int:
        """Returns the number of bytes stored in the `tx_buffer`."""
        return len(self.tx_buffer)
