"""This module contains the low-level helper classes that support the runtime of TransportLayer class methods."""

from enum import IntEnum
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


class COBSStatusCode(IntEnum):
    """Maps integer status codes for COBS processor operations to human-readable names."""

    STANDBY = 11
    """The initial state for the COBS processor."""
    PAYLOAD_TOO_SMALL_ERROR = 12
    """The input payload array size was below the minimum allowed size. Payload encoding failed."""
    PAYLOAD_TOO_LARGE_ERROR = 13
    """The input payload array size was above the maximum allowed size. Payload encoding failed."""
    INVALID_PAYLOAD_DATATYPE_ERROR = 14
    """The input payload array datatype was not valid (not uint8). Payload encoding failed."""
    PAYLOAD_ENCODED = 15
    """Payload was successfully encoded (converted to a packet)."""
    PACKET_TOO_SMALL_ERROR = 16
    """The input packet array size was below the minimum allowed size. Payload decoding failed."""
    PACKET_TOO_LARGE_ERROR = 17
    """The input packet array size was above the maximum allowed size. Payload decoding failed."""
    DELIMITER_NOT_FOUND_ERROR = 18
    """
    The decoder method did not encounter an unencoded delimiter value while attempting to decode the payload. Payload 
    decoding failed.
    """
    DELIMITER_FOUND_TOO_EARLY_ERROR = 19
    """The decoder method encountered the unencoded delimiter before reaching the end of the packet. Payload decoding 
    failed."""
    INVALID_PACKET_DATATYPE_ERROR = 20
    """The input packet array datatype was not valid (not uint8). Payload decoding failed."""
    PAYLOAD_DECODED = 21
    """The payload was successfully decoded from the packet."""


class _COBSProcessor:  # pragma: no cover
    """Provides methods for encoding and decoding data using the Consistent Overhead Byte Stuffing (COBS) scheme.

    Notes:
        This class is intended to be initialized through Numba's 'jitclass' function.

        See the original paper for the details on COBS methodology and specific data packet layouts:
        S. Cheshire and M. Baker, "Consistent overhead byte stuffing," in IEEE/ACM Transactions on Networking, vol. 7,
        no. 2, pp. 159-172, April 1999, doi: 10.1109/90.769765.

        To support error-handling, class methods return fixed byte error-codes. All error codes are stored in the
        COBSStatusCode enumeration. Each method returns the status (success or error) code by setting the class 'status'
        attribute to the latest runtime code.

    Attributes:
        status: Tracks the outcome of the last class method call.
        maximum_payload_size: The maximum size of the payload, in bytes. Due to COBS, cannot exceed 254 bytes.
        minimum_payload_size: The minimum size of the payload, in bytes.
        maximum_packet_size: The maximum size of the packet, in bytes. Due to COBS, it cannot exceed 256 bytes
            (254 payload bytes + 1 overhead + 1 delimiter byte).
        minimum_packet_size: The minimum size of the packet, in bytes. Due to COBS cannot be below 3 bytes.
    """

    def __init__(self) -> None:
        # Constant class parameters (do not modify, they are already optimal for any non-embedded system)
        self.maximum_payload_size: int = 254
        self.minimum_payload_size: int = 1
        self.maximum_packet_size: int = 256
        self.minimum_packet_size: int = 3

        self.status: int = COBSStatusCode.STANDBY  # Initializes to standby

    def encode_payload(self, payload: NDArray[np.uint8], delimiter: np.uint8) -> NDArray[np.uint8]:
        """Encodes the input payload into a transmittable packet using the COBS scheme.

        Args:
            payload: The numpy array that stores the payload to be encoded using COBS scheme. The payload has to be
                between 1 and 254 bytes in length.
            delimiter: The value that is used as the packet delimiter. It is recommended to use the value '0' for
                maximum security.

        Returns:
            The packet numpy array encoded using the COBS scheme, if the method succeeds. An empty
            uninitialized numpy array otherwise.
        """
        # Saves payload size to a separate variable
        size = payload.size

        # Prevents execution if the packet is too small. It is meaningless to send empty packets.
        if size < self.minimum_payload_size:
            self.status = COBSStatusCode.PAYLOAD_TOO_SMALL_ERROR
            return np.empty(0, dtype=payload.dtype)

        # Prevents execution if the payload is too large. Due to using byte-streams and COBS encoding, the
        # overhead byte can only store a maximum value of 255 and for any payload it should be able to store the
        # distance to the end of the packet. 254 bytes is the maximum size that still fits that requirement once
        # overhead and delimiter are added to the payload.
        if size > self.maximum_payload_size:
            self.status = COBSStatusCode.PAYLOAD_TOO_LARGE_ERROR
            return np.empty(0, dtype=payload.dtype)

        # Ensures that the input payload uses uint8 datatype. Since the library uses byte-streams for
        # communication, this is an important prerequisite.
        if payload.dtype is not np.dtype(np.uint8):
            self.status = COBSStatusCode.INVALID_PAYLOAD_DATATYPE_ERROR
            return np.empty(0, dtype=payload.dtype)

        # Initializes the output array, uses payload size + 2 as size to make space for the overhead and
        # delimiter bytes (see COBS scheme for more details on why this is necessary).
        packet = np.empty(size + 2, dtype=payload.dtype)
        packet[-1] = delimiter  # Sets the last byte of the packet to the delimiter byte value
        packet[1:-1] = payload  # Copies input payload into the packet array, leaving spaces for overhead and delimiter.

        # A tracker variable that is used to calculate the distance to the next delimiter value when an
        # unencoded delimiter is required.
        next_delimiter_position = packet.size - 1  # Initializes to the index of the delimiter value added above

        # Iterates over the payload in reverse and replaces every instance of the delimiter value inside the
        # payload with the distance to the next delimiter value (or the value added to the end of the payload).
        # This process ensures that the delimiter value is only found at the end of the packet and, if the delimiter
        # is not 0, potentially also as the overhead byte value. This encodes the payload using the COBS scheme.
        for i in range(size - 1, -1, -1):  # Loops over every index of the payload
            if payload[i] == delimiter:
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
        self.status = COBSStatusCode.PAYLOAD_ENCODED
        return packet

    def decode_payload(self, packet: NDArray[np.uint8], delimiter: np.uint8) -> NDArray[np.uint8]:
        """Decodes the COBS-encoded payload from the input packet.

        Args:
            packet: The numpy array that stores COBS-encoded packet.
            delimiter: The value that is used as the packet delimiter. It is used to optimize the decoding flow and to
                verify the unencoded delimiter at the end of the packet.

        Returns:
            The payload numpy array decoded from the packet if the method succeeds. An empty uninitialized numpy array
            otherwise.
        """
        # noinspection DuplicatedCode
        size = packet.size  # Extracts packet size for the checks below

        # This is necessary due to how this method is used by the main class, where the input to this method
        # happens to be a 'readonly' array. Copying the array removes the readonly flag.
        packet = packet.copy()

        # Prevents execution if the size of the packet is too small. The packet should at minimum have enough
        # space for the overhead byte, one payload byte and the delimiter byte (3 bytes).
        # noinspection DuplicatedCode
        if size < self.minimum_packet_size:
            self.status = COBSStatusCode.PACKET_TOO_SMALL_ERROR
            return np.empty(0, dtype=packet.dtype)

        # Also prevents execution if the size of the packet is too large. The maximum size is enforced due to
        # how the COBS encoding works, as it requires having at most 255 bytes between the overhead byte and the
        # end of the packet.
        if size > self.maximum_packet_size:
            self.status = COBSStatusCode.PACKET_TOO_LARGE_ERROR
            return np.empty(0, dtype=packet.dtype)

        # Ensures that the input packet uses uint8 datatype. Since the library uses byte-streams for
        # communication, this is an important prerequisite.
        if packet.dtype is not np.dtype(np.uint8):
            self.status = COBSStatusCode.INVALID_PACKET_DATATYPE_ERROR
            return np.empty(0, dtype=packet.dtype)

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
            if packet[read_index] == delimiter:
                if read_index == size - 1:
                    # If the delimiter is found at the end of the packet, extracts and returns the decoded
                    # packet to the caller.
                    self.status = COBSStatusCode.PAYLOAD_DECODED
                    return packet[1:-1]
                # If the delimiter is encountered before reaching the end of the packet, this indicates that
                # the packet was corrupted during transmission and the CRC-check failed to recognize the
                # data corruption. In this case, returns an error code.
                self.status = COBSStatusCode.DELIMITER_FOUND_TOO_EARLY_ERROR
                return np.empty(0, dtype=packet.dtype)

            # If the read_index pointed value is not an unencoded delimiter, first extracts the value and saves
            # it to the next_index, as the value is the distance to the next encoded value or the unencoded
            # delimiter.
            next_index = packet[read_index]

            # Decodes the extracted value by overwriting it with the delimiter value
            packet[read_index] = delimiter

        # If this point is reached, that means that the method did not encounter an unencoded delimiter before
        # reaching the end of the packet. While the reasons for this are numerous, overall that means that the
        # packet is malformed and the data is corrupted, so returns an error code.
        self.status = COBSStatusCode.DELIMITER_NOT_FOUND_ERROR
        return np.empty(0, dtype=packet.dtype)


class COBSProcessor:
    """Exposes a pure-python API for encoding and decoding data using the Consistent Overhead Byte Stuffing (COBS)
    scheme.

    This class wraps a JIT-compiled COBS processor implementation, combining the convenience of a pure-python API with
    the speed of the C-compiled processing code.

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
        ]

        # Instantiates the jit class and saves it to the wrapper class attribute. Developer hint: when used as a
        # function, jitclass returns an uninitialized compiled object, so initializing is crucial here.
        self._processor: _COBSProcessor = jitclass(cls_or_spec=_COBSProcessor, spec=cobs_spec)()

    def __repr__(self) -> str:
        """Returns a string representation of the COBSProcessor class instance."""
        return (
            f"COBSProcessor(status={self._processor.status}, "
            f"maximum_payload_size={self._processor.maximum_payload_size}, "
            f"minimum_payload_size={self._processor.minimum_payload_size}, "
            f"maximum_packet_size={self._processor.maximum_packet_size}, "
            f"minimum_packet_size={self._processor.minimum_packet_size})"
        )

    def encode_payload(self, payload: NDArray[np.uint8], delimiter: np.uint8 = _ZERO) -> NDArray[np.uint8]:
        """Encodes the input payload into a transmittable packet using COBS scheme.

        The encoding produces the following packet structure: [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            payload: The numpy array that stores the payload to be encoded using COBS scheme. Has to be between 1 and
                254 bytes in length.
            delimiter: The value that is used as the packet delimiter.

        Returns:
            The numpy array that stores the serialized packet encoded using the COBS scheme.

        Raises:
            TypeError: If the payload or delimiter arguments are not of the correct numpy datatype.
            ValueError: If encoding failed for any reason.
        """
        # Prevents using the method for unsupported input types
        if not isinstance(payload, np.ndarray):
            message = (
                f"Unable to encode payload using COBS scheme. A numpy ndarray with uint8 datatype expected as "
                f"'payload' argument, but instead encountered {payload} of type {type(payload).__name__}."
            )
            console.error(message, error=TypeError)

        elif not isinstance(delimiter, np.uint8):
            message = (
                f"Unable to encode payload using COBS scheme. A scalar numpy uint8 (byte) value expected as "
                f"'delimiter' argument, but instead encountered {delimiter} of type {type(delimiter).__name__}."
            )
            console.error(message, error=TypeError)

        # Calls the encoding method
        packet = self._processor.encode_payload(payload, delimiter)

        # Resolves method runtime status to see if an error exception needs to be raised or the packet has been encoded
        # and can be returned to caller
        self._resolve_encoding_status(payload)

        # If runtime was successful, returns the packet
        return packet

    def _resolve_encoding_status(self, payload: NDArray[np.uint8]) -> None:
        """Resolves the status of the encode_payload() method runtime.

        If the status indicates that the runtime failed, raises the appropriate error message.

        Args:
            payload: The payload that was passed to the encoding method.
        """
        # Success code, verification is complete
        if self._processor.status == COBSStatusCode.PAYLOAD_ENCODED:
            return

        # Payload too small
        if self._processor.status == COBSStatusCode.PAYLOAD_TOO_SMALL_ERROR:
            message = (
                f"Failed to encode the payload using COBS scheme. The size of the input payload "
                f"({payload.size}) is too small. A minimum size of {self._processor.minimum_payload_size} elements "
                f"(bytes) is required. CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Payload too large
        elif self._processor.status == COBSStatusCode.PAYLOAD_TOO_LARGE_ERROR:
            message = (
                f"Failed to encode the payload using COBS scheme. The size of the input payload ({payload.size}) is "
                f"too large. A maximum size of {self._processor.maximum_payload_size} elements (bytes) is required. "
                f"CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Invalid payload datatype
        elif self._processor.status == COBSStatusCode.INVALID_PAYLOAD_DATATYPE_ERROR:
            message = (
                f"Failed to encode the payload using COBS scheme. The datatype of the input payload "
                f"({payload.dtype}) is not supported. Only uint8 (byte) numpy arrays are currently supported as "
                f"payload inputs. CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Unknown status code
        message = (
            f"Failed to encode the payload using COBS scheme. Unexpected inner _COBSProcessor class status code "
            f"({self._processor.status}) encountered. CODE: 0."
        )  # pragma: no cover
        console.error(message, error=RuntimeError)  # pragma: no cover

    def decode_payload(self, packet: NDArray[np.uint8], delimiter: np.uint8 = _ZERO) -> NDArray[np.uint8]:
        """Decodes the COBS-encoded payload from the input packet.

        Expects the input packets to adhere to the following structure:
        [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            packet: The numpy array that stores COBS-encoded packet.
            delimiter: The value that is used as the packet delimiter.

        Returns:
            The numpy array that stores the payload decoded from the packet.

        Raises:
            TypeError: If the packet or delimiter arguments are not of the correct numpy datatype.
            ValueError: If decoding failed for any reason.
        """
        # Prevents using the method for unsupported input types
        if not isinstance(packet, np.ndarray):
            message = (
                f"Unable to decode payload using COBS scheme. A numpy ndarray expected as 'packet' argument, but "
                f"instead encountered {packet} of type {type(packet).__name__}."
            )
            console.error(message, error=TypeError)

        if not isinstance(delimiter, np.uint8):
            message = (
                f"Unable to decode payload using COBS scheme. A scalar numpy uint8 (byte) value expected as "
                f"'delimiter' argument, but instead encountered {delimiter} of type {type(delimiter).__name__}."
            )
            console.error(message, error=TypeError)

        # Calls decoding method
        payload = self._processor.decode_payload(packet, delimiter)

        # Verifies the outcome of decoding method runtime
        self._resolve_decoding_status(packet)

        # Returns the decoded payload to caller if verification was successful
        return payload

    def _resolve_decoding_status(self, packet: NDArray[np.uint8]) -> None:
        """Resolves the status of the decode_payload() method runtime.

        If the status indicates that the runtime failed, raises the appropriate error message.

        Args:
            packet: The packet array that was passed to the decoding method.
        """
        # Runtime successful, verification is complete
        if self._processor.status == COBSStatusCode.PAYLOAD_DECODED:
            return

        # Packet too small
        if self._processor.status == COBSStatusCode.PACKET_TOO_SMALL_ERROR:
            message = (
                f"Failed to decode payload using COBS scheme. The size of the input packet ({packet.size}) is too "
                f"small. A minimum size of {self._processor.minimum_packet_size} elements (bytes) is required. "
                f"CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Packet too large
        elif self._processor.status == COBSStatusCode.PACKET_TOO_LARGE_ERROR:
            message = (
                f"Failed to decode payload using COBS scheme. The size of the input packet ({packet.size}) is too "
                f"large. A maximum size of {self._processor.maximum_packet_size} elements (bytes) is required. "
                f"CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Invalid packet datatype
        elif self._processor.status == COBSStatusCode.INVALID_PACKET_DATATYPE_ERROR:
            message = (
                f"Failed to decode payload using COBS scheme. The datatype of the input packet ({packet.dtype}) is "
                f"not supported. Only uint8 (byte) numpy arrays are currently supported as packet inputs. "
                f"CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Delimiter isn't found at the end of the packet or 'jumping' does not point at the end of the packet. Indicates
        # packet corruption.
        elif self._processor.status == COBSStatusCode.DELIMITER_NOT_FOUND_ERROR:
            message = (
                f"Failed to decode payload using COBS scheme. The decoder did not find the unencoded delimiter "
                f"at the end of the packet. This is either because the end-value is not an unencoded delimiter or "
                f"because the decoding does not end at the final index of the packet. Packet is likely "
                f"corrupted. CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Delimiter encountered before reaching the end of the packet. Indicates packet corruption.
        elif self._processor.status == COBSStatusCode.DELIMITER_FOUND_TOO_EARLY_ERROR:
            message = (
                f"Failed to decode payload using COBS scheme. Found unencoded delimiter before reaching the end of "
                f"the packet. Packet is likely corrupted. CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Unknown code
        message = (
            f"Failed to decode payload using COBS scheme. Unexpected inner _COBSProcessor class status code "
            f"({self._processor.status}) encountered. CODE: 0."
        )  # pragma: no cover
        console.error(message, error=RuntimeError)  # pragma: no cover

    @property
    def processor(self) -> _COBSProcessor:
        """Returns the jit-compiled COBS processor class instance.

        This accessor allows external methods to directly interface with the JIT-compiled class, bypassing the Python
        wrapper.
        """
        return self._processor


class CRCStatusCode(IntEnum):
    """Maps integer status codes for CRC processor operations to human-readable names."""

    STANDBY = 51
    """The initial state for the CRC processor."""
    DATA_BUFFER_DATATYPE_ERROR = 52
    """The buffer storing the data to be checksummed was not of the uint8 numpy datatype. CRC calculation failed."""
    CHECKSUM_CALCULATED = 53
    """The CRC checksum has been successfully calculated."""
    CHECKSUM_SERIALIZED = 54
    """The CRC checksum has been successfully serialized."""
    CHECKSUM_BUFFER_DATATYPE_ERROR = 55
    """The buffer storing the serialized CRC checksum to be deserialized was not of the required datatype."""
    CHECKSUM_BUFFER_SIZE_ERROR = 56
    """The size of the buffer storing the serialized CRC checksum to be deserialized does not match the expected 
    byte-size for the checksum's type."""
    CHECKSUM_DESERIALIZED = 57
    """The CRC checksum has been successfully deserialized into an appropriate unsigned integer type."""


class _CRCProcessor:  # pragma: no cover
    """Provides methods for working with Cyclic Redundancy Check (CRC) checksums used to verify the integrity of
    transferred data packets.

    Notes:
        This class is intended to be initialized through Numba's 'jitclass' function.

        For more information on how the CRC checksum works, see the original paper:
        W. W. Peterson and D. T. Brown, "Cyclic Codes for Error Detection," in Proceedings of the IRE, vol. 49, no. 1,
        pp. 228-235, Jan. 1961, doi: 10.1109/JRPROC.1961.287814.

        To support error-handling, the class returns fixed byte error-codes. All error codes are stored in the
        CRCStatusCodes enumeration. Each method returns the status (success or error) code by setting the class 'status'
        attribute to the latest runtime code.

        To increase runtime speed, this class generates a static CRC lookup table using the input polynomial, which is
        subsequently used to calculate CRC checksums.

    Attributes:
        status: Tracks the outcome of the last class method call
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
        polynomial: np.uint8 | np.uint16 | np.uint32,
        initial_crc_value: np.uint8 | np.uint16 | np.uint32,
        final_xor_value: np.uint8 | np.uint16 | np.uint32,
    ) -> None:
        # No error checking, as it is assumed that the class is always initialized through the CRCProcessor wrapper.

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
        self.polynomial: np.uint8 | np.uint16 | np.uint32 = polynomial
        self.initial_crc_value: np.uint8 | np.uint16 | np.uint32 = initial_crc_value
        self.final_xor_value: np.uint8 | np.uint16 | np.uint32 = final_xor_value
        self.crc_byte_length: np.uint8 = polynomial_size
        self.crc_table = np.empty(256, dtype=crc_type)  # Initializes to empty for efficiency

        self.status: int = CRCStatusCode.STANDBY  # Dynamically updated to track the latest method runtime status

        # Generates the lookup table based on the target polynomial parameters and iteratively sets each variable
        # inside the crc_table placeholder to the calculated values.
        self._generate_crc_table(polynomial=polynomial)

    # noinspection DuplicatedCode
    def calculate_crc_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Calculates the CRC checksum for the input buffer.

        Notes:
            While error runtimes always return 0, a 0-value returned by this method is not necessarily indicative of an
            error. To determine if the method runtime was successful, use the 'status' class attribute.

        Args:
            buffer: The numpy array that stores the data to be checksummed.

        Returns:
            A numpy uint8, uint16, or uint32 integer that represents the calculated CRC checksum value.
        """
        # Verifies that the buffer is using an appropriate datatype (uint8). This method is intended to work
        # with buffers storing byte-serialized data, so explicitly controls for that here.
        if buffer.dtype is not np.dtype(np.uint8):
            self.status = CRCStatusCode.DATA_BUFFER_DATATYPE_ERROR
            return np.uint8(0)

        # Initializes the checksum. The datatype is already correct as it is inferred from the initial_crc_value
        # datatype
        crc_checksum = self.initial_crc_value

        # Loops over each byte inside the buffer and iteratively calculates CRC checksum for the buffer
        for byte in buffer:
            # Calculates the index to retrieve from the CRC table. To do so, combines the high byte of the CRC
            # checksum with the (possibly) modified (corrupted) data_byte using bitwise XOR.
            table_index = (crc_checksum >> (8 * (self.crc_byte_length - 1))) ^ byte

            # Extracts the byte-specific CRC value from the table using the result of the operation above. The
            # retrieved CRC value from the table is then XORed with the checksum that is shifted back to the
            # original position to generate an updated checksum.
            crc_checksum = self._make_polynomial_type((crc_checksum << 8) ^ self.crc_table[table_index])

        # The Final XOR operation may or may not be used (depending on the polynomial). The default
        # polynomial 0x1021 has it set to 0x0000 (0), so it is actually not used. Other polynomials may require
        # this step, so it is kept here for compatibility reasons. The exact algorithmic purpose of the XOR
        # depends on the specific polynomial used.
        crc_checksum ^= self.final_xor_value

        # Sets the status to indicate runtime success and returns calculated checksum to the caller.
        self.status = CRCStatusCode.CHECKSUM_CALCULATED
        return self._make_polynomial_type(crc_checksum)

    def serialize_checksum(self, crc_checksum: np.uint8 | np.uint16 | np.uint32) -> NDArray[np.uint8]:
        """Converts the input checksum value into a numpy array of bytes.

        This method converts a multibyte CRC checksum into a sequence of individual bytes and writes them to a numpy
        uint8 array starting with the highest byte of the checksum.

        Returns:
            A uint8 numpy array that stores the serialized CRC checksum.
        """
        # Precreates the buffer array to store the byte-converted checksum
        buffer = np.empty(self.crc_byte_length, dtype=np.uint8)

        # Appends the CRC checksum to the buffer, starting with the most significant byte (loops over each byte
        # and iteratively adds it to the buffer).
        for i in range(self.crc_byte_length):
            # Extracts the byte from the checksum and inserts it into the buffer. Most of this instruction
            # controls which byte making up the CRC checksum is processed by each iteration of the loop
            buffer[i] = (crc_checksum >> (8 * (self.crc_byte_length - i - 1))) & 0xFF

        # Returns the filled buffer to caller and sets the status to communicate runtime success.
        self.status = CRCStatusCode.CHECKSUM_SERIALIZED
        return buffer

    def deserialize_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Converts the serialized CRC checksum stored in the input buffer to an unsigned numpy integer value.

        This method is used to convert uint8 (byte) numpy arrays to crc checksum integer values. The method assumes
        that the checksum has been converted to bytes starting with the highest byte of the checksum and that the buffer
        is entirely filled with the checksum bytes.

        Notes:
            While error runtimes always return 0, a 0-value returned by this method is not necessarily indicative of an
            error. To determine if the method runtime was successful, use the 'status' class attribute.

        Returns:
            The deserialized numpy uint8, uint16, or uint32 integer checksum value.
        """
        # Verifies that the input buffer uses an appropriate (uint8) datatype. This method is intended to decode
        # CRC checksum values from serialized byte-streams and will not work properly with any other data types.
        if buffer.dtype is not np.dtype(np.uint8):
            self.status = CRCStatusCode.CHECKSUM_BUFFER_DATATYPE_ERROR
            # Note, 0 is a valid value. The only way to know if it comes from a successful or failed runtime is
            # to check the class 'status' attribute that communicates the latest runtime success or error code.
            return np.uint8(0)

        # Ensures that the buffer size exactly matches the number of bytes required to store the CRC checksum.
        if buffer.size != self.crc_byte_length:
            self.status = CRCStatusCode.CHECKSUM_BUFFER_SIZE_ERROR
            return np.uint8(0)

        # Precreates the variable to store the extracted checksum and initializes it to zero
        extracted_crc = self._make_polynomial_type(0)

        # Loops over the input buffer and extracts the CRC checksum from the bytes inside the buffer. Assumes
        # the buffer is entirely filled with the checksum bytes and uses crc_byte_length to constrain processing
        # to the exact number of bytes required.
        for i in range(self.crc_byte_length):
            # Constructs the CRC checksum from the buffer, starting from the most significant byte and moving
            # towards the least significant byte. This matches the process of how it was converted to bytes by
            # the convert_checksum_to_bytes() or an equivalent microcontroller method.
            extracted_crc |= self._make_polynomial_type(buffer[i] << (8 * (self.crc_byte_length - i - 1)))

        # Returns the extracted CRC checksum to caller and sets the status to communicate runtime success.
        self.status = CRCStatusCode.CHECKSUM_DESERIALIZED
        return extracted_crc

    def _generate_crc_table(self, polynomial: np.uint8 | np.uint16 | np.uint32) -> None:
        """Uses the input polynomial to compute the CRC checksums for each
        possible uint8 (byte) value.

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

    def _make_polynomial_type(self, value: Any) -> np.uint8 | np.uint16 | np.uint32:
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
    """Exposes a pure-python API for working with Cyclic Redundancy Check (CRC) checksums used to verify the integrity
    of transferred data packets.

    This class wraps a JIT-compiled CRC processor implementation, combining the convenience of a pure-python API with
    the speed of the C-compiled processing code.

    Attributes:
        _processor: Stores the jit-compiled _CRCProcessor instance, which carries out all computations.

    Args:
        polynomial: The polynomial used to generate the CRC lookup table. Can be provided as a HEX number
            (e.g., 0x1021). Currently only non-reversed polynomials of numpy uint8, uint16, and uint32 datatypes are
            supported.
        initial_crc_value: The initial value to which the CRC checksum variable is initialized during calculation.
            This value depends on the chosen polynomial algorithm and should use the same datatype as the polynomial
            argument. It can be provided as a HEX number (e.g., 0xFFFF).
        final_xor_value: The final XOR value to be applied to the calculated CRC checksum value. This value depends on
            the chosen polynomial algorithm and should use the same datatype as the polynomial argument. It can be
            provided as a HEX number (e.g., 0x0000).

    Raises:
        TypeError: If class initialization arguments are not of the valid type.
    """

    def __init__(
        self,
        polynomial: np.uint8 | np.uint16 | np.uint32,
        initial_crc_value: np.uint8 | np.uint16 | np.uint32,
        final_xor_value: np.uint8 | np.uint16 | np.uint32,
    ) -> None:
        # Ensures that all inputs use the same valid type. Note, uint64 is currently not supported primarily to maintain
        # implicit compatibility with older AVR boards that do not support the uint64 type. That said, both the C++ and
        # this Python codebase are written in a way that will natively scale to uint 64 if this static guard is
        # modified to allow it.
        if not isinstance(polynomial, (np.uint8, np.uint16, np.uint32)):
            message = (
                f"Unable to initialize the CRCProcessor class. A numpy uint8, uint16 or uint32 scalar expected as "
                f"'polynomial' argument, but encountered {polynomial} of type {type(polynomial).__name__}."
            )
            console.error(message, error=TypeError)

        if not isinstance(initial_crc_value, (np.uint8, np.uint16, np.uint32)):
            message = (
                f"Unable to initialize the CRCProcessor class. A numpy uint8, uint16 or uint32 scalar expected as "
                f"'initial_crc_value' argument, but encountered {initial_crc_value} of type "
                f"{type(initial_crc_value).__name__}."
            )
            console.error(message, error=TypeError)

        if not isinstance(final_xor_value, (np.uint8, np.uint16, np.uint32)):
            message = (
                f"Unable to initialize the CRCProcessor class. A numpy uint8, uint16 or uint32 scalar expected as "
                f"'final_xor_value' argument, but encountered {final_xor_value} of "
                f"type {type(final_xor_value).__name__}."
            )
            console.error(message, error=TypeError)

        if not (polynomial.dtype == initial_crc_value.dtype == final_xor_value.dtype):
            message = (
                "Unable to initialize the CRCProcessor class. All arguments "
                "('polynomial', 'initial_crc_value', 'final_xor_value') must have the same data type. Instead, "
                f"encountered ({polynomial.dtype}, {initial_crc_value.dtype}, {final_xor_value.dtype})"
            )
            console.error(message, error=TypeError)

        # Converts the input polynomial type from numpy to numba format so that it can be used in the spec list below
        if polynomial.dtype is np.dtype(np.uint8):
            crc_type = uint8
        elif polynomial.dtype is np.dtype(np.uint16):
            crc_type = uint16
        else:
            crc_type = uint32

        # The template for the numba compiler to assign specific datatypes to variables used by CRCProcessor class.
        crc_spec = [
            ("status", uint8),
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
            f"CRCProcessor(status={self._processor.status}, "
            f"polynomial={hex(self._processor.polynomial)}, "
            f"initial_crc_value={hex(self._processor.initial_crc_value)}, "
            f"final_xor_value={hex(self._processor.final_xor_value)}, "
            f"crc_byte_length={self._processor.crc_byte_length})"
        )

    def calculate_crc_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Calculates the CRC checksum for the data in the input buffer.

        Args:
            buffer: The uint8 numpy array that stores the data to be checksummed.

        Returns:
            The calculated numpy uint8, uint16, or uint32 integer CRC checksum value.

        Raises:
            TypeError: If the input buffer is not a numpy array.
            ValueError: If CRC checksum calculation fails for any reason.
        """
        # Prevents using the method for unsupported input types
        if not isinstance(buffer, np.ndarray):
            message = (
                f"Unable to calculate the CRC checksum for the input buffer. A uint8 numpy ndarray expected as "
                f"'buffer' argument, but instead encountered {buffer} of type {type(buffer).__name__}."
            )
            console.error(message, error=TypeError)

        # Calls the appropriate _CRCProcessor method to calculate the crc checksum
        checksum = self._processor.calculate_crc_checksum(buffer)

        # Verifies that checksum calculation was successful
        self._resolve_checksum_calculation_status(buffer)

        # Since other methods expect numpy values, the checksum is explicitly cast to the correct type here. Numba has a
        # limitation, where it prefers python types and casts all outputs to them regardless of the type assigned during
        # numba runtime. This is why the types need to be resolved explicitly at the level of the wrapper.
        if self._processor.crc_byte_length == _ONE_BYTE:
            return np.uint8(checksum)
        if self._processor.crc_byte_length == _TWO_BYTE:
            return np.uint16(checksum)
        return np.uint32(checksum)

    def _resolve_checksum_calculation_status(self, buffer: NDArray[np.uint8]) -> None:
        """Resolves the status of the calculate_crc_checksum() method runtime.

        If the runtime failed, raises the appropriate error message.

        Args:
            buffer: The data buffer that was provided to the CRC checksum calculation method.
        """
        # Success code, verification successful
        if self._processor.status == CRCStatusCode.CHECKSUM_CALCULATED:
            return

        # Incorrect buffer datatype
        if self._processor.status == CRCStatusCode.DATA_BUFFER_DATATYPE_ERROR:
            message = (
                f"CRC checksum calculation failed. The datatype of the input buffer ({buffer.dtype}) is not "
                f"supported. Only uint8 (byte) numpy arrays are currently supported as buffer inputs. "
                f"CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Unexpected status code
        message = (
            f"CRC checksum calculation failed. Unexpected inner _CRCProcessor class status code "
            f"({self._processor.status}) encountered. CODE: 0."
        )  # pragma: no cover
        console.error(message, error=RuntimeError)  # pragma: no cover

    def serialize_checksum(self, crc_checksum: np.uint8 | np.uint16 | np.uint32) -> NDArray[np.uint8]:
        """Converts the input integer CRC checksum value into a byte numpy array.

        Returns:
            A uint8 numpy array entirely filled with the serialized CRC checksum bytes.

        Raises:
            TypeError: If the input crc_checksum is not a numpy uint8, uint16, or uint32 integer.
            ValueError: If checksum conversion fails for any reason.
        """
        # Prevents using the method for unsupported input types
        if not isinstance(crc_checksum, (np.uint8, np.uint16, np.uint32)):
            message = (
                f"Unable to convert the CRC checksum scalar to an array of bytes. A uint8, uint16 or uint32 "
                f"value expected as 'crc_checksum' argument, but instead encountered {crc_checksum} of type "
                f"{type(crc_checksum).__name__}."
            )
            console.error(message, error=TypeError)

        # Calls the appropriate _CRCProcessor method to convert the crc checksum to an array of bytes
        checksum_bytes = self._processor.serialize_checksum(crc_checksum)

        # At the time of writing this method cannot fail, and this is more or less a static check that the returned
        # code matches the success code in case something changes in the future.
        self._resolve_checksum_serialization_status()
        return checksum_bytes

    def _resolve_checksum_serialization_status(self) -> None:
        """Resolves the status of the serialize_checksum() method runtime.

        If the runtime failed, raises the appropriate error message.
        """
        # Success code, verification successful
        if self._processor.status == CRCStatusCode.CHECKSUM_SERIALIZED:
            return

        # Unknown status code
        message = (
            f"CRC checksum to bytes conversion failed. Unexpected inner _CRCProcessor class status code "
            f"({self._processor.status}) encountered. CODE: 0."
        )  # pragma: no cover
        console.error(message, error=RuntimeError)  # pragma: no cover

    def deserialize_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Converts the input buffer that stores serialized CRC checksum to an integer checksum value.

        Returns:
            The deserialized numpy uint8, uint16, or uint32 integer CRC checksum value.

        Raises:
            TypeError: If the input buffer is not a numpy array.
            ValueError: If checksum conversion fails for any reason.
        """
        # Prevents using the method for unsupported input types
        if not isinstance(buffer, np.ndarray):
            message = (
                f"Unable to convert the array of bytes to the CRC checksum. A uint8 numpy ndarray expected as 'buffer' "
                f"argument, but instead encountered {buffer} of type {type(buffer).__name__}."
            )
            console.error(message, error=TypeError)

        # Calls the appropriate _CRCProcessor method to convert the aray of crc checksum bytes to an integer value
        checksum = self._processor.deserialize_checksum(buffer)

        # Verifies method runtime status
        self._resolve_checksum_deserialization_status(buffer)

        # Since other methods expect numpy values, the checksum is explicitly cast to the correct type here. Numba has a
        # limitation, where it prefers python types and casts all outputs to them regardless of the type assigned during
        # numba runtime. This is why the types need to be resolved explicitly at the level of the wrapper.
        if self._processor.crc_byte_length == _ONE_BYTE:
            return np.uint8(checksum)
        if self._processor.crc_byte_length == _TWO_BYTE:
            return np.uint16(checksum)
        return np.uint32(checksum)

    def _resolve_checksum_deserialization_status(self, buffer: NDArray[np.uint8]) -> None:
        """Resolves the status of the deserialize_checksum() method runtime.

        If the runtime failed, raises the appropriate error message.

        Args:
            buffer: The buffer that was provided to the CRC checksum conversion method.
        """
        # Success code, verification successful
        if self._processor.status == CRCStatusCode.CHECKSUM_DESERIALIZED:
            return

        # Invalid buffer datatype
        if self._processor.status == CRCStatusCode.CHECKSUM_BUFFER_DATATYPE_ERROR:
            message = (
                f"Bytes to CRC checksum conversion failed. The datatype of the input buffer to be converted "
                f"({buffer.dtype}) is not supported. Only uint8 (byte) numpy arrays are currently supported as buffer "
                f"inputs. CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # The size of the buffer does not match the number of bytes required to represent the checksum datatype
        elif self._processor.status == CRCStatusCode.CHECKSUM_BUFFER_SIZE_ERROR:
            message = (
                f"Bytes to CRC checksum conversion failed. The byte-size of the input buffer to be converted "
                f"({buffer.size}) does not match the size required to represent the specified checksum datatype "
                f"({self._processor.crc_byte_length}). CODE: {self._processor.status}."
            )
            console.error(message, error=ValueError)

        # Unknown status code
        message = (
            f"Bytes to CRC checksum conversion failed. Unexpected inner _CRCProcessor class status code "
            f"({self._processor.status}) encountered when. CODE: 0."
        )  # pragma: no cover
        console.error(message, error=RuntimeError)  # pragma: no cover

    @property
    def crc_byte_length(self) -> np.uint8:
        """Returns the byte-size used by the CRC checksums."""
        return self._processor.crc_byte_length

    @property
    def crc_table(self) -> NDArray[np.uint8 | np.uint16 | np.uint32]:
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
    def polynomial(self) -> np.uint8 | np.uint16 | np.uint32:
        """Returns the polynomial used for checksum calculation."""
        return self._processor.polynomial

    @property
    def initial_crc_value(self) -> np.uint8 | np.uint16 | np.uint32:
        """Returns the initial value used for checksum calculation."""
        return self._processor.initial_crc_value

    @property
    def final_xor_value(self) -> np.uint8 | np.uint16 | np.uint32:
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
