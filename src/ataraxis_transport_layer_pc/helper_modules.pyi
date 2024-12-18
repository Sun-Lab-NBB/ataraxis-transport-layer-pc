from typing import Any

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

class _COBSProcessor:
    """Provides methods for encoding and decoding data using the Consistent Overhead Byte Stuffing (COBS) scheme.

    This class is intended to be initialized through Numba\'s \'jitclass\' function. The intended way to do so is through
    first initializing a COBSProcessor (no underscore) class and then accessing the jit-compiled core through the
    \'processor\' property. Initializing this class directly will not have the tangible performance benefits offered by
    the jit-compiled class.

    Notes:
        See the original paper for the details on COBS methodology and specific data packet layouts:
        S. Cheshire and M. Baker, "Consistent overhead byte stuffing," in IEEE/ACM Transactions on Networking, vol. 7,
        no. 2, pp. 159-172, April 1999, doi: 10.1109/90.769765.

        To support error-handling, the class returns fixed byte error-codes. Available error codes can be obtained
        via class attributes. Each method returns the status (success or error) code by setting the class \'status\'
        attribute to the latest runtime code.

    Attributes:
        status: Tracks the latest method runtime status byte-code.
        standby: The integer code used during class initialization (before any method is called).
        maximum_payload_size: The maximum size of the payload, in bytes. Due to COBS, cannot exceed 254 bytes.
        minimum_payload_size: The minimum size of the payload, in bytes. No algorithmic minimum enforced, but
            does not make sense to have it below 1 byte.
        maximum_packet_size: The maximum size of the packet, in bytes. Due to COBS, it cannot exceed 256 bytes
            (254 payload bytes + 1 overhead + 1 delimiter byte).
        minimum_packet_size: The minimum size of the packet, in bytes. Due to COBS cannot be below 3 bytes.
        payload_too_small_error: The input payload array size was below the min_payload_size during encoding.
        payload_too_large_error: The input payload array size was above the max_payload_size during encoding.
        invalid_payload_datatype_error: The input payload array datatype was not valid for the encoding
            method (not uint8).
        payload_encoded: Payload was successfully encoded (into a packet).
        packet_too_small_error: The input packet array size was below the min_packet_size during decoding.
        packet_too_large_error: The input packet array size was above the max_packet_size during decoding.
        delimiter_not_found_error: The decoder method did not encounter an unencoded delimiter during its
            runtime.
        delimiter_found_too_early_error: The decoder method encountered the unencoded delimiter before reaching
            the end of the packet.
        invalid_packet_datatype_error: The input packet array datatype was not valid for the decoder method
            (not uint8).
        payload_decoded: Packet was successfully decoded into payload.
    """

    maximum_payload_size: int
    minimum_payload_size: int
    maximum_packet_size: int
    minimum_packet_size: int
    standby: int
    payload_too_small_error: int
    payload_too_large_error: int
    invalid_payload_datatype_error: int
    payload_encoded: int
    packet_too_small_error: int
    packet_too_large_error: int
    delimiter_not_found_error: int
    delimiter_found_too_early_error: int
    invalid_packet_datatype_error: int
    payload_decoded: int
    status: Incomplete
    def __init__(self) -> None: ...
    def encode_payload(self, payload: NDArray[np.uint8], delimiter: np.uint8 = ...) -> NDArray[np.uint8]:
        """Encodes the input payload into a transmittable packet using COBS scheme.

        Eliminates all instances of the delimiter value from the payload by replacing each with the distance to the
        next consecutive instance or, if no more instances are discovered, to the end of the payload. Next, prepends an
        overhead byte value to the beginning of the payload. The overhead byte stores the distance to the first
        instance of the (eliminated) delimiter value or the end of the payload. Finally, appends an unencoded
        delimiter byte value to the end of the payload to mark the end of the packet.

        Args:
            payload: The numpy array that stores the payload to be encoded using COBS scheme. Has to use uint8
                datatype and be between 1 and 254 bytes in length.
            delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter.

        Returns:
            The packet uint8 numpy array encoded using COBS scheme, if the method succeeds or an empty
            uninitialized numpy array otherwise. Sets the class status to the method runtime status code.
        """
    def decode_payload(self, packet: NDArray[np.uint8], delimiter: np.uint8 = ...) -> NDArray[np.uint8]:
        """Decodes the COBS-encoded payload from the input packet.

        Traverses the input packet by jumping between encoded delimiter values and restoring them to the original value.
        Removes the overhead and the delimiter bytes once the payload has been decoded. This method doubles-up as packet
        corruption detector. Specifically, it expects that the input packet always ends with the unencoded delimiter
        and that there are no unencoded delimiter occurrences amongst the traversed variables. Any deviation from this
        expectation is interpreted as packet corruption.

        Args:
            packet: The numpy array that stores COBS-encoded packet. The array should be using uint8 datatype
                and has to be entirely filled with the packet data. That means that the first index (0) should
                store the overhead byte and the last valid index of the packet should store the unencoded
                delimiter. The packet should be between 3 and 256 bytes in length.
            delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter. It is used to
                optimize the decoding flow and to verify the unencoded delimiter at the end of the packet.

        Returns:
            The payload uint8 numpy array decoded from the packet if the method succeeds or an empty
            uninitialized numpy array otherwise. Sets the class status to the method runtime status code.
        """

class COBSProcessor:
    """Wraps a jit-compiled _COBSProcessor class that provides methods for encoding and decoding data using the
    Consistent Overhead Byte Stuffing (COBS) scheme.

    This class functions as a wrapper that provides a consistent Python API for the internal instance of a
    jit-compiled _COBSProcessor class. This allows achieving python-like experience when using the class, while
    simultaneously benefiting from fast compiled code generated through numba jit-optimization. The wrapper
    automatically converts internal class runtime status codes into exception error messages where appropriate to
    notify users about runtime errors.

    Notes:
        For the maximum execution speed, you can access the private methods directly via the 'processor' property,
        although this is highly discouraged.

        See the API documentation for the _COBSProcessor class for more details about the COBS encoding and decoding
        methodology.

    Attributes:
        _processor: Stores the jit-compiled _COBSProcessor class, which carries out all computations.
    """

    _processor: Incomplete
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the COBSProcessor class instance."""
    def encode_payload(self, payload: NDArray[np.uint8], delimiter: np.uint8 = ...) -> NDArray[np.uint8]:
        """Encodes the input payload into a transmittable packet using COBS scheme.

        The encoding produces the following packet structure: [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            payload: The numpy array that stores the payload to be encoded using COBS scheme. Has to use uint8
                datatype and be between 1 and 254 bytes in length.
            delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter.

        Returns:
            The packet uint8 numpy array encoded using COBS scheme.

        Raises:
            TypeError: If the payload or delimiter arguments are not of the correct numpy datatype.
            ValueError: If encoding failed for any reason.
        """
    def _resolve_encoding_status(self, payload: NDArray[np.uint8]) -> None:
        """Resolves the status of the encode_payload() method runtime.

        If the status was not successful, raises the appropriate error message.

        Args:
            payload: The payload that was passed to the encoding method.

        Raises:
            ValueError: If input payload parameters were not valid.
            RuntimeError: If the status code returned by the encoder method is not one of the expected values.
        """
    def decode_payload(self, packet: NDArray[np.uint8], delimiter: np.uint8 = ...) -> NDArray[np.uint8]:
        """Decodes the COBS-encoded payload from the input packet.

        Expects the input packets to adhere to the following structure:
        [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            packet: The numpy array that stores COBS-encoded packet. The array should be using uint8 datatype and has to
                be entirely filled with the packet data. The first index (0) should store the overhead byte, and the
                last valid index of the packet should store the unencoded delimiter. The packet should be between 3 and
                256 bytes in length.
            delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter. This input is used to
                optimize the decoding flow and to verify the unencoded delimiter at the end of the packet.

        Returns:
            The payload uint8 numpy array decoded from the packet.

        Raises:
            TypeError: If the packet or delimiter arguments are not of a correct numpy datatype.
            ValueError: If decoding failed for any reason.
        """
    def _resolve_decoding_status(self, packet: NDArray[np.uint8]) -> None:
        """Resolves the status of the decode_payload() method runtime.

        If the status was not successful, raises the appropriate error message.

        Args:
            packet: The packet array that was passed to the decoding method.

        Raises:
            ValueError: If the parameters of the input packet are not valid. This includes the case of the packet being
                corrupted.
            RuntimeError: If the status code returned by the decoder method is not one of the expected values.
        """
    @property
    def processor(self) -> _COBSProcessor:
        """Returns the jit-compiled _COBSProcessor class instance.

        This accessor represents a convenient way of unwrapping the jit-compiled class, so that its methods can be
        used directly. This is helpful for using them from other jit-methods or to bypass the overhead of error
        checking.
        """

class _CRCProcessor:
    """Provides methods for working with CRC checksums used to verify the integrity of transferred data packets.

    This class is intended to be initialized through Numba\'s \'jitclass\' function. The intended way to do so is through
    first initializing a CRCProcessor (no underscore) class and then accessing the jit-compiled core through the
    \'processor\' property. Initializing this class directly will not have the tangible performance benefits offered by
    the jit-compiled class.

    Notes:
        For more information on how the CRC checksum works, see the original paper:
        W. W. Peterson and D. T. Brown, "Cyclic Codes for Error Detection," in Proceedings of the IRE, vol. 49, no. 1,
        pp. 228-235, Jan. 1961, doi: 10.1109/JRPROC.1961.287814.

        To support error-handling, the class returns fixed byte error-codes. Available error codes can be obtained
        via class attributes. Each method returns the status (success or error) code by setting the class \'status\'
        attribute to the latest runtime code.

        To increase runtime speed, this class generates a static CRC lookup table using the input polynomial, which is
        subsequently used to calculate CRC checksums. This statically reserves 256, 512, or 1024 bytes of RAM to store
        the table.

    Attributes:
        status: Tracks the latest method runtime status byte-code.
        polynomial: Stores the polynomial used for the CRC checksum calculation.
        initial_crc_value: Stores the initial value used for the CRC checksum calculation.
        final_xor_value: Stores the final XOR value used for the CRC checksum calculation.
        crc_byte_length: Stores the length of the CRC polynomial in bytes.
        crc_table: The array that stores the CRC lookup table. The lookup table is used to speed up CRC checksum
            calculation by pre-computing the checksum value for each possible byte-value (from 0 through 255:
            256 values total).
        standby: The integer code used during class initialization (before any method is called).
        calculate_checksum_buffer_datatype_error: The buffer provided to the calculate_crc_checksum()
            method was not of the required uint8 numpy datatype.
        checksum_calculated: The CRC checksum has been successfully calculated.
        checksum_converted_to_bytes: The CRC checksum has been successfully converted to an uint8 numpy array.
        convert_checksum_invalid_buffer_datatype_error: The buffer provided to the
            convert_bytes_to_checksum() method was not of the required uint8 numpy datatype.
        convert_checksum_invalid_buffer_size_error: The buffer provided to the convert_bytes_to_checksum()
            method was not of the byte-size required to store the byte-converted crc checksum value.
        checksum_converted_to_integer: The CRC checksum has been successfully converted from an uint8 numpy
            array of bytes to an appropriate numpy unsigned integer (uint8, uint16, or uint32).

    Args:
        polynomial: The polynomial used to generate the CRC lookup table. Can be provided as a HEX number
            (e.g., 0x1021). Currently only non-reversed polynomials of numpy uint8, uint16 and uint32 datatypes are
            supported.
        initial_crc_value: The initial value to which the CRC checksum variable is initialized during calculation.
            This value depends on the chosen polynomial algorithm and should use the same datatype as the polynomial
            argument. It can be provided as a HEX number (e.g., 0xFFFF).
        final_xor_value: The final XOR value to be applied to the calculated CRC checksum value. This value depends on
            the chosen polynomial algorithm and should use the same datatype as the polynomial argument. It can be
            provided as a HEX number (e.g., 0x0000).
    """

    polynomial: Incomplete
    initial_crc_value: Incomplete
    final_xor_value: Incomplete
    crc_byte_length: Incomplete
    crc_table: Incomplete
    standby: int
    calculate_checksum_buffer_datatype_error: int
    checksum_calculated: int
    checksum_converted_to_bytes: int
    convert_checksum_invalid_buffer_datatype_error: int
    convert_checksum_invalid_buffer_size_error: int
    checksum_converted_to_integer: int
    status: Incomplete
    def __init__(
        self,
        polynomial: np.uint8 | np.uint16 | np.uint32,
        initial_crc_value: np.uint8 | np.uint16 | np.uint32,
        final_xor_value: np.uint8 | np.uint16 | np.uint32,
    ) -> None: ...
    def calculate_crc_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Calculates the checksum for the input buffer.

        This method loops over the contents of the buffer and iteratively computes the CRC checksum for the entire
        buffer. Assumes that the buffer is entirely made up of the data to be checksummed.

        Notes:
            While error runtimes always return value 0, any 0-value returned by this method is potentially a valid
            value. To determine if the method runtime was successful, use 'status' class attribute. The returned value
            is not meaningful until it is verified using the status code!

        Args:
            buffer: The uint8 numpy array that stores the data to be checksummed.

        Returns:
            A numpy uint8, uint16, or uint32 integer that represents the calculated CRC checksum value. The returned
            data type depends on the polynomial used during class initialization. Also sets the 'status' class
            attribute to communicate the status of the method's runtime.
        """
    def convert_checksum_to_bytes(self, crc_checksum: np.uint8 | np.uint16 | np.uint32) -> NDArray[np.uint8]:
        """Converts the input checksum value into a numpy array of bytes.

        This method converts a multibyte CRC checksum into a sequence of individual bytes and writes them to a numpy
        uint8 array starting with the highest byte of the checksum.

        Returns:
            A uint8 numpy array entirely filled with the CRC checksum bytes. Also sets the 'status' class
            attribute to communicate method runtime status.
        """
    def convert_bytes_to_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Converts the CRC checksum stored in the input buffer as a series of bytes to an unsigned numpy integer value.

        This method is used to convert uint8 (byte) numpy arrays to crc checksum integer values. The method assumes
        that the checksum has been converted to bytes starting with the highest byte of the checksum and that the buffer
        is entirely filled with the checksum bytes.

        Notes:
            While error runtimes always return 0, any 0-value returned by this method is potentially a valid
            value. To determine if the method runtime was successful or failed, use 'status' class attribute.
            The returned value is not meaningful until it is verified using the status code!

        Returns:
            A numpy uint8, uint16, or uint32 integer that represents the converted CRC checksum value. The returned
            value data type depends on the polynomial datatype that was used during class initialization. Also sets
            the 'status' class attribute to communicate method runtime status.
        """
    def _generate_crc_table(self, polynomial: np.uint8 | np.uint16 | np.uint32) -> None:
        """Uses the polynomial specified during class instantiation to compute the CRC checksums for each
        possible uint8 (byte) value.

        The method updates the precompiled empty crc_table with polynomial-derived CRC values. This method is only
        intended to be called by the class initialization method. Do not use this method outside the class
        initialization context!

        Notes:
            While the PC is fast enough to work without a pregenerated table, this method is used to
            maintain algorithmic similarity to the version of the library used for Microcontrollers. Also, using
            a static table is still faster even for PCs.

        Args:
            polynomial: The polynomial to use for the generation of the CRC lookup table.
        """
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

class CRCProcessor:
    """Wraps a jit-compiled _CRCProcessor class that provides methods for working with CRC checksums used to verify
    the integrity of transferred data packets.

    This class functions as a wrapper that provides a consistent Python API for the internal instance of a
    jit-compiled _CRCProcessor class. This allows achieving python-like experience when using the class while
    simultaneously benefiting from fast compiled code generated through numba jit-optimization. The wrapper
    automatically converts internal class runtime status codes into exception error messages where appropriate to
    notify users about runtime errors.

    Notes:
        For the maximum execution speed, you can access the private methods directly via the 'processor' property,
        although this is highly discouraged.

        See the API documentation for the _CRCProcessor class for more details about CRC checksum generation and usage.

    Attributes:
        _processor: Stores the jit-compiled _CRCProcessor class, which carries out all computations.

    Args:
        polynomial: The polynomial used to generate the CRC lookup table. Can be provided as a HEX number
            (e.g., 0x1021). Currently only non-reversed polynomials of numpy uint8, uint16 and uint32 datatypes are
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

    _processor: Incomplete
    def __init__(
        self,
        polynomial: np.uint8 | np.uint16 | np.uint32,
        initial_crc_value: np.uint8 | np.uint16 | np.uint32,
        final_xor_value: np.uint8 | np.uint16 | np.uint32,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the CRCProcessor object."""
    def calculate_crc_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Calculates the CRC checksum for the data in the input buffer.

        Args:
            buffer: The uint8 numpy array that stores the data to be checksummed.

        Returns:
            A numpy uint8, uint16, or uint32 integer that stores the calculated CRC checksum value.

        Raises:
            TypeError: If the input buffer is not a numpy array.
            ValueError: If CRC checksum calculation fails for any reason.
        """
    def _resolve_checksum_calculation_status(self, buffer: NDArray[np.uint8]) -> None:
        """Resolves the status of the calculate_crc_checksum() method runtime.

        If the status was not successful, raises the appropriate error message.

        Args:
            buffer: The data buffer that was provided to the CRC checksum calculation method.

        Raises:
            ValueError: If input buffer parameters were not valid.
            RuntimeError: If the status code returned by the CRC calculator method is not one of the expected values.
        """
    def convert_checksum_to_bytes(self, crc_checksum: np.uint8 | np.uint16 | np.uint32) -> NDArray[np.uint8]:
        """Converts the input numpy scalar checksum into a byte numpy array.

        Returns:
            A uint8 numpy array entirely filled with the CRC checksum bytes.

        Raises:
            TypeError: If the input crc_checksum is not a numpy uint8, uint16, or uint32 integer.
            ValueError: If checksum conversion fails for any reason.
        """
    def _resolve_bytes_conversion_status(self) -> None:
        """Resolves the status of the convert_checksum_to_bytes() method runtime.

        If the status was not successful, raises the appropriate error message.

        Raises:
            RuntimeError: If the status code returned by the checksum converter method is not one of the expected
                values.
        """
    def convert_bytes_to_checksum(self, buffer: NDArray[np.uint8]) -> np.uint8 | np.uint16 | np.uint32:
        """Converts the input buffer that stores crc checksum bytes to an unsigned numpy integer checksum.

        Returns:
            A numpy uint8, uint16, or uint32 integer that stores the converted CRC checksum value.
            The returned value datatype depends on the polynomial datatype that was used during class initialization.

        Raises:
            TypeError: If the input buffer is not a numpy array.
            ValueError: If checksum conversion fails for any reason.
        """
    def _resolve_checksum_conversion_status(self, buffer: NDArray[np.uint8]) -> None:
        """Resolves the status of the convert_bytes_to_checksum() method runtime.

        If the status was not successful, raises the appropriate error message.

        Args:
            buffer: The buffer that was provided to the CRC checksum conversion method.

        Raises:
            ValueError: If input buffer parameters were not valid.
            RuntimeError: If the status code returned by the checksum converter method is not one of the expected
                values.
        """
    @property
    def crc_byte_length(self) -> np.uint8:
        """Returns the byte-size used by CRC checksums."""
    @property
    def crc_table(self) -> NDArray[np.uint8 | np.uint16 | np.uint32]:
        """Returns the CRC checksum lookup table."""
    @property
    def processor(self) -> _CRCProcessor:
        """Returns the jit-compiled _CRCProcessor class instance.

        This accessor represents a convenient way of unwrapping the jit-compiled class, so that its methods can be
        used directly. This is helpful for using them from other jit-methods or to bypass the overhead of error
        checking.
        """
    @property
    def polynomial(self) -> np.uint8 | np.uint16 | np.uint32:
        """Returns the polynomial used for checksum calculation."""
    @property
    def initial_crc_value(self) -> np.uint8 | np.uint16 | np.uint32:
        """Returns the initial value used for checksum calculation."""
    @property
    def final_xor_value(self) -> np.uint8 | np.uint16 | np.uint32:
        """Returns the final XOR value used for checksum calculation."""

class SerialMock:
    """Mocks the behavior of PySerial's `Serial` class for testing purposes.

    This class provides a mock implementation of the `Serial` class, enabling unit tests for TransportLayer class
    without requiring an actual hardware connection. It replicates the core functionalities of PySerial's `Serial`
    class that are relevant to testing, such as reading and writing data, while simplifying the overall behavior.

    Key differences from `Serial`:
        The `tx_buffer` and `rx_buffer` attributes are exposed directly, allowing test cases to verify the state of
        transmitted and received data. The class only supports methods used by `TransportLayer` for testing, and
        omits other methods not relevant to this specific use case.

    Attributes:
        is_open: Boolean flag indicating if the mock serial port is open.
        tx_buffer: Byte buffer that stores transmitted data.
        rx_buffer: Byte buffer that stores received data.
    """

    is_open: bool
    tx_buffer: bytes
    rx_buffer: bytes
    def __init__(self) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the SerialMock object."""
    def open(self) -> None:
        """Opens the mock serial port, setting `is_open` to True."""
    def close(self) -> None:
        """Closes the mock serial port, setting `is_open` to False."""
    def write(self, data: bytes) -> None:
        """Writes data to the `tx_buffer`.

        Args:
            data: Data to be written to the output buffer. Must be a bytes' object.

        Raises:
            TypeError: If `data` is not a bytes' object.
            Exception: If the mock serial port is not open.
        """
    def read(self, size: int = 1) -> bytes:
        """Reads a specified number of bytes from the `rx_buffer`.

        Args:
            size: Number of bytes to read from the input buffer. Defaults to 1.

        Returns:
            A bytes' object containing the requested data from the `rx_buffer`.

        Raises:
            Exception: If the mock serial port is not open.
        """
    def reset_input_buffer(self) -> None:
        """Clears the `rx_buffer`.

        Raises:
            Exception: If the mock serial port is not open.
        """
    def reset_output_buffer(self) -> None:
        """Clears the `tx_buffer`.

        Raises:
            Exception: If the mock serial port is not open.
        """
    @property
    def in_waiting(self) -> int:
        """Returns the number of bytes available in the `rx_buffer`."""
    @property
    def out_waiting(self) -> int:
        """Returns the number of bytes available in the `tx_buffer`."""
