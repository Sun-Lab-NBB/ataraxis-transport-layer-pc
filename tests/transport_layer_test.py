"""This file contains the test functions that verify the functionality and error-handling of all
TransportLayer class methods. Special care is taken to fully test the 4 major methods: write_data(),
read_data(), send_data(), and receive_data(). You can also use this file if you need more examples on how to use
class methods.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray
from ataraxis_base_utilities import error_format

from ataraxis_transport_layer.helper_modules import (
    SerialMock,
    CRCProcessor,
    COBSProcessor,
)
from ataraxis_transport_layer.transport_layer import SerialTransportLayer


@dataclass
class SampleDataClass:
    """A simple dataclass used to test 'structure' serialization capability of the SerialTransportLayer class. Has
     to use numpy arrays and scalars as field types to support serialization.

    Attributes:
        uint_value: Any numpy unsigned integer scalar value. Used to test the ability to serialize scalar dataclass
            fields.
        uint_array: Any numpy array value. Used to test the ability to serialize numpy array dataclass fields.
    """

    uint_value: np.unsignedinteger
    uint_array: np.ndarray


@pytest.fixture()
def protocol() -> SerialTransportLayer:
    """Returns a SerialTransportLayer instance with test mode enabled."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        test_mode=True,
    )

    return protocol


def test_init_and_repr():
    # Valid initialization
    protocol = SerialTransportLayer(port="COM3", baudrate=9600, test_mode=True)

    # Adjusted expected_repr with extra space for maximum_tx_payload_size
    representation_string = (
        f"SerialTransportLayer(port & baudrate=MOCKED, polynomial={protocol._crc_processor.polynomial}, "
        f"start_byte={protocol._start_byte}, delimiter_byte={protocol._delimiter_byte}, timeout={protocol._timeout} "
        f"us, maximum_tx_payload_size = {protocol._max_tx_payload_size}, "
        f"maximum_rx_payload_size={protocol._max_rx_payload_size})"
    )
    assert repr(protocol) == representation_string


def test_init_errors():
    port = None
    message = (
        f"Unable to initialize SerialTransportLayer class. Expected a string value for 'port' argument, but "
        f"encountered None of type NoneType."
    )
    with pytest.raises(
            TypeError,
            match=error_format(message),
    ):
        # noinspection PyTypeChecker
        SerialTransportLayer(
            port=port,  # This should trigger the TypeError
            baudrate=115200,
            start_byte=129,
            delimiter_byte=0,
            timeout=10000,
            test_mode=True,
        )

    # Invalid baudrate
    with pytest.raises(ValueError, match=r"Expected a positive integer value for 'baudrate'"):
        SerialTransportLayer(
            port="COM7",
            baudrate=-9600,  # Invalid baudrate
            start_byte=129,
            delimiter_byte=0,
            timeout=10000,
            test_mode=True,
        )

    # Invalid start_byte
    with pytest.raises(ValueError, match=r"Expected an integer value between 0 and 255 for 'start_byte'"):
        SerialTransportLayer(
            port="COM7",
            baudrate=115200,
            start_byte=300,  # Invalid start_byte
            delimiter_byte=0,
            timeout=10000,
            test_mode=True,
        )

    with pytest.raises(ValueError, match=r"Expected an integer value between 0 and 255 for 'delimiter_byte'"):
        SerialTransportLayer(
            port="COM7",
            baudrate=115200,
            start_byte=129,
            delimiter_byte=300,  # Invalid delimiter_byte
            timeout=10000,
            test_mode=True,
        )

    with pytest.raises(ValueError, match=r"Expected an integer value of 0 or above for 'timeout'"):
        SerialTransportLayer(
            port="COM7",
            baudrate=115200,
            start_byte=129,
            delimiter_byte=0,
            timeout=-5000,  # Invalid timeout
            test_mode=True,
        )

    with pytest.raises(ValueError, match=r"The 'start_byte' and 'delimiter_byte' cannot be the same"):
        SerialTransportLayer(
            port="COM7",
            baudrate=115200,
            start_byte=129,
            delimiter_byte=129,  # start_byte and delimiter_byte are the same
            timeout=10000,
            test_mode=True,
        )


def test_read_data_unsupported_input_type(protocol):
    """Test that an unsupported input type in read_data raises a TypeError with the correct message."""

    # Prepare an unsupported data_object (e.g., a string)
    unsupported_data_object = "unsupported_type"
    message = (
        f"Failed to read the data from the reception buffer. Encountered an unsupported input data_object "
        f"type ({type(unsupported_data_object).__name__}). At this time, only the following numpy scalar or array "
        f"types are supported: {protocol._accepted_numpy_scalars}. Alternatively, a dataclass with all attributes "
        f"set to supported numpy scalar or array types is also supported."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        # Directly raise TypeError if the input type is unsupported
        # noinspection PyTypeChecker
        protocol.read_data(data_object=unsupported_data_object)


def test_read_data_empty_array():
    """Test that attempting to read an empty array raises a ValueError and logs an error."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)
    # Prepare the reception buffer with some data, although it shouldn't matter
    protocol._reception_buffer[:10] = np.arange(10, dtype=np.uint8)
    protocol._bytes_in_reception_buffer = 10

    # Create an empty array to read into
    empty_array = np.empty(0, dtype=np.uint8)
    message = (
        "Failed to read the data from the reception buffer. Encountered an empty (size 0) numpy array as "
        "input data_object. Reading empty arrays is not supported."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        protocol.read_data(empty_array)


def test_full_data_flow():
    """Tests the full data flow: writing data, sending it, receiving it, and reading it back."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Scalar test
    test_scalar = np.uint8(42)
    protocol.write_data(test_scalar)
    send_status = protocol.send_data()
    assert send_status
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Loopback
    receive_status = protocol.receive_data()
    assert receive_status
    assert protocol.bytes_in_reception_buffer == test_scalar.nbytes
    received_scalar, end_index = protocol.read_data(np.uint8(0))
    assert end_index == test_scalar.nbytes
    assert received_scalar == test_scalar

    # Reset buffers before next test
    protocol.reset_transmission_buffer()
    protocol.reset_reception_buffer()
    protocol._port.tx_buffer = b""  # Clear explicitly
    protocol._port.rx_buffer = b""  # Clear explicitly

    # Array test
    test_array = np.array([1, 2, 3, 4], dtype=np.uint8)
    protocol.write_data(test_array)
    send_status = protocol.send_data()
    assert send_status
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Loopback
    receive_status = protocol.receive_data()
    assert receive_status

    print(f"Encoded tx_buffer content: {protocol._port.tx_buffer}")
    print(f"Reception buffer content: {protocol._reception_buffer[:protocol.bytes_in_reception_buffer]}")

    assert protocol.bytes_in_reception_buffer == test_array.nbytes
    received_array, end_index = protocol.read_data(np.zeros(len(test_array), dtype=np.uint8))
    assert end_index == test_array.nbytes
    assert np.array_equal(received_array, test_array)


def test_serial_transfer_protocol_buffer_manipulation():
    """Tests the functionality of the SerialTransportLayer class' write_data() and read_data() methods. This, by
    extension, also tests all internal private methods that enable the proper functioning of the main two methods. Also
    test buffer reset methods.
    """
    # Instantiates the tested SerialTransportLayer class
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        polynomial=np.uint16(0x1021),
        initial_crc_value=np.uint16(0xFFFF),
        final_crc_xor_value=np.uint16(0x0000),
        maximum_transmitted_payload_size=np.uint8(254),
        start_byte=np.uint8(129),
        delimiter_byte=np.uint8(0),
        timeout=np.uint64(20000),
        test_mode=True,  # Makes it port-agnostic by switching to SerialMock class as the 'serial interface'
    )

    # Verifies that the class initializes in test mode, which involves using SerialMock instead of the Serial class
    # from pySerial third-party library.
    # noinspection PyUnresolvedReferences
    assert isinstance(protocol._port, SerialMock)

    # Instantiate scalar objects for testing
    unsigned_8 = np.uint8(10)
    unsigned_16 = np.uint16(451)
    unsigned_32 = np.uint32(123456)
    signed_8 = np.int8(-10)
    signed_16 = np.int16(-451)
    signed_32 = np.int32(-123456)
    float_32 = np.float32(312.142)
    boolean_8 = np.bool_(True)

    # Tests Scalar object writing. Specifically, every supported scalar object from 1 to 4 bytes in size
    # Also tests automatic bytes-tracker-based start index calculation as no explicit start index is provided
    end_index = protocol.write_data(unsigned_8)
    assert end_index == 1
    end_index = protocol.write_data(unsigned_16)
    assert end_index == 3
    end_index = protocol.write_data(unsigned_32)
    assert end_index == 7
    end_index = protocol.write_data(signed_8)
    assert end_index == 8
    end_index = protocol.write_data(signed_16)
    assert end_index == 10
    end_index = protocol.write_data(signed_32)
    assert end_index == 14
    end_index = protocol.write_data(float_32)
    assert end_index == 18
    end_index = protocol.write_data(boolean_8)
    assert end_index == 19

    # Verifies that the bytes_in_transmission_buffer tracker matches the expected value (end_index) and that the
    # transmission buffer was indeed set to the expected byte values (matching written data).
    expected_buffer = np.array(
        [
            10,
            195,
            1,
            64,
            226,
            1,
            0,
            246,
            61,
            254,
            192,
            29,
            254,
            255,
            45,
            18,
            156,
            67,
            1,
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(expected_buffer, protocol.transmission_buffer[:end_index])
    assert protocol.bytes_in_transmission_buffer == 19

    # Instantiate numpy array objects for testing
    unsigned_array_64 = np.array([1, 2, 3, 4, 5], dtype=np.uint64)
    signed_array_64 = np.array([-1, -2, -3, -4, -5], dtype=np.int64)
    float_array_64 = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)

    # Test array object writing for all supported array types (8 bytes in size)
    end_index = protocol.write_data(unsigned_array_64)
    assert end_index == 59
    end_index = protocol.write_data(signed_array_64)
    assert end_index == 99
    end_index = protocol.write_data(float_array_64)
    assert end_index == 139

    # Modifies the expected buffer to account for the newly added data (120 newly added bytes) so that the buffer state
    # can be verified against the expected values. Then verifies the _transmission_buffer actually looks as expected
    # and that the bytes' tracker was updated accordingly.
    next_expected_array = np.array(
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            254,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            253,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            252,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            251,
            255,
            255,
            255,
            255,
            255,
            255,
            255,
            154,
            153,
            153,
            153,
            153,
            153,
            241,
            63,
            154,
            153,
            153,
            153,
            153,
            153,
            1,
            64,
            102,
            102,
            102,
            102,
            102,
            102,
            10,
            64,
            154,
            153,
            153,
            153,
            153,
            153,
            17,
            64,
            0,
            0,
            0,
            0,
            0,
            0,
            22,
            64,
        ],
        dtype=np.uint8,
    )
    expected_buffer = np.concatenate((expected_buffer, next_expected_array))
    assert np.array_equal(expected_buffer, protocol.transmission_buffer[:end_index])
    assert protocol.bytes_in_transmission_buffer == 139

    # Instantiate a test dataclass and simulate 'structure' writing
    test_class = SampleDataClass(uint_value=np.uint8(50), uint_array=np.array([1, 2, 3], np.uint8))

    # Write the test class to the protocol buffer, overwriting the beginning of the buffer
    end_index = protocol.write_data(test_class, start_index=0)
    assert end_index == 4  # Ensures the returned index matches the expectation and is different from payload size
    assert protocol.bytes_in_transmission_buffer == 139

    # Verify that the data inside the buffer was overwritten as expected
    expected_buffer[0:4] = [50, 1, 2, 3]
    assert np.array_equal(
        expected_buffer,
        protocol.transmission_buffer[: protocol.bytes_in_transmission_buffer],
    )

    # Restore the initial portion of the buffer and re-write the dataclass to the end of the payload for reading tests
    end_index = protocol.write_data(unsigned_8, start_index=0)
    end_index = protocol.write_data(unsigned_16, start_index=end_index)
    protocol.write_data(unsigned_32, start_index=end_index)
    end_index = protocol.write_data(test_class)
    assert end_index == 143  # Ensures the data was written

    # Record the final state of the transmission buffer to support reset method tests later
    expected_buffer = protocol.transmission_buffer

    # Copies the contents of the transmission buffer into the reception buffer to test data reading. Since there is no
    # exposed mechanism for doing so, directly accesses private attributes.
    # noinspection PyUnresolvedReferences
    protocol._reception_buffer = protocol.transmission_buffer

    # Also transfers the payload size from the transmission to the reception tracker, as this is necessary for the
    # data to be readable (the tracker is typically set by the receive_data() method during data reception). Same as
    # above, directly overwrites the private class attribute.
    # noinspection PyUnresolvedReferences
    protocol._bytes_in_reception_buffer = protocol.bytes_in_transmission_buffer

    # Verifies reading scalar values from the buffer works as expected. Provides zero-initialized prototype objects
    # to the read function and expects returned object values to match those used for writing
    unsigned_8_read, end_index = protocol.read_data(np.uint8(0))
    assert unsigned_8_read == unsigned_8
    assert end_index == 1
    unsigned_16_read, end_index = protocol.read_data(np.uint16(0), start_index=end_index)
    assert unsigned_16_read == unsigned_16
    assert end_index == 3
    unsigned_32_read, end_index = protocol.read_data(np.uint32(0), start_index=end_index)
    assert unsigned_32_read == unsigned_32
    assert end_index == 7
    signed_8_read, end_index = protocol.read_data(np.int8(0), start_index=end_index)
    assert signed_8_read == signed_8
    assert end_index == 8
    signed_16_read, end_index = protocol.read_data(np.int16(0), start_index=end_index)
    assert signed_16_read == signed_16
    assert end_index == 10
    signed_32_read, end_index = protocol.read_data(np.int32(0), start_index=end_index)
    assert signed_32_read == signed_32
    assert end_index == 14
    float_32_read, end_index = protocol.read_data(np.float32(0), start_index=end_index)
    assert float_32_read == float_32
    assert end_index == 18
    boolean_8_read, end_index = protocol.read_data(np.bool_(False), start_index=end_index)
    assert boolean_8_read == boolean_8
    assert end_index == 19

    # Ensures the read operation does not affect the reception buffer payload tracker, as read operations are not meant
    # to write to the tracker variable
    assert protocol.bytes_in_reception_buffer == 143

    # Verifies that reading array data from buffer works as expected
    unsigned_array_64_read, end_index = protocol.read_data(np.zeros(5, dtype=np.uint64), start_index=end_index)
    assert np.array_equal(unsigned_array_64_read, unsigned_array_64)
    assert end_index == 59
    signed_array_64_read, end_index = protocol.read_data(np.zeros(5, dtype=np.int64), start_index=end_index)
    assert np.array_equal(signed_array_64_read, signed_array_64)
    assert end_index == 99
    float_array_64_read, end_index = protocol.read_data(np.zeros(5, dtype=np.float64), start_index=end_index)
    assert np.array_equal(float_array_64_read, float_array_64)
    assert end_index == 139

    # Verifies that reading dataclasses works as expected
    test_class_read: SampleDataClass = SampleDataClass(uint_value=np.uint8(0), uint_array=np.zeros(3, dtype=np.uint8))
    # noinspection PyTypeChecker
    test_class_read, end_index = protocol.read_data(test_class_read, start_index=end_index)
    assert test_class_read.uint_value == test_class.uint_value
    assert np.array_equal(test_class_read.uint_array, test_class.uint_array)
    assert end_index == 143

    # Tests that calling reset_transmission_buffer() method correctly resets the transmission bytes tracker, but does
    # not alter the buffer itself.
    assert protocol.bytes_in_transmission_buffer == 143
    protocol.reset_transmission_buffer()
    assert protocol.bytes_in_transmission_buffer == 0
    assert np.array_equal(expected_buffer, protocol.transmission_buffer)

    # Tests that calling reset_reception_buffer() method correctly resets the reception bytes tracker, but does
    # not alter the buffer itself.
    assert protocol.bytes_in_reception_buffer == 143
    protocol.reset_reception_buffer()
    assert protocol.bytes_in_reception_buffer == 0
    assert np.array_equal(expected_buffer, protocol.reception_buffer)


def test_write_data_unsupported_input_type_error(protocol):
    """Test that a TypeError is raised for unsupported input types."""

    # Create an unsupported input type (neither numpy scalar nor dataclass)
    invalid_data = None
    message = (
        f"Failed to write the data to the transmission buffer. Encountered an unsupported input data_object "
        f"type ({type(invalid_data).__name__}). At this time, only the following numpy scalar or array "
        f"types are supported: {protocol._accepted_numpy_scalars}. Alternatively, a dataclass with all attributes "
        f"set to supported numpy scalar or array types is also supported."
    )
    with pytest.raises(TypeError, match=error_format(message)):
        protocol.write_data(invalid_data)


def test_write_data_empty_array_error(protocol):
    """Test that attempting to write an empty array raises a ValueError."""
    message = (
        "Failed to write the data to the transmission buffer. Encountered an empty (size 0) numpy array as input "
        "data_object. Writing empty arrays is not supported."
    )
    empty_array: NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
    with pytest.raises(
        ValueError,
        match=error_format(message),
    ):
        # noinspection PyTypeChecker
        protocol.write_data(empty_array)


def test_write_data_non_multidimensional_array_error(protocol):
    """Test that attempting to write a multidimensional array raises a ValueError."""
    message = (
        "Failed to write the data to the transmission buffer. Encountered a multidimensional numpy array with 2 "
        "dimensions as input data_object. At this time, only one-dimensional (flat) arrays are supported."
    )
    invalid_array: np.ndarray = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.write_data(invalid_array)


def test_write_data_insufficient_buffer_size_error(protocol):
    """Test that a ValueError is raised when the buffer does not have enough space."""

    # Reduces the size of the transmission buffer to simulate the error
    protocol._transmission_buffer = np.zeros(10, dtype=np.uint8)
    protocol._bytes_in_transmission_buffer = 8  # Only 2 bytes of space left

    # Create a large numpy array that exceeds the buffer size
    large_data = np.ones(5, dtype=np.uint8)
    message = (
        f"Failed to write the data to the transmission buffer. The transmission buffer does not have enough "
        f"space to write the data starting at the index {7}. Specifically, given the data size of "
        f"{large_data.nbytes} bytes, the required buffer size is {7 + large_data.nbytes} bytes, but the available "
        f"size is {protocol._transmission_buffer.size} bytes."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        # noinspection PyTypeChecker
        protocol.write_data(large_data, start_index=7)


def test_serial_transfer_protocol_data_transmission():
    """Tests the send_data() and receive_data() methods of the SerialTransportLayer class."""
    # Initialize the tested class
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        polynomial=np.uint16(0x1021),
        initial_crc_value=np.uint16(0xFFFF),
        final_crc_xor_value=np.uint16(0x0000),
        maximum_transmitted_payload_size=254,
        start_byte=129,
        delimiter_byte=0,
        timeout=20000,
        test_mode=True,
        allow_start_byte_errors=False,
    )

    cobs_processor = COBSProcessor()
    crc_processor = CRCProcessor(
        polynomial=np.uint16(0x1021),
        initial_crc_value=np.uint16(0xFFFF),
        final_xor_value=np.uint16(0x0000),
    )

    test_array = np.array([1, 2, 3, 0, 0, 6, 0, 8, 0, 0], dtype=np.uint8)
    protocol.write_data(test_array)
    assert protocol.bytes_in_transmission_buffer == test_array.nbytes

    send_status = protocol.send_data()
    assert send_status
    assert protocol.bytes_in_transmission_buffer == 0

    # Encode payload and calculate checksum
    encoded_payload = cobs_processor.encode_payload(test_array, delimiter=np.uint8(0))
    checksum = crc_processor.calculate_crc_checksum(encoded_payload)
    checksum_bytes = crc_processor.convert_checksum_to_bytes(checksum)

    # Align payload size calculation with `send_data` logic
    payload_size = np.uint8(len(test_array))  # Adjust based on actual logic in `send_data`
    expected_packet = np.concatenate(([payload_size], encoded_payload, checksum_bytes))

    tx_buffer = np.frombuffer(protocol._port.tx_buffer, dtype=np.uint8)

    print("TX Buffer:", tx_buffer)
    print("Expected Packet:", expected_packet)

    # Assert start byte and packet contents
    assert tx_buffer[0] == 129  # Start byte
    assert np.array_equal(tx_buffer[1 : 1 + expected_packet.size], expected_packet)


def test_receive_data_errors(protocol):
    """Tests SerialTransportLayer class send_data() and receive_data() method error handling."""
    test_payload: NDArray[np.uint8] = np.array([1, 2, 3, 4, 0, 0, 7, 8, 9, 10], dtype=np.uint8)
    # noinspection PyTypeChecker
    protocol.write_data(test_payload)
    protocol.send_data()

    preamble = np.array([129, 10], dtype=np.uint8)
    packet = protocol._cobs_processor.encode_payload(payload=test_payload, delimiter=np.uint8(0))
    checksum = protocol._crc_processor.calculate_crc_checksum(packet)
    checksum = protocol._crc_processor.convert_checksum_to_bytes(checksum)
    test_data = np.concatenate((preamble, packet, checksum), axis=0)
    empty_buffer = np.zeros(20, dtype=np.uint8)

    # CODE 101
    # Test case for noise buffer with no start byte
    protocol._port.rx_buffer = empty_buffer.tobytes()
    assert not protocol.receive_data()

    # CODE 102
    # Tests that receiving an empty message with start byte errors turned on raises a RuntimeError
    protocol._allow_start_byte_errors = True
    protocol._port.rx_buffer = empty_buffer.tobytes()
    message = (
        "Failed to parse the incoming serial packet data. Unable to find the start_byte "
        "(129) value among the bytes stored inside the serial buffer."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol._leftover_bytes = bytes()  # Clears leftover bytes aggregator
        protocol.receive_data()

    # CODE 0
    # Test case for missing packet size after start byte
    empty_buffer[-1] = 129
    protocol._port.rx_buffer = empty_buffer.tobytes()
    message = (
        f"Failed to parse the size of the incoming serial packet. The packet size byte was not received in "
        f"time ({protocol._timeout} microseconds), following the reception of the START byte."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol._leftover_bytes = bytes()  # Clears leftover bytes aggregator
        protocol.receive_data()

    # CODE 2
    # Test case for packet staling
    test_data[1] = 110
    test_data[13] = 1  # Replaces the original delimiter byte to avoid Delimiter Byte Found Too Early error
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        "Failed to parse the incoming serial packet data. The byte number 15 out of 114 "
        "was not received in time (20000 microseconds), following the reception of the previous byte. "
        "Packet reception staled."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol._leftover_bytes = bytes()  # Clears leftover bytes aggregator
        protocol.receive_data()
    test_data[1] = 10  # Restores the packet size to the expected number
    test_data[13] = 0  # Restores the delimiter byte

    # CODE 103
    # Message contains an invalid payload size
    test_data[1] = 255  # Replaces the original delimiter byte to avoid Delimiter Byte Found Too Early error
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to parse the incoming serial packet data. The parsed size of the COBS-encoded payload "
        f"(255), is outside the expected boundaries ({protocol._min_rx_payload_size} to "
        f"{protocol._max_rx_payload_size}). This likely indicates a mismatch in the transmission parameters between "
        f"this system and the Microcontroller."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol._leftover_bytes = bytes()  # Clears leftover bytes aggregator
        protocol.receive_data()
    # Restores the correct payload size
    test_data[1] = 10

    # CODE 104
    # Delimiter found too early
    save = int(test_data[-4])
    test_data[-4] = 0  # Inserts the delimiter 1 position before the actual delimiter position
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to parse the incoming serial packet data. Delimiter byte value "
        f"({protocol._delimiter_byte}) encountered at byte number {14}, instead of the "
        f"expected byte number {17}. This likely indicates "
        f"packet corruption or mismatch in the transmission parameters between this system "
        f"and the Microcontroller."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol._leftover_bytes = bytes()  # Clears leftover bytes aggregator
        protocol.receive_data()
    test_data[-4] = save  # Restores the replaced value

    # CODE 105
    # Delimiter not found
    test_data[-3] = 10  # Overrides the delimiter
    protocol._port.rx_buffer = test_data.tobytes()
    message = (
        f"Failed to parse the incoming serial packet data. Delimiter byte value "
        f"({protocol._delimiter_byte}) expected as the last payload byte (17), but instead encountered "
        f"{159}. This likely indicates packet corruption or mismatch in the transmission parameters between this system "
        f"and the Microcontroller."
    )
    with pytest.raises(
        RuntimeError,
        match=error_format(message),
    ):
        protocol._leftover_bytes = bytes()  # Clears leftover bytes aggregator
        protocol.receive_data()
    test_data[-3] = 0  # Restores the delimiter


def test_receive_data_crc_verification_error(protocol):
    """Test that a CRC checksum mismatch triggers an appropriate error message."""
    # Creates, encodes, and 'receives' the test payload
    test_message = np.zeros(shape=1, dtype=np.uint8)
    # noinspection PyTypeChecker
    protocol.write_data(test_message)
    protocol.send_data()
    protocol._port.rx_buffer = protocol._port.tx_buffer

    # Replaces the valid checksum with an invalid placeholder to simulate the error
    encoded_packet = np.frombuffer(buffer=protocol._port.rx_buffer, dtype=np.uint8).copy()
    expected_checksum = encoded_packet[-2:].copy()
    encoded_packet[-2:] = np.array([0x00, 0x00], dtype=np.uint8)  # Fake checksum
    protocol._port.rx_buffer = encoded_packet.tobytes()
    message = (
        f"Failed to verify the received serial packet's integrity. The checksum value transmitted with the packet "
        f"{hex(protocol._crc_processor.convert_bytes_to_checksum(encoded_packet[-2:].copy()))} did not match the "
        f"expected value based on the packet data "
        f"{hex(protocol._crc_processor.convert_bytes_to_checksum(expected_checksum))}. This indicates the packet was "
        f"corrupted during transmission or reception."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        protocol.receive_data()


def test_write_data_buffer_tracker_update(protocol):
    """Test that the buffer tracker (_bytes_in_transmission_buffer) is updated correctly after a successful write()
    call."""

    # Mock the transmission buffer with enough space
    protocol._transmission_buffer = np.zeros(20, dtype=np.uint8)
    protocol._bytes_in_transmission_buffer = 10

    # Create a numpy scalar and write it to the buffer
    data = np.uint8(123)
    end_index = protocol.write_data(data_object=data, start_index=10)

    # Ensure the tracker is updated correctly
    assert protocol._bytes_in_transmission_buffer == end_index


def test_send_data_empty_buffer_error(protocol):
    """Test that send_data returns False when transmission buffer is empty."""

    # Ensures the buffer is empty
    assert protocol.bytes_in_transmission_buffer == 0

    # Attempt to send data
    message = (
        f"Failed to encode the payload using COBS scheme. The size of the input payload "
        f"({0}) is too small. A minimum size of {protocol._cobs_processor._processor.minimum_payload_size} elements "
        f"(bytes) is required. CODE: 12."
    )
    with pytest.raises(
        ValueError,
        match=error_format(message),
    ):
        send_status = protocol.send_data()

        # Expect send_data to return False
        assert not send_status
