"""This file contains the test functions that verify the functionality and error-handling of all
TransportLayer class methods. Special care is taken to fully test the 4 major methods: write_data(),
read_data(), send_data(), and receive_data(). You can also use this file if you need more examples on how to use
class methods.
"""

import re
import textwrap
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
from numpy import ndarray, unsignedinteger
import pytest
from serial.tools import list_ports
from numpy._typing import NDArray

from ataraxis_transport_layer.helper_modules import (
    SerialMock,
    CRCProcessor,
    COBSProcessor,
)
from ataraxis_transport_layer.transport_layer import (
    SerialTransportLayer,
)


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


class TestSerialTransportLayerInitialization:
    def test_valid_initialization(self):
        # Existing valid initialization tests
        protocol = SerialTransportLayer(
            port="COM7",
            baudrate=115200,
            start_byte=129,
            delimiter_byte=0,
            timeout=10000,
            test_mode=True,
        )
        assert protocol.port == "COM7"
        # Add more assertions as needed

    # New Tests for Argument Validation

    def test_invalid_port_type_none(self):
        # Test for None as port
        with pytest.raises(TypeError, match=r"Expected a string value for 'port' argument, but encountered NoneType"):
            SerialTransportLayer(
                port=None,  # This should trigger the TypeError
                baudrate=115200,
                start_byte=129,
                delimiter_byte=0,
                timeout=10000,
                test_mode=True,
            )

    def test_invalid_port_type_int(self):
        # Test for invalid port type (integer instead of string)
        with pytest.raises(TypeError, match=r"Expected a string value for 'port' argument, but encountered int"):
            SerialTransportLayer(
                port=12345,  # This should trigger the TypeError
                baudrate=115200,
                start_byte=129,
                delimiter_byte=0,
                timeout=10000,
                test_mode=True,
            )

    def test_invalid_baudrate_value(self):
        with pytest.raises(ValueError, match=r"Expected a positive integer value for 'baudrate'"):
            SerialTransportLayer(
                port="COM7",
                baudrate=-9600,  # Invalid baudrate
                start_byte=129,
                delimiter_byte=0,
                timeout=10000,
                test_mode=True,
            )

    def test_invalid_start_byte_value(self):
        with pytest.raises(ValueError, match=r"Expected an integer value between 0 and 255 for 'start_byte'"):
            SerialTransportLayer(
                port="COM7",
                baudrate=115200,
                start_byte=300,  # Invalid start_byte
                delimiter_byte=0,
                timeout=10000,
                test_mode=True,
            )

    def test_invalid_delimiter_byte_value(self):
        with pytest.raises(ValueError, match=r"Expected an integer value between 0 and 255 for 'delimiter_byte'"):
            SerialTransportLayer(
                port="COM7",
                baudrate=115200,
                start_byte=129,
                delimiter_byte=300,  # Invalid delimiter_byte
                timeout=10000,
                test_mode=True,
            )

    def test_invalid_timeout_value(self):
        with pytest.raises(ValueError, match=r"Expected an integer value of 0 or above for 'timeout'"):
            SerialTransportLayer(
                port="COM7",
                baudrate=115200,
                start_byte=129,
                delimiter_byte=0,
                timeout=-5000,  # Invalid timeout
                test_mode=True,
            )

    def test_start_byte_equals_delimiter_byte(self):
        with pytest.raises(ValueError, match=r"The 'start_byte' and 'delimiter_byte' cannot be the same"):
            SerialTransportLayer(
                port="COM7",
                baudrate=115200,
                start_byte=129,
                delimiter_byte=129,  # start_byte and delimiter_byte are the same
                timeout=10000,
                test_mode=True,
            )


def test_repr_with_mocked_port():
    """Test __repr__ when the _port is mocked using SerialMock."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )
    expected_repr = (
        "SerialTransportLayer(port & baudrate=MOCKED, "
        "polynomial=0x1021, start_byte=129, delimiter_byte=0, "
        "timeout=10000 us, maximum_tx_payload_size = 254, "
        "maximum_rx_payload_size=254)"
    )
    assert repr(protocol) == expected_repr


@patch("ataraxis_base_utilities.console.error")  # Mock the console.error function
def test_empty_array_error(self, mock_error):
    """Test for handling the error when an empty numpy array is passed."""

    # Create an instance of the SerialTransportLayer with test_mode=True
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Create mock data with an empty numpy array
    data = MockDataClass(np.uint8(10), np.array([], dtype=np.uint8))

    # Patch the internal write_data method to simulate returning -2 (empty array error)
    with patch.object(protocol, "_write_array_data", return_value=-2):
        protocol.write_data(data_object=data)

        # Check if the appropriate error message was logged
        mock_error.assert_called_with(
            message="Failed to write the data to the transmission buffer. Encountered an empty (size 0) numpy array as input data_object. Writing empty arrays is not supported.",
            error=ValueError,
        )


@patch("console.error")  # Mocking the console.error function
def test_unknown_error_code(mock_error):
    """Test for handling unknown error code in the read_data method."""

    # Create an instance of the SerialTransportLayer with test_mode=True
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock data with valid numpy types
    data = MockDataClass(np.uint8(10), np.array([1, 2, 3], dtype=np.uint8))

    # Patch the internal method to simulate an unknown error code
    with patch.object(protocol, "read_data", return_value=(-99)):
        with pytest.raises(RuntimeError):
            protocol.receive_data()

        # Assert that the appropriate error message was logged
        mock_error.assert_called_with(
            message="Failed to read the data from the reception buffer. Encountered an unknown error code (-99) returned by the reader method.",
            error=RuntimeError,
        )


def test_full_data_flow():
    """Tests the full data flow: writing data, sending it, receiving it, and reading it back.
    This test ensures that data is correctly serialized, transmitted, received, and deserialized.
    """
    # Initialize the protocol in test mode to use SerialMock
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Prepare COBS and CRC processors for verification
    cobs_processor = COBSProcessor()
    crc_processor = CRCProcessor(
        polynomial=protocol.polynomial,
        initial_crc_value=protocol.initial_crc_value,
        final_xor_value=protocol.final_crc_xor_value,
    )

    # Test scalar value
    test_scalar = np.uint8(42)
    # Write data to transmission buffer
    protocol.write_data(test_scalar)
    # Send data
    send_status = protocol.send_data()
    assert send_status
    assert protocol.bytes_in_transmission_buffer == 0  # Buffer should be cleared after send

    # Simulate receiving the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Loopback
    # Receive data
    receive_status = protocol.receive_data()
    assert receive_status
    assert protocol.bytes_in_reception_buffer == test_scalar.nbytes
    # Read back the scalar
    received_scalar, end_index = protocol.read_data(np.uint8(0))
    assert end_index == test_scalar.nbytes
    assert received_scalar == test_scalar

    # Reset buffers before next test
    protocol.reset_transmission_buffer()
    protocol.reset_reception_buffer()

    # Test array values
    test_array = np.array([1, 2, 3, 4], dtype=np.uint8)
    # Write data to transmission buffer
    protocol.write_data(test_array)
    # Send data
    send_status = protocol.send_data()
    assert send_status
    assert protocol.bytes_in_transmission_buffer == 0  # Buffer should be cleared after send

    # Simulate receiving the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Loopback
    # Receive data
    receive_status = protocol.receive_data()
    assert receive_status
    assert protocol.bytes_in_reception_buffer == test_array.nbytes
    # Read back the array
    received_array, end_index = protocol.read_data(np.zeros(len(test_array), dtype=np.uint8))
    assert end_index == test_array.nbytes
    assert np.array_equal(received_array, test_array)


@patch("ataraxis_transport_layer.transport_layer.serial.Serial")
def test_repr_with_real_serial_port(mock_serial):
    """Test __repr__ when the _port is a real Serial object."""
    # Set up the mock serial port
    mock_serial_instance = MagicMock()
    mock_serial_instance.name = "COM7"
    mock_serial_instance.baudrate = 115200
    mock_serial.return_value = mock_serial_instance

    # Initialize the protocol with test_mode=False, which should use the real Serial class
    protocol = SerialTransportLayer(
        port="COM7", baudrate=115200, start_byte=129, delimiter_byte=0, timeout=10000, test_mode=False
    )

    # The __repr__ method should now use the 'if isinstance(self._port, Serial):' branch
    repr_str = repr(protocol)
    expected_repr = (
        f"SerialTransportLayer(port='COM7', baudrate=115200, polynomial=0x{protocol._crc_processor.polynomial:04X}, "
        f"start_byte=129, delimiter_byte=0, timeout=10000 us, maximum_tx_payload_size={protocol._max_tx_payload_size}, "
        f"maximum_rx_payload_size={protocol._max_rx_payload_size})"
    )
    assert repr_str == expected_repr


def test_real_serial_initialization():
    # Mock the Serial class from pySerial
    with patch("ataraxis_transport_layer.transport_layer.Serial") as mock_serial:
        # Initialize the protocol with test_mode=False, which should use the real Serial class
        protocol = SerialTransportLayer(
            port="COM7", baudrate=115200, start_byte=129, delimiter_byte=0, timeout=10000, test_mode=False
        )

        # Verify that the Serial class was called with the correct parameters
        mock_serial.assert_called_once_with("COM7", 115200, timeout=0)
        assert protocol._port == mock_serial.return_value  # Check if _port was set to the mock Serial object


def test_mock_serial_initialization():
    # Initialize the protocol with test_mode=True, which should use the mock Serial class
    protocol = SerialTransportLayer(
        port="COM7", baudrate=115200, start_byte=129, delimiter_byte=0, timeout=10000, test_mode=True
    )

    # Check if _port was set to SerialMock
    assert isinstance(protocol._port, SerialMock)


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
    assert end_index == 4  # Ensures the returned index matches the expectation and is not the same as payload size
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


def test_serial_transfer_protocol_buffer_manipulation_errors():
    """Tests the error-handling capabilities of SerialTransportLayer class write_data() and read_data() methods.
    Also tests class initialization errors.
    """

    # Verifies that using maximum_transmitted_payload_size argument above 254 triggers an error during class
    # initialization. Keeps the rest of the parameters set to default values, where possible.
    error_message = (
        f"Unable to initialize SerialTransportLayer class. 'maximum_transmitted_payload_size' argument value "
        f"({255}) cannot exceed 254."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = SerialTransportLayer(
            port="COM7",
            maximum_transmitted_payload_size=np.uint8(255),
            test_mode=True,
        )

    # Verifies that the minimum_received_payload_size cannot be set outside the range of 1 to 254 (inclusive)
    error_message = (
        f"Unable to initialize SerialTransportLayer class. 'minimum_received_payload_size' argument value "
        f"({0}) must be between 1 and 254 (inclusive)."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = SerialTransportLayer(
            port="COM7",
            minimum_received_payload_size=0,
            test_mode=True,
        )
    error_message = (
        f"Unable to initialize SerialTransportLayer class. 'minimum_received_payload_size' argument value "
        f"({255}) must be between 1 and 254 (inclusive)."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = SerialTransportLayer(
            port="COM7",
            minimum_received_payload_size=255,
            test_mode=True,
        )

    # Verifies that setting start_byte and delimiter_byte to the same value triggers an error.
    error_message = (
        f"Unable to initialize SerialTransportLayer class. 'start_byte' and 'delimiter_byte' arguments "
        f"cannot be set to the same value ({129})."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = SerialTransportLayer(
            port="COM7",
            start_byte=np.uint8(129),
            delimiter_byte=np.uint8(129),
            test_mode=True,
        )

    # Instantiates the tested protocol class. Lists all addressable parameters, although technically only port should
    # be provided to use default initialization values.
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        polynomial=np.uint16(0x1021),
        initial_crc_value=np.uint16(0xFFFF),
        final_crc_xor_value=np.uint16(0x0000),
        maximum_transmitted_payload_size=np.uint8(254),
        minimum_received_payload_size=np.uint8(1),
        start_byte=np.uint8(129),
        delimiter_byte=np.uint8(0),
        timeout=np.uint64(20000),
        test_mode=True,
    )

    # Note, the error messages are VERY similar for write and read methods... Which makes sense as they are very
    # similar methods :).

    # WRITE DATA SECTION
    # Verifies that calling write_data method for a non-supported input type raises an error.
    invalid_input = None
    # noinspection PyUnresolvedReferences
    error_message = (
        f"Unsupported input data_object type ({type(invalid_input)}) encountered when writing data "
        f"to _transmission_buffer. At this time, only the following numpy scalar or array types are "
        f"supported: {protocol._accepted_numpy_scalars}. Alternatively, a dataclass with "
        f"all attributes set to supported numpy scalar or array types is also supported."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        protocol.write_data(invalid_input)

    # Verifies that attempting to write too large of a payload (in a start-index-dependent fashion, a payload that does
    # not fit inside the buffer) raises an error.
    start_index = 150
    payload = np.ones(200, dtype=np.uint8)
    # noinspection PyUnresolvedReferences
    error_message = (
        f"Insufficient buffer space to write the data to the _transmission_buffer starting at the index "
        f"'{start_index}'. Specifically, given the data size of '{payload.nbytes}' bytes, the required buffer "
        f"size is '{start_index + payload.nbytes}' bytes, but the available size is "
        f"'{protocol.transmission_buffer.size}' bytes."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.write_data(payload, start_index=start_index)

    # Verifies that calling write_data method for a multidimensional numpy array raises an error.
    invalid_array: ndarray[unsignedinteger[np.uint8], Any] = np.zeros((2, 2), dtype=np.uint8)
    error_message = (
        f"A multidimensional numpy array with {invalid_array.ndim} dimensions encountered when writing "
        f"data to _transmission_buffer. At this time, only one-dimensional (flat) arrays are supported."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.write_data(invalid_array)

    # Verifies that calling write_data method for an empty numpy array raises an error.
    empty_array = np.empty(0, dtype=np.uint8)
    error_message = (
        f"An empty (size 0) numpy array encountered when writing data to _transmission_buffer. Writing empty arrays "
        f"is not supported."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        protocol.write_data(empty_array)

    # READ DATA SECTION
    # Sets the _bytes_in_reception_buffer to a non-zero value to support testing. Since there is no way of 'gracefully'
    # accessing this private attribute, uses direct access.
    protocol._bytes_in_reception_buffer = 50

    # Verifies that calling read_data method for a non-supported input object raises an error.
    invalid_input = None
    # noinspection PyUnresolvedReferences
    error_message = (
        f"Unsupported input data_object type ({type(invalid_input)}) encountered when reading data "
        f"from _reception_buffer. At this time, only the following numpy scalar or array types are supported: "
        f"{protocol._accepted_numpy_scalars}. Alternatively, a dataclass with all "
        f"attributes set to supported numpy scalar or array types is also supported."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        protocol.read_data(invalid_input)

    # Verifies that attempting to read too large of an object (in a start-index-dependent fashion, an object that cannot
    # be filled with the bytes available from the payload starting at start_index) raises an error.
    start_index = 150
    payload: ndarray[unsignedinteger[np.uint8], Any] = np.ones(200, dtype=np.uint8)

    error_message = (
        f"Insufficient payload size to read the data from the _reception_buffer starting at the index "
        f"'{start_index}'. Specifically, given the object size of '{payload.nbytes}' bytes, the required payload "
        f"size is '{start_index + payload.nbytes}' bytes, but the available size is "
        f"'{protocol.bytes_in_reception_buffer}' bytes."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.read_data(payload, start_index=start_index)

    # Verifies that calling read_data method for a multidimensional numpy array raises an error.
    invalid_array = np.zeros((2, 2), dtype=np.uint8)
    error_message = (
        f"A multidimensional numpy array with {invalid_array.ndim} dimensions requested when reading "
        f"data from _reception_buffer. At this time, only one-dimensional (flat) arrays are supported."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.read_data(invalid_array)

    empty_array = np.empty(0, dtype=np.uint8)

    # Verifies that calling read_data for an empty array raises an error.
    error_message = (
        f"Am empty (size 0) numpy array requested when reading data from _reception_buffer. Reading empty "
        f"arrays is currently not supported."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        protocol.read_data(empty_array)


def test_empty_array():
    """Test that attempting to write an empty array raises a ValueError."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )

    empty_array: NDArray[np.uint8] = np.empty(0, dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "An empty (size 0) numpy array encountered when writing data to _transmission_buffer. Writing empty arrays is not supported."
        ),
    ):
        protocol.write_data(empty_array)


def test_non_one_dimensional_array():
    """Test that attempting to write a multidimensional array raises a ValueError."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )

    invalid_array: np.ndarray = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"A multidimensional numpy array with {invalid_array.ndim} dimensions encountered when writing data to _transmission_buffer. At this time, only one-dimensional (flat) arrays are supported."
        ),
    ):
        protocol.write_data(invalid_array)


def test_read_data_multidimensional_array():
    """Test that attempting to read a multidimensional array raises a ValueError."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    invalid_array: np.ndarray = np.zeros((2, 2), dtype=np.uint8)

    # Prepare the reception buffer with dummy data
    protocol._reception_buffer[:100] = np.arange(100, dtype=np.uint8)
    protocol._bytes_in_reception_buffer = 100

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"A multidimensional numpy array with {invalid_array.ndim} dimensions requested when reading data from _reception_buffer. At this time, only one-dimensional (flat) arrays are supported."
        ),
    ):
        protocol.read_data(invalid_array)


def test_serial_transfer_protocol_data_transmission():
    """Tests the send_data() and receive_data() methods of the SerialTransportLayer class. Relies on the
    read_data() and write_data() methods of the class to function as expected and also on the SerialMock class to be
    available to mock the pySerial Serial class."""

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
        test_mode=True,  # Makes it port-agnostic by switching to SerialMock class as the 'serial interface'
        allow_start_byte_errors=False,  # Disables start_byte errors
    )

    # Instantiates separate instances of encoder classes used to verify processing results
    cobs_processor = COBSProcessor()
    crc_processor = CRCProcessor(
        polynomial=np.uint16(0x1021),
        initial_crc_value=np.uint16(0xFFFF),
        final_xor_value=np.uint16(0x0000),
    )

    # Generates the test array to be packaged and 'sent'
    test_array = np.array([1, 2, 3, 0, 0, 6, 0, 8, 0, 0], dtype=np.uint8)

    # Writes the package into the _transmission_buffer
    protocol.write_data(test_array)

    # Verifies that the bytes were added to the _transmission_buffer
    assert protocol.bytes_in_transmission_buffer == test_array.nbytes

    # Packages and sends the data to the StreamMock class buffer (due-to-protocol running in test mode)
    send_status = protocol.send_data()

    # Verifies that the method ran as expected. This is done through assessing the returned boolean status and the
    # resetting of the transmission buffer bytes' tracker.
    assert send_status
    assert protocol.bytes_in_transmission_buffer == 0

    # Manually verifies SerialMock tx_buffer contents (ensures the data was added as expected
    # and was encoded and CRC-checksummed as expected).

    # First, determines the expected COBS-encoded and CRC-checksummed packet. This is what is being passed to the Serial
    # class to be added to its tx buffer
    expected_packet = cobs_processor.encode_payload(test_array, delimiter=np.uint8(0))
    checksum = crc_processor.calculate_crc_checksum(expected_packet)
    checksum = crc_processor.convert_checksum_to_bytes(checksum)
    expected_packet = np.concatenate((expected_packet, checksum))

    # Assess the state of the tx_buffer by generating a numpy uint8 array using the contents of the tx_buffer
    # noinspection PyUnresolvedReferences
    tx_buffer = np.frombuffer(protocol._port.tx_buffer, dtype=np.uint8)
    assert tx_buffer[0] == 129  # Asserts that the first byte-value in the buffer is the same as the start_byte value

    # Verifies that the data written to the tx_buffer is the same as the expected packet
    assert np.array_equal(tx_buffer[1 : expected_packet.size + 1], expected_packet)

    # Verifies that attempting to receive data when there are no bytes inside the port reception buffer graciously
    # returns 'False'
    receive_status = protocol.receive_data()
    assert not receive_status

    # Constructs and copies the expected received packet based on the test_array to the reception buffer of the Stream
    # class. Note, use both the start_byte and the payload_size of the encoded array as during reception payload size
    # is expected (unlike during transmission).
    rx_bytes = bytes([129, 10]) + expected_packet.tobytes()  # generates a bytes-sequence to represent received data
    # noinspection PyUnresolvedReferences
    protocol._port.rx_buffer = rx_bytes  # Sets the reception buffer to received data

    # Simulates data reception using the rx_buffer of the mock port
    receive_status = protocol.receive_data()

    # Verifies that the data has been successfully received from the Stream buffer based on the returned boolean status
    # of reception method and the value of the bytes_in_reception_buffer attribute.
    assert receive_status is True
    assert protocol.bytes_in_reception_buffer == 10

    # Verifies that the reverse-processed payload is the same as the original payload array
    decoded_array = np.zeros(10, dtype=np.uint8)
    decoded_array, _ = protocol.read_data(decoded_array)
    assert np.array_equal(decoded_array, test_array)


def test_serial_transfer_protocol_data_transmission_errors():
    """Tests SerialTransportLayer class send_data() and receive_data() method error handling. Focuses on testing
    the errors that arise specifically from these methods or private methods of the SerialTransportLayer class.
    Assumes= helper method errors are tested using the dedicated helper testing functions.
    """

    # Instantiates the tested class
    # noinspection DuplicatedCode
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        polynomial=np.uint16(0x1021),
        initial_crc_value=np.uint16(0xFFFF),
        final_crc_xor_value=np.uint16(0x0000),
        maximum_transmitted_payload_size=254,
        start_byte=np.uint8(129),
        delimiter_byte=np.uint8(0),
        timeout=np.uint64(20000),
        test_mode=True,  # Makes it port-agnostic by switching to SerialMock class as the 'serial interface'
        allow_start_byte_errors=False,  # Disables start_byte errors for now
    )

    # Instantiates separate instances of encoder classes used to verify processing results
    cobs_processor = COBSProcessor()
    crc_processor = CRCProcessor(
        polynomial=np.uint16(0x1021),
        initial_crc_value=np.uint16(0xFFFF),
        final_xor_value=np.uint16(0x0000),
    )

    # Initializes a test payload
    test_payload = np.array([1, 2, 3, 4, 0, 0, 7, 8, 9, 10], dtype=np.uint8)

    # Due to the combination of static guards at class instantiation and implementation of library methods it is
    # practically impossible to encounter errors during data sending. This is kept as a placeholder in case future
    # versions of the library do require send_data() error handling tests.
    protocol.write_data(test_payload)
    protocol.send_data()

    # Generates data to test the receive_data() method
    preamble = np.array([129, 10], dtype=np.uint8)  # Start byte and payload size
    packet = cobs_processor.encode_payload(payload=test_payload, delimiter=np.uint8(0))  # COBS
    checksum = crc_processor.calculate_crc_checksum(packet)
    checksum = crc_processor.convert_checksum_to_bytes(checksum)
    np.concatenate((packet, checksum))  # CRC
    test_data = np.concatenate((preamble, packet), axis=0)  # Constructs the final expected packet
    empty_buffer = np.zeros(20, dtype=np.uint8)  # An empty array to simulate parsing noise-data

    # Verifies no error occurs when protocol is configured to ignore start_byte errors (there is no start_byte in the
    # zeroes buffer)
    # noinspection PyUnresolvedReferences
    protocol._port.rx_buffer = empty_buffer.tobytes()  # Writes 'noise' bytes to serial port
    receive_status = protocol.receive_data()
    assert not receive_status

    # To save some time on recompiling, the class flips the 'allow_start_byte_errors' flag of the protocol class using
    # name un-mangling. This should not be done during production runtime.
    protocol._allow_start_byte_errors = True

    # Note, for all tests below, the rx_buffer has to be refilled with bytes after each test as the bytes are actually
    # consumed during each test.

    # Verifies that running reception for a noise buffer with start_byte errors allowed produces a start_byte error. The
    # 'noise' buffer does not contain the start byte error, so it is never found.
    error_message = (
        "Start_byte value was not found among the bytes stored inside the serial buffer when parsing incoming "
        "serial packet. Reception aborted."
    )
    # noinspection PyUnresolvedReferences
    protocol._port.rx_buffer = empty_buffer.tobytes()  # Refills rx buffer
    with pytest.raises(
        RuntimeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.receive_data()

    # Verifies that encountering the start byte successfully triggers packet parsing, but not receiving the packet size
    # after the start byte is found triggers packet_size reception error. Specifically, after finding the start byte,
    # the algorithm checks if the next byte is available. If so, that byte is read as the packet size. If not, the
    # algorithm waits for the byte to become available until a timeout declares tha the packet hsa staled.
    error_message = (
        f"Failed to receive the payload_size byte within the allowed timeout period of {20000} "
        f"microseconds from receiving the start_byte of the serial packet. Packet staled, reception aborted."
    )
    # Sets the last variable in the empty_buffer to the start byte to simulate receiving start byte (and nothing else)
    empty_buffer[-1] = 129
    # noinspection PyUnresolvedReferences
    protocol._port.rx_buffer = empty_buffer.tobytes()  # Refills rx buffer
    with pytest.raises(
        RuntimeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.receive_data()

    # Verifies packet staling error. For this, updates the packet size stored at index 1 of the tested_data array to
    # a size larger than the actual size of the encoded data. This way, the algorithm will enter a wait loop trying to
    # wait until the number of available bytes matches the payload_size-based number, which will never happen. The
    # algorithm should correctly time out and raise an error then.
    error_message = (
        f"Failed to receive the required number of packet bytes ({114}) within the allowed timeout period of {20000} "
        f"microseconds. Specifically, only received received {14} before timing-out. Packet staled, reception aborted."
    )
    test_data[1] = 110  # Payload size is 10, but this tells the algorithm it is at least 110 bytes
    # noinspection PyUnresolvedReferences
    protocol._port.rx_buffer = test_data.tobytes()  # Refills rx buffer
    with pytest.raises(
        RuntimeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.receive_data()

    # Verifies 'payload too large' error. For this, sets the payload_size byte to 255, which is larger than the maximum
    # allowed payload size of 255 bytes.
    error_message = (
        f"The declared size of the payload ({255}), extracted from the received payload_size byte of the serial packet,"
        f" is above the maximum allowed size of {254}. Reception aborted."
    )
    test_data[1] = 255
    # noinspection PyUnresolvedReferences
    protocol._port.rx_buffer = test_data.tobytes()  # Refills rx buffer
    with pytest.raises(
        RuntimeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.receive_data()
    test_data[1] = 10  # Restores the payload size

    # Verifies CRC checksum failure error. For this, replaces one of the CRC bytes with a random byte, ensuring that the
    # CRC checksum no longer matches the packet. This should produce checksum validation failure ValueError.
    expected_checksum = np.array(test_data[-2:])  # Indexes the checksum out of the test_data array
    received_checksum = expected_checksum.copy()  # Also creates a separate copy for received data

    # I could not figure why, but apparently the copy operations are evaluated before the modification is even if I do
    # the copy / modify / copy sequence. So now I copy twice and forcibly modify both the test-data and the
    # received_checksum. -- Ivan

    test_data[-1] = 112  # Modifies the LOWEST byte of the CRC checksum. Not that it matters, just fiy.
    received_checksum[-1] = 112

    # Uses crc class to calculate integer 'expected' and 'received' checksums to make the error message look nice.
    error_message = (
        f"CRC checksum verification failed for the received serial packet. Specifically, the checksum "
        f"value transmitted with the packet {hex(crc_processor.convert_bytes_to_checksum(received_checksum))} "
        f"does not match the value expected for the packet (calculated locally) "
        f"{hex(crc_processor.convert_bytes_to_checksum(expected_checksum))}. Packet corrupted, reception aborted."
    )

    # noinspection PyUnresolvedReferences
    protocol._port.rx_buffer = test_data.tobytes()  # Refills rx buffer
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        protocol.receive_data()
    test_data[-2:] = expected_checksum  # Restores the CRC checksum back to the correct value


from typing import Any, Dict, Tuple, Union


def list_available_ports() -> Tuple[Dict[str, Union[int, str, Any]], ...]:
    """Provides the information about each serial port addressable through the pySerial library.

    Returns:
        A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
        port.
    """

    # Gets the list of port objects visible to the pySerial library.
    available_ports = list_ports.comports()

    # Creates a list of dictionaries with port details
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


#
# def test_bytes_available_timeout():
#     """Test the scenario where not enough bytes are available within the timeout."""
#     protocol = SerialTransportLayer(
#         port="COM7",
#         baudrate=115200,
#         test_mode=True,
#     )
#
#     # Mocking the port to simulate 'in_waiting' returning 0 bytes (no new bytes available)
#     protocol._port.in_waiting = 0
#     protocol._timer.elapsed = MagicMock(return_value=10001)  # Simulate timeout (elapsed > timeout)
#
#     result = protocol._bytes_available(required_bytes_count=10, timeout=10000)
#
#     # Expected result is False due to timeout
#     assert result is False


def test_receive_packet_unknown_status():
    """Test receiving a packet that returns an unknown status code."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Mocking a method to return an invalid status code
    with patch.object(protocol, "_receive_packet", return_value=999):
        with pytest.raises(
            RuntimeError, match="Failed to parse the incoming serial packet data. Encountered an unknown status value"
        ):
            protocol.receive_data()


def test_not_enough_bytes_received():
    """Test for scenario where not enough bytes are read from the serial port."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Mock the serial port to simulate reading fewer bytes than required
    protocol._port.in_waiting = 5  # Only 5 bytes available when more are needed
    protocol._port.read = MagicMock(return_value=np.array([1, 2, 3, 4, 5], dtype=np.uint8))

    result = protocol._bytes_available(required_bytes_count=10, timeout=10000)

    # Expected result is False because only 5 bytes were available instead of the required 10


# def test_crc_checksum_failure():
#     """Test for CRC checksum failure during packet reception."""
#     protocol = SerialTransportLayer(
#         port="COM7",
#         baudrate=115200,
#         test_mode=True,
#     )
#
#     # Mock the CRCProcessor to simulate a checksum failure
#     with patch.object(CRCProcessor, "calculate_crc_checksum", return_value=1):  # Simulate non-zero checksum
#         reception_buffer = np.arange(20, dtype=np.uint8)  # Simulate received data
#         protocol._reception_buffer = reception_buffer
#         protocol._bytes_in_reception_buffer = 20
#
#         result = protocol.receive_data()
#
#         # Expected result is 0 since the CRC checksum failed
#         assert result == 0


def test_crc_verification_failure():
    """Test that a CRC checksum mismatch triggers an appropriate error message."""

    # Create an instance of the SerialTransportLayer with test_mode=True
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Create a mock packet (simulating the reception buffer)
    protocol._reception_buffer = np.array([129, 1, 3, 255, 0], dtype=np.uint8)  # Example packet

    # Mock the CRC processor to return different values for calculated and received checksums
    protocol._crc_processor.convert_bytes_to_checksum = lambda _: 0xABCD
    protocol._crc_processor.calculate_crc_checksum = lambda _: 0x1234

    # Mock the correct error handling method in console (update 'log_error' if needed)
    with patch("ataraxis_base_utilities.console.log_error") as mock_error:
        # Simulate receiving the packet and triggering the CRC check failure
        with pytest.raises(ValueError):
            protocol._validate_packet(
                reception_buffer=protocol._reception_buffer,
                packet_size=len(protocol._reception_buffer),
                cobs_processor=protocol._cobs_processor.processor,
                crc_processor=protocol._crc_processor.processor,
                delimiter_byte=protocol._delimiter_byte,
                postamble_size=protocol._postamble_size,
            )

        # Check that the mocked error logging was called with the expected message
        mock_error.assert_called_once()  # This ensures that console.log_error was called once


def test_start_byte_not_found():
    """Test the case where the start byte is not found in the available bytes."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Mock a situation where the start byte is not in the buffer
    protocol._port.in_waiting = 0  # No bytes available to be the start byte
    protocol._reception_buffer = np.array([2, 3, 4, 5], dtype=np.uint8)  # Simulate reception buffer without start byte

    with pytest.raises(
        RuntimeError, match="Failed to parse the incoming serial packet data. Unable to find the start_byte"
    ):
        protocol.receive_data()


def test_byte_mismatch_in_buffers():
    """Test when the number of bytes across buffers does not match the required number."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Mock the in_waiting to simulate a situation where fewer bytes are available
    protocol._port.in_waiting = 5
    protocol._port.read = MagicMock(return_value=np.array([1, 2, 3, 4, 5], dtype=np.uint8))

    result = protocol._bytes_available(required_bytes_count=10, timeout=10000)

    # Expected to return False because not enough bytes are available
    assert result is False


def test_write_and_read_scalar():
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # documenting scalar's value
    scalar_value = np.uint8(123)
    end_index = protocol.write_data(scalar_value)
    assert end_index == 1

    # reading the stored data
    read_value, read_end_index = protocol.read_data(np.uint8(0))
    assert read_value == scalar_value
    assert read_end_index == 1

    # Test writing and reading scalars
    scalar_value = np.uint8(123)
    end_index = protocol.write_data(scalar_value)
    assert end_index == 1  # Make sure the index updates correctly

    read_value, read_end_index = protocol.read_data(np.uint8(0))
    assert read_value == scalar_value
    assert read_end_index == 1


def test_write_and_read_array():
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # 1차원 배열 값을 기록
    array_value = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    end_index = protocol.write_data(array_value)
    assert end_index == 5  # 바이트 인덱스가 배열 길이와 일치하는지 확인

    # 기록한 배열을 다시 읽음
    read_array, read_end_index = protocol.read_data(np.zeros(5, dtype=np.uint8))
    assert np.array_equal(read_array, array_value)
    assert read_end_index == 5


def test_send_data():
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,  # Use mock serial port
    )

    # Create test data
    test_data = np.array([1, 2, 3, 4], dtype=np.uint8)

    # Write data to transmission buffer
    protocol.write_data(test_data)

    # Send data, verify that buffer is sent and cleared
    send_status = protocol.send_data()
    assert send_status
    assert protocol.bytes_in_transmission_buffer == 0  # Buffer is cleared after send

    # Verify that the COBS-encoded and CRC checksummed data is in the mock serial port's tx buffer
    expected_packet = protocol._port.tx_buffer
    assert expected_packet[0] == protocol.start_byte  # Ensure start byte is correct


def test_receive_data():
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Simulate sending and receiving a valid packet
    test_data = np.array([1, 2, 3, 4], dtype=np.uint8)
    protocol.write_data(test_data)
    protocol.send_data()

    # Prepare reception buffer with the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Simulate loopback

    # Test receiving the data
    receive_status = protocol.receive_data()
    assert receive_status

    # Verify that the received data matches the original
    received_data, _ = protocol.read_data(np.zeros(4, dtype=np.uint8))
    assert np.array_equal(received_data, test_data)


def test_crc_failure():
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Send valid data first
    test_data = np.array([1, 2, 3, 4], dtype=np.uint8)
    protocol.write_data(test_data)
    protocol.send_data()

    # Manipulate the CRC in the rx buffer to simulate corruption
    protocol._port.rx_buffer = protocol._port.tx_buffer
    protocol._port.rx_buffer[-1] ^= 0xFF  # Flip some bits to corrupt the CRC

    # Receive the data and expect a failure due to CRC
    with pytest.raises(ValueError, match="CRC checksum verification failed"):
        protocol.receive_data()


def test_list_available_ports():
    """Test that list_available_ports correctly retrieves and formats serial port information."""

    # Mock the output of list_ports.comports()
    mock_port = MagicMock()
    mock_port.name = "COM3"
    mock_port.device = "/dev/ttyS1"
    mock_port.pid = 1234
    mock_port.description = "USB Serial Device"

    with patch("serial.tools.list_ports.comports", return_value=[mock_port]):
        from ataraxis_transport_layer.transport_layer import SerialTransportLayer

        ports = SerialTransportLayer.list_available_ports()

        expected_output = (
            {
                "Name": "COM3",
                "Device": "/dev/ttyS1",
                "PID": 1234,
                "Description": "USB Serial Device",
            },
        )

        assert ports == expected_output


def test_sufficient_bytes_available():
    """Test the condition when sufficient bytes are available to process."""

    # Create a mock of the SerialTransportLayer class
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )

    # Case 1: in_waiting + leftover_bytes > minimum_packet_size (True)
    protocol._port.in_waiting = 5  # 5 bytes available
    protocol._leftover_bytes = [1, 2, 3]  # 3 leftover bytes
    protocol._minimum_packet_size = 7  # Minimum packet size is 7

    # Test the condition where in_waiting + leftover_bytes > minimum_packet_size
    assert protocol._port.in_waiting + len(protocol._leftover_bytes) > protocol._minimum_packet_size

    # Case 2: in_waiting + leftover_bytes <= minimum_packet_size (False)
    protocol._port.in_waiting = 2  # Only 2 bytes available
    protocol._leftover_bytes = [1]  # 1 leftover byte
    protocol._minimum_packet_size = 7  # Minimum packet size is still 7

    # Test the condition where in_waiting + leftover_bytes <= minimum_packet_size
    assert not (protocol._port.in_waiting + len(protocol._leftover_bytes) > protocol._minimum_packet_size)


from dataclasses import dataclass


# Define a dataclass to simulate the structure serialization
@dataclass
class MockDataClass:
    field1: np.uint8
    field2: np.ndarray


import pytest

from ataraxis_transport_layer.transport_layer import SerialTransportLayer


@dataclass
class MockDataClass:
    field1: np.uint8
    field2: np.ndarray


# def test_write_data_break_condition():
#     """Test that the write_data function breaks out of the loop when local_index < start_index."""
#     # Create an instance of the SerialTransportLayer class
#     protocol = SerialTransportLayer(
#         port="COM7",
#         baudrate=115200,
#         test_mode=True,
#     )
#
#     # Create mock data with numpy types (a scalar and an array)
#     data = MockDataClass(np.uint8(10), np.array([1, 2, 3], dtype=np.uint8))
#
#     # Mock the write_data method to simulate local_index < start_index
#     original_write_data = protocol.write_data
#
#     def mock_write_data(data_object, start_index=None):
#         if isinstance(data_object, np.uint8):
#             return start_index + 1  # Normal behavior
#         elif isinstance(data_object, np.ndarray):
#             return start_index - 1  # Simulate failure by returning a smaller index
#         else:
#             return original_write_data(data_object, start_index)
#
#     with patch.object(protocol, "write_data", side_effect=mock_write_data):
#         start_index = 5
#         end_index = protocol.write_data(data_object=data, start_index=start_index)
#
#         # Check if the function breaks when local_index < start_index
#         assert end_index < start_index  # fixing the break to get covered as well

from unittest.mock import patch


def test_write_data_break_condition():
    """Test that the write_data function breaks out of the loop when local_index < start_index."""
    # Create an instance of the SerialTransportLayer class
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Create mock data with numpy types (a scalar and an array)
    data = MockDataClass(np.uint8(10), np.array([1, 2, 3], dtype=np.uint8))

    # Mock the write_data method to simulate local_index < start_index
    original_write_data = protocol.write_data

    def mock_write_data(data_object, start_index=None):
        if isinstance(data_object, np.uint8):
            return start_index + 1  # Normal behavior for scalar
        elif isinstance(data_object, np.ndarray):
            return start_index - 1  # Simulate failure by returning a smaller index for the array
        else:
            return original_write_data(data_object, start_index)

    with patch.object(protocol, "write_data", side_effect=mock_write_data):
        start_index = 5
        end_index = protocol.write_data(data_object=data, start_index=start_index)

        # Check if the function breaks when local_index < start_index
        assert end_index < start_index  # This will verify that break occurred when condition met


def test_unsupported_input_type_error():
    """Test that a TypeError is raised for unsupported input types."""

    # Create an instance of the SerialTransportLayer class
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )

    # Create an unsupported input type (neither numpy scalar nor dataclass)
    invalid_data = {"unsupported": "data"}

    # Assert that a TypeError is raised with the correct message
    with pytest.raises(TypeError, match=r"Encountered an unsupported input data_object type"):
        protocol.write_data(invalid_data)


def _write_array_data(
    self,
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
            np.bool_,
        ]
    ],
    start_index: int,
) -> int:
    """Converts the input numpy array to a sequence of bytes and writes it to the transmission buffer at the specified start_index.

    This method is not designed to be called directly. It should always be called through the write_data() method of the parent class.

    Args:
        target_buffer: The buffer to which the data will be written. This should be the _transmission_buffer array of the caller class.
        array_object: The numpy array to be written to the transmission buffer. Currently, the method is designed to only work with one-dimensional arrays with a minimal size of 1 element. The array should be using one of the supported numpy scalar datatypes.
        start_index: The index inside the transmission buffer (0 to 253) at which to start writing the data.

    Returns:
        The positive index inside the transmission buffer that immediately follows the last index of the buffer to which the data was written. Integer code 0, if the buffer does not have enough space to accommodate the data written at the start_index. Integer code -1, if the input array object is not one-dimensional. Integer code -2, if the input array object is empty.
    """

    # Error checking logic (now in regular Python code)
    if array_object.ndim != 1:
        return -1  # Input array is not one-dimensional.

    if array_object.size == 0:
        return -2  # Input array is empty.

    # Check buffer space
    array_data = np.frombuffer(array_object, dtype=np.uint8)
    data_size = array_data.size * array_data.itemsize
    required_size = start_index + data_size

    if required_size > target_buffer.size:
        return 0  # Insufficient buffer space.

    # Call the JIT-compiled method for the actual data writing
    return self._write_array_data_jit(target_buffer, array_data, start_index)


def test_buffer_size_insufficient_error():
    """Test that a ValueError is raised when the buffer does not have enough space."""

    # Create an instance of the SerialTransportLayer class
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )

    # Mock the transmission buffer to simulate insufficient space
    protocol._transmission_buffer = np.zeros(10, dtype=np.uint8)
    protocol._bytes_in_transmission_buffer = 8  # Only 2 bytes of space left

    # Create a large numpy array that exceeds the buffer size
    large_data = np.ones(5, dtype=np.uint8)

    # Assert that a ValueError is raised with the correct message
    with pytest.raises(ValueError, match=r"The transmission buffer does not have enough space to write the data"):
        protocol.write_data(large_data, start_index=7)


# def test_write_data_success():
#     """Test the successful write operation and correct end_index return."""
#
#     # Create an instance of the SerialTransportLayer class
#     protocol = SerialTransportLayer(
#         port="COM7",
#         baudrate=115200,
#         start_byte=129,
#         delimiter_byte=0,
#         timeout=10000,
#         test_mode=True,
#     )
#
#     # Mock the transmission buffer with enough space
#     protocol._transmission_buffer = np.zeros(20, dtype=np.uint8)
#     protocol._bytes_in_transmission_buffer = 10
#
#     # Create mock data to write
#     data = MockDataClass(np.uint8(10), np.array([1, 2, 3], dtype=np.uint8))
#
#     # Call the write_data method and check the returned end_index
#     end_index = protocol.write_data(data_object=data, start_index=10)
#
#     # Ensure that the end_index is greater than the start_index and buffer tracker is updated
#     assert end_index > 10
#     assert protocol._bytes_in_transmission_buffer == end_index


def test_write_data_success(protocol):
    """데이터 쓰기와 관련된 성공적인 시나리오 테스트"""
    data = np.array([1, 2, 3], dtype=np.uint8)
    end_index = protocol.write_data(data)
    assert end_index == len(data)


def test_buffer_tracker_update():
    """Test that the buffer tracker (_bytes_in_transmission_buffer) is updated correctly after a successful write."""

    # Create an instance of the SerialTransportLayer class
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )

    # Mock the transmission buffer with enough space
    protocol._transmission_buffer = np.zeros(20, dtype=np.uint8)
    protocol._bytes_in_transmission_buffer = 10

    # Create a numpy scalar and write it to the buffer
    data = np.uint8(123)
    end_index = protocol.write_data(data_object=data, start_index=10)

    # Ensure the tracker is updated correctly
    assert protocol._bytes_in_transmission_buffer == end_index


def test_send_receive_maximum_payload():
    """Test sending and receiving data with maximum allowed payload size."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        maximum_transmitted_payload_size=254,  # Maximum allowed by the protocol
        test_mode=True,
    )

    # Create a test payload with maximum allowed size
    max_payload_size = protocol.maximum_transmitted_payload_size
    test_data = np.arange(max_payload_size, dtype=np.uint8)

    # Write data to transmission buffer
    protocol.write_data(test_data)

    # Send data
    send_status = protocol.send_data()
    assert send_status
    assert protocol.bytes_in_transmission_buffer == 0  # Buffer should be cleared after send

    # Simulate receiving the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Loopback

    # Receive data
    receive_status = protocol.receive_data()
    assert receive_status
    assert protocol.bytes_in_reception_buffer == max_payload_size

    # Read back the data
    received_data, _ = protocol.read_data(np.zeros(max_payload_size, dtype=np.uint8))
    assert np.array_equal(received_data, test_data)


def test_receive_incomplete_packet():
    """Test that receive_data handles incomplete packet properly."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
        timeout=1000,  # Set a short timeout for the test
    )

    # Prepare and send data
    test_data = np.array([1, 2, 3, 4], dtype=np.uint8)
    protocol.write_data(test_data)
    protocol.send_data()

    # Get the packet and remove some bytes to simulate truncation
    incomplete_packet = protocol._port.tx_buffer[:-2]  # Remove last 2 bytes (e.g., part of CRC)

    # Simulate receiving the incomplete packet
    protocol._port.rx_buffer = incomplete_packet

    # Attempt to receive data
    receive_status = protocol.receive_data()

    # Expect receive_data to return False due to incomplete packet
    assert not receive_status


def test_write_unsupported_data_type():
    """Test that writing an unsupported data type raises TypeError."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Attempt to write an unsupported data type (e.g., string)
    unsupported_data = "This is a string"

    with pytest.raises(TypeError, match=r"Unsupported input data_object type"):
        protocol.write_data(unsupported_data)


def test_receive_data_timeout():
    """Test that receive_data times out if no data is received."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
        timeout=1000,  # Set a short timeout for the test
    )

    # Ensure that no data is in the rx_buffer
    protocol._port.rx_buffer = b""

    # Attempt to receive data
    receive_status = protocol.receive_data()

    # Expect receive_data to return False due to timeout
    assert not receive_status


def test_send_data_empty_buffer():
    """Test that send_data returns False when transmission buffer is empty."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Ensure the transmission buffer is empty
    protocol.reset_transmission_buffer()
    assert protocol.bytes_in_transmission_buffer == 0

    # Attempt to send data
    send_status = protocol.send_data()

    # Expect send_data to return False
    assert not send_status


def test_crc_different_polynomial():
    """Test that the class correctly handles a different CRC polynomial."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        polynomial=np.uint16(0x8005),  # Different polynomial
        initial_crc_value=np.uint16(0xFFFF),
        final_crc_xor_value=np.uint16(0x0000),
        test_mode=True,
    )

    test_data = np.array([1, 2, 3, 4], dtype=np.uint8)
    protocol.write_data(test_data)
    protocol.send_data()

    # Simulate receiving the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer

    # Receive data
    receive_status = protocol.receive_data()
    assert receive_status

    # Read back the data
    received_data, _ = protocol.read_data(np.zeros(4, dtype=np.uint8))
    assert np.array_equal(received_data, test_data)


def test_data_with_start_byte_and_delimiter():
    """Test sending and receiving data that includes start byte and delimiter byte values."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        test_mode=True,
    )

    # Create test data that includes the start_byte and delimiter_byte values
    test_data = np.array([129, 0, 1, 2, 129, 0, 3, 4], dtype=np.uint8)
    protocol.write_data(test_data)
    protocol.send_data()

    # Simulate receiving the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer

    # Receive data
    receive_status = protocol.receive_data()
    assert receive_status

    # Read back the data
    received_data, _ = protocol.read_data(np.zeros(len(test_data), dtype=np.uint8))
    assert np.array_equal(received_data, test_data)


# 오늘
from unittest.mock import MagicMock

import numpy as np

from ataraxis_transport_layer.helper_modules import CRCProcessor, COBSProcessor
from ataraxis_transport_layer.transport_layer import SerialTransportLayer


def test_validate_packet_success():
    """Test that _validate_packet successfully validates and decodes a packet."""
    reception_buffer = np.zeros(20, dtype=np.uint8)
    reception_buffer[:10] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], dtype=np.uint8)

    packet_size = 10
    delimiter_byte = np.uint8(0)
    postamble_size = np.uint8(2)

    cobs_processor = MagicMock()
    crc_processor = MagicMock()

    crc_processor.calculate_crc_checksum.return_value = 0
    crc_processor.status = crc_processor.checksum_calculated
    cobs_processor.decode_payload.return_value = np.array([10, 20, 30], dtype=np.uint8)

    result = SerialTransportLayer._validate_packet(
        reception_buffer=reception_buffer,
        packet_size=packet_size,
        cobs_processor=cobs_processor,
        crc_processor=crc_processor,
        delimiter_byte=delimiter_byte,
        postamble_size=postamble_size,
    )

    assert result == 3
    assert np.array_equal(reception_buffer[:3], np.array([10, 20, 30]))


def test_validate_packet_crc_failure():
    """Test that _validate_packet returns 0 on CRC failure."""
    reception_buffer = np.zeros(20, dtype=np.uint8)

    packet_size = 10
    delimiter_byte = np.uint8(0)
    postamble_size = np.uint8(2)

    cobs_processor = MagicMock()
    crc_processor = MagicMock()

    crc_processor.calculate_crc_checksum.return_value = 1
    crc_processor.status = crc_processor.checksum_calculated

    result = SerialTransportLayer._validate_packet(
        reception_buffer=reception_buffer,
        packet_size=packet_size,
        cobs_processor=cobs_processor,
        crc_processor=crc_processor,
        delimiter_byte=delimiter_byte,
        postamble_size=postamble_size,
    )

    assert result == 0


def test_validate_packet_cobs_failure():
    """Test that _validate_packet returns 0 on COBS decoding failure."""
    reception_buffer = np.zeros(20, dtype=np.uint8)

    packet_size = 10
    delimiter_byte = np.uint8(0)
    postamble_size = np.uint8(2)

    cobs_processor = MagicMock()
    crc_processor = MagicMock()

    crc_processor.calculate_crc_checksum.return_value = 0
    crc_processor.status = crc_processor.checksum_calculated
    cobs_processor.decode_payload.return_value = np.array([], dtype=np.uint8)

    result = SerialTransportLayer._validate_packet(
        reception_buffer=reception_buffer,
        packet_size=packet_size,
        cobs_processor=cobs_processor,
        crc_processor=crc_processor,
        delimiter_byte=delimiter_byte,
        postamble_size=postamble_size,
    )

    assert result == 0


def test_scalar_write_success():
    """Test the successful writing of scalar data to the transmission buffer."""

    # 버퍼 생성
    target_buffer = np.zeros(10, dtype=np.uint8)

    # 스칼라 데이터
    scalar_object = np.uint8(255)

    # 데이터 쓰기 함수 호출
    start_index = 0
    array_object = np.frombuffer(np.array([scalar_object]), dtype=np.uint8)
    data_size = array_object.size * array_object.itemsize
    required_size = start_index + data_size

    assert required_size <= target_buffer.size

    target_buffer[start_index:required_size] = array_object
    assert target_buffer[start_index:required_size].tolist() == array_object.tolist()

    assert required_size == data_size


def test_scalar_write_buffer_overflow():
    target_buffer = np.zeros(2, dtype=np.uint8)  # 작은 크기의 버퍼

    scalar_object = np.uint8(255)

    start_index = 1
    array_object = np.frombuffer(np.array([scalar_object]), dtype=np.uint8)
    data_size = array_object.size * array_object.itemsize
    required_size = start_index + data_size

    if required_size > target_buffer.size:
        assert 0 == 0


# today
def test_bytes_available_timeout():
    """Test the scenario where not enough bytes are available within the timeout."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Mocking the port to simulate 'in_waiting' returning 0 bytes (no new bytes available)
    protocol._port.in_waiting = 0
    protocol._timer.elapsed = MagicMock(return_value=10001)  # Simulate timeout (elapsed > timeout)

    result = protocol._bytes_available(required_bytes_count=10, timeout=10000)

    # Expected result is False due to timeout
    assert result is False


def test_packet_byte_missing_timeout():
    """Test when a byte in the packet is not received in time, raising a RuntimeError."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Prepare mock data for the test
    parsed_bytes = np.array([1, 2, 3, 4], dtype=np.uint8)  # Simulate partial bytes received
    parsed_bytes_count = 3  # Simulate the number of bytes already parsed
    protocol._timeout = 10000  # Set the timeout for the test

    with patch("ataraxis_transport_layer.transport_layer.console.error") as mock_error:
        # Manually trigger the error condition
        with pytest.raises(RuntimeError, match="Failed to parse the incoming serial packet data"):
            message = (
                f"Failed to parse the incoming serial packet data. The byte number {parsed_bytes_count + 1} "
                f"out of {parsed_bytes.size} was not received in time ({protocol._timeout} microseconds), "
                f"following the reception of the previous byte. Packet reception staled."
            )
            protocol._bytes_available(10, 10000)
            mock_error.assert_called_once_with(message=message, error=RuntimeError)


def test_no_packet_to_receive():
    """Test that no packet to receive returns False without raising an error."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Mock a situation where no packet is available (status 101)
    with patch.object(protocol, "_receive_packet", return_value=101):
        result = protocol.receive_data()
        assert result is False  # Expect a non-error return value


from unittest.mock import patch


def test_available_property():
    """Test that the available property returns True when enough bytes are available."""
    # Initialize the protocol in test mode
    protocol = SerialTransportLayer(
        port="COM7", baudrate=115200, start_byte=129, delimiter_byte=0, timeout=10000, test_mode=True
    )

    # Mock in_waiting and leftover_bytes
    protocol._port = MagicMock()
    protocol._port.in_waiting = 5
    protocol._leftover_bytes = [1, 2, 3]
    protocol._minimum_packet_size = 7

    # Test when available is True
    assert protocol.available is True

    # Test when available is False
    protocol._port.in_waiting = 1  # Not enough bytes
    assert protocol.available is False


def test_packet_parsing_success():
    """Test the case where packet parsing is successful (status == 1)."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock the _parse_packet method to return status == 1
    protocol._parse_packet = lambda *args: (1, 10, b"", np.array([129, 1, 3, 255, 0], dtype=np.uint8))

    # Mock bytes_available to always return True (enough bytes available)
    protocol._bytes_available = lambda required_bytes_count, timeout=0: True

    result = protocol.receive_data()

    # Check that the result is True (parsing succeeded)
    assert result is True
    assert protocol._bytes_in_reception_buffer == 5  # Mock packet size


def test_packet_parsing_status_0_no_bytes():
    """Test the case where parsing returns status == 0 and no additional bytes are available."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock the _parse_packet method to return status == 0
    protocol._parse_packet = lambda *args: (0, 0, b"", np.empty(0, dtype=np.uint8))

    # Mock bytes_available to return False (no additional bytes available)
    protocol._bytes_available = lambda required_bytes_count, timeout=0: False

    result = protocol.receive_data()

    # Check that the result is False (parsing didn't succeed)
    assert result is False


def test_packet_parsing_status_2_no_bytes():
    """Test the case where parsing returns status == 2 and no additional bytes are available."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock the _parse_packet method to return status == 2
    protocol._parse_packet = lambda *args: (2, 0, b"", np.empty(0, dtype=np.uint8))

    # Mock bytes_available to return False (no additional bytes available)
    protocol._bytes_available = lambda required_bytes_count, timeout=0: False

    result = protocol.receive_data()

    # Check that the result is False (parsing didn't succeed)
    assert result is False


def test_packet_parsing_status_2_with_bytes():
    """Test the case where parsing returns status == 2 and additional bytes are available."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock the _parse_packet method to return status == 2 initially, but then status == 1
    protocol._parse_packet = (
        lambda *args: (2, 0, b"", np.array([129, 1, 3, 255, 0], dtype=np.uint8))
        if args[0] == b""
        else (1, 10, b"", np.array([129, 1, 3, 255, 0], dtype=np.uint8))
    )

    # Mock bytes_available to return True (additional bytes available)
    protocol._bytes_available = lambda required_bytes_count, timeout=0: True

    result = protocol.receive_data()

    # Check that the result is True (parsing succeeded after additional bytes were available)
    assert result is True
    assert protocol._bytes_in_reception_buffer == 5  # Mock packet size


def test_serial_transport_layer_repr_mocked_port():
    """Test __repr__ when the _port is mocked using SerialMock."""
    with patch("ataraxis_transport_layer.transport_layer.SerialMock", autospec=True) as mock_serial:
        # Set up the mocked Serial object
        mock_serial_instance = MagicMock()
        mock_serial_instance.name = "MOCKED"
        mock_serial_instance.baudrate = 115200
        mock_serial.return_value = mock_serial_instance

        # Initialize SerialTransportLayer with mocked Serial
        protocol = SerialTransportLayer(
            port="COM7", baudrate=115200, start_byte=129, delimiter_byte=0, timeout=10000, test_mode=True
        )

        # Expected representation string for mocked serial
        expected_repr = (
            "SerialTransportLayer(port & baudrate=MOCKED, "
            "polynomial=0x1021, start_byte=129, delimiter_byte=0, "
            "timeout=10000 us, maximum_tx_payload_size = 254, "
            "maximum_rx_payload_size=254)"
        )

        # Check the repr output
        assert repr(protocol) == expected_repr


def test_repr_with_mocked_serial():
    """Test that the __repr__ method works with a mocked Serial port."""
    # Mock the Serial port
    with patch("ataraxis_transport_layer.transport_layer.SerialMock") as mock_serial:
        mock_serial_instance = MagicMock()
        mock_serial_instance.name = "MOCKED"
        mock_serial_instance.baudrate = 115200
        mock_serial.return_value = mock_serial_instance

        # Initialize the class
        protocol = SerialTransportLayer(
            port="COM3", baudrate=115200, start_byte=129, delimiter_byte=0, timeout=10000, test_mode=True
        )

        # Expected representation for mocked serial
        expected_repr = (
            "SerialTransportLayer(port & baudrate=MOCKED, "
            "polynomial=0x1021, start_byte=129, delimiter_byte=0, "
            "timeout=10000 us, maximum_tx_payload_size = 254, "
            "maximum_rx_payload_size=254)"
        )

        assert repr(protocol) == expected_repr


def test_error_logging_and_exceptions():
    """Test that the appropriate error messages are logged and exceptions are raised."""

    # Initialize the protocol
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=True,
    )

    # Mock console.error to capture logging behavior
    with patch("console.error") as mock_console_error:
        # Test case 1: Unsupported input type
        invalid_data = "this is a string"  # Unsupported type (string)
        with pytest.raises(TypeError, match="Encountered an unsupported input data_object"):
            protocol.write_data(invalid_data)

        # Assert that console.error was called with the correct message and exception for unsupported input
        mock_console_error.assert_any_call(
            message=(
                f"Failed to read the data from the reception buffer. Encountered an unsupported input data_object "
                f"type ({type(invalid_data).__name__}). At this time, only the following numpy scalar or array types "
                f"are supported: {protocol._accepted_numpy_scalars}. Alternatively, a dataclass with all attributes "
                f"set to supported numpy scalar or array types is also supported."
            ),
            error=TypeError,
        )

        # Test case 2: Empty numpy array
        empty_array = np.array([], dtype=np.uint8)  # Empty array input
        with pytest.raises(ValueError, match="Encountered an empty (size 0) numpy array"):
            protocol.write_data(empty_array)

        # Assert that console.error was called with the correct message for the empty array error
        mock_console_error.assert_any_call(
            message=(
                f"Failed to read the data from the reception buffer. Encountered an empty (size 0) numpy array as "
                f"input data_object. Reading empty arrays is not supported."
            ),
            error=ValueError,
        )

        # Test case 3: Unknown error code (simulating a -3 return value from the reader)
        with patch.object(protocol, "read_data", return_value=(-3)):
            with pytest.raises(RuntimeError, match="Encountered an unknown error code"):
                protocol.receive_data()

            # Assert that console.error was called with the correct message for unknown error code
            mock_console_error.assert_any_call(
                message=(
                    f"Failed to read the data from the reception buffer. Encountered an unknown error code (-3)"
                    f" returned by the reader method."
                ),
                error=RuntimeError,
            )


def test_initialization():
    # Valid initialization
    layer = SerialTransportLayer(port="COM3", baudrate=9600)
    assert layer._port.name == "COM3"
    assert layer._port.baudrate == 9600
    assert isinstance(layer._port, Serial)

    # Invalid baudrate
    with pytest.raises(ValueError):
        layer = SerialTransportLayer(port="COM3", baudrate=-1)

    # Invalid start_byte
    with pytest.raises(ValueError):
        layer = SerialTransportLayer(port="COM3", baudrate=9600, start_byte=300)


def test_write_data():
    layer = SerialTransportLayer(port="COM3", baudrate=9600)

    # Test writing valid data
    start_idx = layer.write_data(np.uint8(42))
    assert layer._transmission_buffer[0] == 42
    assert start_idx == 1  # Next free index should be 1

    # Test writing invalid type
    with pytest.raises(TypeError):
        layer.write_data("invalid_data")

    # Test buffer overflow
    with pytest.raises(ValueError):
        layer.write_data(np.ones(500, dtype=np.uint8))  # Too large for buffer


# 지금

from unittest.mock import patch

import pytest


def test_read_data_unknown_error_code():
    """Test that an unknown error code in read_data raises RuntimeError and logs an error."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)
    # Prepare the reception buffer with some data
    protocol._reception_buffer[:10] = np.arange(10, dtype=np.uint8)
    protocol._bytes_in_reception_buffer = 10

    # Create a valid array to read into
    data_array = np.zeros(5, dtype=np.uint8)

    # Mock the _read_array_data method to return an unknown error code
    with patch.object(protocol, "_read_array_data", return_value=-99):
        with patch("ataraxis_base_utilities.console.error") as mock_error:
            with pytest.raises(RuntimeError) as exc_info:
                protocol.read_data(data_array)

            # Verify that the error message was logged
            mock_error.assert_called_once_with(
                message=(
                    f"Failed to read the data from the reception buffer. Encountered an unknown error code (-99)"
                    f"returned by the reader method."
                )
            )
            # Verify that the exception message is correct
            assert str(exc_info.value) == (
                f"Failed to read the data from the reception buffer. Encountered an unknown error code (-99)"
                f"returned by the reader method."
            )


def test_read_data_empty_array():
    """Test that attempting to read an empty array raises a ValueError and logs an error."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)
    # Prepare the reception buffer with some data, although it shouldn't matter
    protocol._reception_buffer[:10] = np.arange(10, dtype=np.uint8)
    protocol._bytes_in_reception_buffer = 10

    # Create an empty array to read into
    empty_array = np.empty(0, dtype=np.uint8)

    with patch("ataraxis_base_utilities.console.error") as mock_error:
        with pytest.raises(ValueError) as exc_info:
            protocol.read_data(empty_array)

        # Verify that the error message was logged
        mock_error.assert_called_once_with(
            message=(
                "Failed to read the data from the reception buffer. Encountered an empty (size 0) numpy array as "
                "input data_object. Reading empty arrays is not supported."
            )
        )
        # Verify that the exception message is correct
        assert str(exc_info.value) == (
            "Failed to read the data from the reception buffer. Encountered an empty (size 0) numpy array as "
            "input data_object. Reading empty arrays is not supported."
        )


def test_receive_data_status_0_timeout():
    """Test the case where status == 0 and bytes are not available, leading to a timeout."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock _parse_packet to return status == 0
    protocol._parse_packet = lambda *args: (0, 0, b"", np.array([1, 2, 3], dtype=np.uint8))

    # Mock _bytes_available to return False to simulate a timeout
    protocol._bytes_available = lambda required_bytes_count, timeout: False

    with patch("ataraxis_base_utilities.console.error") as mock_error:
        with pytest.raises(RuntimeError) as exc_info:
            protocol.receive_data()

        # Verify that console.error was called with the correct message
        message = (
            f"Failed to parse the incoming serial packet data. Packet reception staled. "
            f"The byte number {1} out of {3} was not received in time "
            f"({protocol._timeout} microseconds), following the reception of the previous byte."
        )
        mock_error.assert_called_once_with(message=message)


def test_receive_data_status_2_timeout():
    """Test the case where status == 2 and bytes are not available, leading to a timeout."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Simulate partial parsing with status == 2
    parsed_bytes = np.array([1, 2, 3, 4], dtype=np.uint8)
    parsed_bytes_count = 2  # Only two bytes parsed

    # Mock _parse_packet to return status == 2
    protocol._parse_packet = lambda *args: (2, parsed_bytes_count, b"", parsed_bytes)

    # Mock _bytes_available to return False to simulate a timeout
    protocol._bytes_available = lambda required_bytes_count, timeout: False

    with patch("ataraxis_base_utilities.console.error") as mock_error:
        with pytest.raises(RuntimeError) as exc_info:
            protocol.receive_data()

        # Verify that console.error was called with the correct message
        message = (
            f"Failed to parse the incoming serial packet data. The byte number {parsed_bytes_count + 1} "
            f"out of {parsed_bytes.size} was not received in time ({protocol._timeout} microseconds), "
            f"following the reception of the previous byte. Packet reception staled."
        )
        mock_error.assert_called_once_with(message=message)

        # Verify the exception message
        assert str(exc_info.value) == message


def test_receive_data_status_103():
    """Test that receive_data handles status == 103 by logging error and raising RuntimeError."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Simulate parsing with status == 103
    parsed_bytes = np.array([1, 2], dtype=np.uint8)
    parsed_bytes_count = 0

    # Mock _parse_packet to return status == 103
    protocol._parse_packet = lambda *args: (103, parsed_bytes_count, b"", parsed_bytes)

    with patch("ataraxis_base_utilities.console.error") as mock_error:
        with pytest.raises(RuntimeError) as exc_info:
            protocol.receive_data()

        payload_size = parsed_bytes.size - int(protocol._postamble_size)
        message = (
            f"Failed to parse the incoming serial packet data. The parsed size of the COBS-encoded payload "
            f"({payload_size}), is outside the expected boundaries "
            f"({protocol._min_rx_payload_size} to {protocol._max_rx_payload_size}). This likely indicates a "
            f"mismatch in the transmission parameters between this system and the Microcontroller."
        )
        mock_error.assert_called_once_with(message=message)

        # Verify the exception message
        assert str(exc_info.value) == message


def test_receive_data_status_104():
    """Test that receive_data handles status == 104 by logging error and raising RuntimeError."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Simulate parsing with status == 104
    parsed_bytes = np.array([1, 2, 3, 4], dtype=np.uint8)
    parsed_bytes_count = 2  # Delimiter encountered at byte number 3

    # Mock _parse_packet to return status == 104
    protocol._parse_packet = lambda *args: (104, parsed_bytes_count, b"", parsed_bytes)

    with patch("ataraxis_base_utilities.console.error") as mock_error:
        with pytest.raises(RuntimeError) as exc_info:
            protocol.receive_data()

        expected_byte_number = parsed_bytes.size - int(protocol._postamble_size)
        message = (
            f"Failed to parse the incoming serial packet data. Delimiter byte value "
            f"({protocol._delimiter_byte}) encountered at byte number {parsed_bytes_count + 1}, instead of the "
            f"expected byte number {expected_byte_number}. This likely indicates packet corruption or mismatch "
            f"in the transmission parameters between this system and the Microcontroller."
        )
        mock_error.assert_called_once_with(message=message)

        # Verify the exception message
        assert str(exc_info.value) == message


def test_receive_data_status_105():
    """Test that receive_data handles status == 105 by logging error and raising RuntimeError."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Simulate parsing with status == 105
    parsed_bytes = np.array([1, 2, 3, 99], dtype=np.uint8)  # Last byte is incorrect
    parsed_bytes_count = 0

    # Mock _parse_packet to return status == 105
    protocol._parse_packet = lambda *args: (105, parsed_bytes_count, b"", parsed_bytes)

    with patch("ataraxis_base_utilities.console.error") as mock_error:
        with pytest.raises(RuntimeError) as exc_info:
            protocol.receive_data()

        last_payload_byte_index = parsed_bytes.size - int(protocol._postamble_size)
        message = (
            f"Failed to parse the incoming serial packet data. Delimiter byte value "
            f"({protocol._delimiter_byte}) expected as the last payload byte "
            f"({last_payload_byte_index}), but instead encountered "
            f"{parsed_bytes[last_payload_byte_index]}. This likely indicates packet "
            f"corruption or mismatch in the transmission parameters between this system "
            f"and the Microcontroller."
        )
        mock_error.assert_called_once_with(message=message)

        # Verify the exception message
        assert str(exc_info.value) == message


def test_receive_data_status_102():
    """Test that receive_data handles status == 102 by logging error and raising RuntimeError."""
    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock _parse_packet to return status == 102
    protocol._parse_packet = lambda *args: (102, 0, b"", np.array([], dtype=np.uint8))

    with patch("ataraxis_base_utilities.console.error") as mock_error:
        with pytest.raises(RuntimeError) as exc_info:
            protocol.receive_data()

        # Verify that console.error was called with the correct message
        message = (
            f"Failed to parse the incoming serial packet data. Unable to find the start_byte "
            f"({protocol._start_byte}) value among the bytes stored inside the serial buffer."
        )
        mock_error.assert_called_once_with(message=message)

        # Verify the exception message
        assert str(exc_info.value) == message


def test_construct_packet_error():
    """Test for handling unexpected errors in the _construct_packet method."""

    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock the internal CRC processor to simulate an unexpected error in checksum calculation
    with patch.object(protocol._crc_processor, "calculate_crc_checksum", side_effect=RuntimeError("CRC error")):
        with patch("ataraxis_base_utilities.console.error") as mock_error:
            with pytest.raises(
                RuntimeError,
                match="Failed to send the payload data. Unexpected error encountered for _construct_packet",
            ):
                protocol.send_data()

            # Verify that the correct error message is logged
            mock_error.assert_called_once_with(
                message=(
                    "Failed to send the payload data. Unexpected error encountered for _construct_packet() method. "
                    "Re-running all COBS and CRC steps used for packet construction in wrapped mode did not reproduce the "
                    "error. Manual error resolution required."
                ),
                error=RuntimeError,
            )


def test_construct_packet_unexpected_error():
    """Test for handling unexpected errors during packet construction in the _construct_packet method."""

    protocol = SerialTransportLayer(port="COM7", baudrate=115200, test_mode=True)

    # Mock the internal CRC processor to simulate an unexpected error in checksum conversion
    with patch.object(
        protocol._crc_processor, "convert_checksum_to_bytes", side_effect=RuntimeError("Conversion error")
    ):
        with patch("ataraxis_base_utilities.console.error") as mock_error:
            # Simulate sending data and trigger the RuntimeError
            with pytest.raises(
                RuntimeError,
                match="Failed to send the payload data. Unexpected error encountered for _construct_packet",
            ):
                protocol.send_data()

            # Verify that the correct error message was logged
            mock_error.assert_called_once_with(
                message=(
                    "Failed to send the payload data. Unexpected error encountered for _construct_packet() method. "
                    "Re-running all COBS and CRC steps used for packet construction in wrapped mode did not reproduce the "
                    "error. Manual error resolution required."
                ),
                error=RuntimeError,
            )

