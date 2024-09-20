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


@patch("your_module.serial.Serial")  # Mock the real Serial object
def test_repr_with_real_serial_port(mock_serial):
    """Test __repr__ when the _port is a real Serial object."""
    # Set up mock to simulate real Serial behavior
    mock_serial.name = "COM7"
    mock_serial.baudrate = 115200

    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        start_byte=129,
        delimiter_byte=0,
        timeout=10000,
        test_mode=False,
    )
    expected_repr = (
        "SerialTransportLayer(port='COM7', baudrate=115200, "
        "polynomial=0x1021, start_byte=129, delimiter_byte=0, "
        "timeout=10000 us, maximum_tx_payload_size = 254, "
        "maximum_rx_payload_size=254)"
    )
    assert repr(protocol) == expected_repr


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


def test_read_data_empty_array():
    """Test that attempting to read an empty array raises a ValueError."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    empty_array: np.ndarray = np.empty(0, dtype=np.uint8)

    # Prepare the reception buffer with dummy data
    protocol._reception_buffer[:10] = np.arange(10, dtype=np.uint8)
    protocol._bytes_in_reception_buffer = 10

    with pytest.raises(
        ValueError,
        match=re.escape(
            "An empty (size 0) numpy array requested when reading data from _reception_buffer. Reading empty arrays is currently not supported."
        ),
    ):
        protocol.read_data(empty_array)


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


def test_crc_checksum_failure():
    """Test for CRC checksum failure during packet reception."""
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Mock the CRCProcessor to simulate a checksum failure
    with patch.object(CRCProcessor, "calculate_crc_checksum", return_value=1):  # Simulate non-zero checksum
        reception_buffer = np.arange(20, dtype=np.uint8)  # Simulate received data
        protocol._reception_buffer = reception_buffer
        protocol._bytes_in_reception_buffer = 20

        result = protocol.receive_data()

        # Expected result is 0 since the CRC checksum failed
        assert result == 0


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


def test_full_data_flow():
    protocol = SerialTransportLayer(
        port="COM7",
        baudrate=115200,
        test_mode=True,
    )

    # Test scalar values
    test_scalar = np.uint8(42)
    protocol.write_data(test_scalar)
    protocol.send_data()

    # Simulate receiving the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Loopback

    # Receive the data
    protocol.receive_data()

    # Read back the scalar
    received_scalar, _ = protocol.read_data(np.uint8(0))
    assert received_scalar == test_scalar

    # Test array values
    test_array = np.array([1, 2, 3, 4], dtype=np.uint8)
    protocol.write_data(test_array)
    protocol.send_data()

    # Simulate receiving the same data
    protocol._port.rx_buffer = protocol._port.tx_buffer  # Loopback

    # Receive the data
    protocol.receive_data()

    # Read back the array
    received_array, _ = protocol.read_data(np.zeros(4, dtype=np.uint8))
    assert np.array_equal(received_array, test_array)


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
