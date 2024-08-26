"""This file contains the test functions used to verify the functioning and error handling of all classes available
through the helper_modules file. You can also use this file if you need more examples on how to use helper_modules
methods.
"""

import re
import textwrap

import numpy as np
import pytest

from src.ataraxis_transport_layer.helper_modules import COBSProcessor, CRCProcessor, SerialMock


def test_cobs_processor():
    """Tests normal functioning of encode_payload() and decode_payload() methods of the COBSProcessor class."""

    # Instantiates the tested class
    processor = COBSProcessor()
    payload = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    delimiter = np.uint8(0)

    # Tests successful payload encoding
    encoded_packet = processor.encode_payload(payload, delimiter)
    assert encoded_packet.tolist() == [6, 1, 2, 3, 4, 5, 0]

    # Tests successful packet decoding
    decoded_payload = processor.decode_payload(encoded_packet, delimiter)
    assert decoded_payload.tolist() == payload.tolist()


def test_cobs_processor_errors():
    """Tests error handling of encode_payload() and decode_payload() methods of the COBSProcessor class."""

    # Instantiates the tested class
    processor = COBSProcessor()
    payload = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
    delimiter = np.uint8(0)

    # Tests invalid payload array encoder input error
    invalid_input = None
    error_message = (
        f"A numpy ndarray payload expected, but instead encountered '{type(invalid_input)}' when encoding payload "
        f"using COBS scheme."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = processor.encode_payload(invalid_input, delimiter)

    # Tests invalid delimiter encoder input error
    invalid_input = None
    error_message = (
        f"A scalar numpy uint8 (byte) delimiter expected, but instead encountered '{type(invalid_input)}' when "
        f"encoding payload using COBS scheme."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = processor.encode_payload(payload, invalid_input)

    # Tests payload too small encoder error
    empty_payload = np.array([], dtype=np.uint8)
    error_message = (
        f"The size of the input payload ({empty_payload.size}) is too small to be encoded using COBS scheme. "
        f"A minimum size of {1} elements (bytes) is required. CODE: 12."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.encode_payload(empty_payload, delimiter)

    # Tests payload too large encoder error
    large_payload = np.ones(255, dtype=np.uint8)
    error_message = (
        f"The size of the input payload ({large_payload.size}) is too large to be encoded using COBS scheme. "
        f"A maximum size of {254} elements (bytes) is required. CODE: 13."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.encode_payload(large_payload, delimiter)

    # Tests incorrect payload datatype encoder error
    wrong_dtype_payload = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    error_message = (
        f"The datatype of the input payload to be encoded using COBS scheme ({wrong_dtype_payload.dtype}) is not "
        f"supported. Only uint8 (byte) numpy arrays are currently supported as payload inputs. CODE: 14."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.encode_payload(wrong_dtype_payload, delimiter)

    # Tests invalid packet array decoder input error
    invalid_input = None
    error_message = (
        f"A numpy ndarray packet expected, but instead encountered '{type(invalid_input)}' when decoding packet "
        f"using COBS scheme."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = processor.decode_payload(invalid_input, delimiter)

    # Tests invalid delimiter decoder input error
    invalid_input = None
    # Needed for this test specifically, expects that the 'error-free' runtime has been verified
    encoded_packet = processor.encode_payload(payload, delimiter)
    error_message = (
        f"A scalar numpy uint8 (byte) delimiter expected, but instead encountered '{type(invalid_input)}' when "
        f"decoding packet using COBS scheme."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = processor.decode_payload(encoded_packet, invalid_input)

    # Tests packet too small decoder error
    small_packet = np.array([1, 2], dtype=np.uint8)
    error_message = (
        f"The size of the input packet ({small_packet.size}) is too small to be decoded using COBS scheme. "
        f"A minimum size of {3} elements (bytes) is required. CODE: 16."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.decode_payload(small_packet, delimiter)

    # Tests packet too large decoder error
    large_packet = np.ones(257, dtype=np.uint8)
    error_message = (
        f"The size of the input packet ({large_packet.size}) is too large to be decoded using COBS scheme. "
        f"A maximum size of {256} elements (bytes) is required. CODE: 17."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.decode_payload(large_packet, delimiter)

    # Tests incorrect packet datatype decoder error
    wrong_dtype_packet = np.array([6, 1, 2, 3, 4, 5, 0], dtype=np.uint16)
    error_message = (
        f"The datatype of the input packet to be decoded using COBS scheme ({wrong_dtype_packet.dtype}) is not "
        f"supported. Only uint8 (byte) numpy arrays are currently supported as packet inputs. CODE: 20."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.decode_payload(wrong_dtype_packet, delimiter)

    # Tests packet decoder corruption error where an unencoded delimiter (0) is found before reaching the end of the
    # packet (delimiter_found_too_early_error)
    corrupted_packet = np.array([4, 1, 2, 3, 0, 5, 0], dtype=np.uint8)
    error_message = (
        f"Unencoded delimiter found before reaching the end of the packet during COBS-decoding sequence. "
        f"Packet is likely corrupted. CODE: 19."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.decode_payload(corrupted_packet, delimiter)

    # Tests packet decoder corruption error where the unencoded payload is not found at the end of the payload or, for
    # that matter, at all (delimiter_not_found_error)
    corrupted_packet = np.array([6, 1, 2, 3, 4, 5, 6], dtype=np.uint8)
    error_message = (
        f"Attempting to decode the packet using COBS scheme does not result in reaching the unencoded delimiter"
        f"at the end of the packet. This is either because the end-value is not an unencoded delimiter or "
        f"because the traversal process does not point at the final index of the packet. Packet is likely "
        f"corrupted. CODE: 18."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        _ = processor.decode_payload(corrupted_packet, delimiter)


def test_crc_processor_generate_table_crc_8():
    """Verifies that the generate_crc_table() method of the CRCProcessor class works for CRC8 polynomials."""
    # Defines crc-8 polynomial parameters and test values
    polynomial = np.uint8(0x07)
    initial_crc_value = np.uint8(0x00)
    final_xor_value = np.uint8(0x00)
    expected_array = [
        0x00,
        0x07,
        0x0E,
        0x09,
        0x1C,
        0x1B,
        0x12,
        0x15,
        0x38,
        0x3F,
        0x36,
        0x31,
        0x24,
        0x23,
        0x2A,
        0x2D,
        0x70,
        0x77,
        0x7E,
        0x79,
        0x6C,
        0x6B,
        0x62,
        0x65,
        0x48,
        0x4F,
        0x46,
        0x41,
        0x54,
        0x53,
        0x5A,
        0x5D,
        0xE0,
        0xE7,
        0xEE,
        0xE9,
        0xFC,
        0xFB,
        0xF2,
        0xF5,
        0xD8,
        0xDF,
        0xD6,
        0xD1,
        0xC4,
        0xC3,
        0xCA,
        0xCD,
        0x90,
        0x97,
        0x9E,
        0x99,
        0x8C,
        0x8B,
        0x82,
        0x85,
        0xA8,
        0xAF,
        0xA6,
        0xA1,
        0xB4,
        0xB3,
        0xBA,
        0xBD,
        0xC7,
        0xC0,
        0xC9,
        0xCE,
        0xDB,
        0xDC,
        0xD5,
        0xD2,
        0xFF,
        0xF8,
        0xF1,
        0xF6,
        0xE3,
        0xE4,
        0xED,
        0xEA,
        0xB7,
        0xB0,
        0xB9,
        0xBE,
        0xAB,
        0xAC,
        0xA5,
        0xA2,
        0x8F,
        0x88,
        0x81,
        0x86,
        0x93,
        0x94,
        0x9D,
        0x9A,
        0x27,
        0x20,
        0x29,
        0x2E,
        0x3B,
        0x3C,
        0x35,
        0x32,
        0x1F,
        0x18,
        0x11,
        0x16,
        0x03,
        0x04,
        0x0D,
        0x0A,
        0x57,
        0x50,
        0x59,
        0x5E,
        0x4B,
        0x4C,
        0x45,
        0x42,
        0x6F,
        0x68,
        0x61,
        0x66,
        0x73,
        0x74,
        0x7D,
        0x7A,
        0x89,
        0x8E,
        0x87,
        0x80,
        0x95,
        0x92,
        0x9B,
        0x9C,
        0xB1,
        0xB6,
        0xBF,
        0xB8,
        0xAD,
        0xAA,
        0xA3,
        0xA4,
        0xF9,
        0xFE,
        0xF7,
        0xF0,
        0xE5,
        0xE2,
        0xEB,
        0xEC,
        0xC1,
        0xC6,
        0xCF,
        0xC8,
        0xDD,
        0xDA,
        0xD3,
        0xD4,
        0x69,
        0x6E,
        0x67,
        0x60,
        0x75,
        0x72,
        0x7B,
        0x7C,
        0x51,
        0x56,
        0x5F,
        0x58,
        0x4D,
        0x4A,
        0x43,
        0x44,
        0x19,
        0x1E,
        0x17,
        0x10,
        0x05,
        0x02,
        0x0B,
        0x0C,
        0x21,
        0x26,
        0x2F,
        0x28,
        0x3D,
        0x3A,
        0x33,
        0x34,
        0x4E,
        0x49,
        0x40,
        0x47,
        0x52,
        0x55,
        0x5C,
        0x5B,
        0x76,
        0x71,
        0x78,
        0x7F,
        0x6A,
        0x6D,
        0x64,
        0x63,
        0x3E,
        0x39,
        0x30,
        0x37,
        0x22,
        0x25,
        0x2C,
        0x2B,
        0x06,
        0x01,
        0x08,
        0x0F,
        0x1A,
        0x1D,
        0x14,
        0x13,
        0xAE,
        0xA9,
        0xA0,
        0xA7,
        0xB2,
        0xB5,
        0xBC,
        0xBB,
        0x96,
        0x91,
        0x98,
        0x9F,
        0x8A,
        0x8D,
        0x84,
        0x83,
        0xDE,
        0xD9,
        0xD0,
        0xD7,
        0xC2,
        0xC5,
        0xCC,
        0xCB,
        0xE6,
        0xE1,
        0xE8,
        0xEF,
        0xFA,
        0xFD,
        0xF4,
        0xF3,
    ]

    # Instantiates CRCProcessor class using crc_8 polynomial
    crc_processor = CRCProcessor(
        polynomial=polynomial, initial_crc_value=initial_crc_value, final_xor_value=final_xor_value
    )

    # Verifies that the class generates the expected lookup table for the crc_8 polynomial
    assert np.array_equal(crc_processor.crc_table, np.array(expected_array, dtype=np.uint8))


def test_crc_processor_generate_table_crc_16():
    """Verifies that the generate_crc_table() method of the CRCProcessor class works for CRC16 polynomials."""
    # Defines crc-16 polynomial parameters and test values
    polynomial = np.uint16(0x1021)
    initial_crc_value = np.uint16(0xFFFF)
    final_xor_value = np.uint16(0x0000)
    expected_array = [
        0x0000,
        0x1021,
        0x2042,
        0x3063,
        0x4084,
        0x50A5,
        0x60C6,
        0x70E7,
        0x8108,
        0x9129,
        0xA14A,
        0xB16B,
        0xC18C,
        0xD1AD,
        0xE1CE,
        0xF1EF,
        0x1231,
        0x0210,
        0x3273,
        0x2252,
        0x52B5,
        0x4294,
        0x72F7,
        0x62D6,
        0x9339,
        0x8318,
        0xB37B,
        0xA35A,
        0xD3BD,
        0xC39C,
        0xF3FF,
        0xE3DE,
        0x2462,
        0x3443,
        0x0420,
        0x1401,
        0x64E6,
        0x74C7,
        0x44A4,
        0x5485,
        0xA56A,
        0xB54B,
        0x8528,
        0x9509,
        0xE5EE,
        0xF5CF,
        0xC5AC,
        0xD58D,
        0x3653,
        0x2672,
        0x1611,
        0x0630,
        0x76D7,
        0x66F6,
        0x5695,
        0x46B4,
        0xB75B,
        0xA77A,
        0x9719,
        0x8738,
        0xF7DF,
        0xE7FE,
        0xD79D,
        0xC7BC,
        0x48C4,
        0x58E5,
        0x6886,
        0x78A7,
        0x0840,
        0x1861,
        0x2802,
        0x3823,
        0xC9CC,
        0xD9ED,
        0xE98E,
        0xF9AF,
        0x8948,
        0x9969,
        0xA90A,
        0xB92B,
        0x5AF5,
        0x4AD4,
        0x7AB7,
        0x6A96,
        0x1A71,
        0x0A50,
        0x3A33,
        0x2A12,
        0xDBFD,
        0xCBDC,
        0xFBBF,
        0xEB9E,
        0x9B79,
        0x8B58,
        0xBB3B,
        0xAB1A,
        0x6CA6,
        0x7C87,
        0x4CE4,
        0x5CC5,
        0x2C22,
        0x3C03,
        0x0C60,
        0x1C41,
        0xEDAE,
        0xFD8F,
        0xCDEC,
        0xDDCD,
        0xAD2A,
        0xBD0B,
        0x8D68,
        0x9D49,
        0x7E97,
        0x6EB6,
        0x5ED5,
        0x4EF4,
        0x3E13,
        0x2E32,
        0x1E51,
        0x0E70,
        0xFF9F,
        0xEFBE,
        0xDFDD,
        0xCFFC,
        0xBF1B,
        0xAF3A,
        0x9F59,
        0x8F78,
        0x9188,
        0x81A9,
        0xB1CA,
        0xA1EB,
        0xD10C,
        0xC12D,
        0xF14E,
        0xE16F,
        0x1080,
        0x00A1,
        0x30C2,
        0x20E3,
        0x5004,
        0x4025,
        0x7046,
        0x6067,
        0x83B9,
        0x9398,
        0xA3FB,
        0xB3DA,
        0xC33D,
        0xD31C,
        0xE37F,
        0xF35E,
        0x02B1,
        0x1290,
        0x22F3,
        0x32D2,
        0x4235,
        0x5214,
        0x6277,
        0x7256,
        0xB5EA,
        0xA5CB,
        0x95A8,
        0x8589,
        0xF56E,
        0xE54F,
        0xD52C,
        0xC50D,
        0x34E2,
        0x24C3,
        0x14A0,
        0x0481,
        0x7466,
        0x6447,
        0x5424,
        0x4405,
        0xA7DB,
        0xB7FA,
        0x8799,
        0x97B8,
        0xE75F,
        0xF77E,
        0xC71D,
        0xD73C,
        0x26D3,
        0x36F2,
        0x0691,
        0x16B0,
        0x6657,
        0x7676,
        0x4615,
        0x5634,
        0xD94C,
        0xC96D,
        0xF90E,
        0xE92F,
        0x99C8,
        0x89E9,
        0xB98A,
        0xA9AB,
        0x5844,
        0x4865,
        0x7806,
        0x6827,
        0x18C0,
        0x08E1,
        0x3882,
        0x28A3,
        0xCB7D,
        0xDB5C,
        0xEB3F,
        0xFB1E,
        0x8BF9,
        0x9BD8,
        0xABBB,
        0xBB9A,
        0x4A75,
        0x5A54,
        0x6A37,
        0x7A16,
        0x0AF1,
        0x1AD0,
        0x2AB3,
        0x3A92,
        0xFD2E,
        0xED0F,
        0xDD6C,
        0xCD4D,
        0xBDAA,
        0xAD8B,
        0x9DE8,
        0x8DC9,
        0x7C26,
        0x6C07,
        0x5C64,
        0x4C45,
        0x3CA2,
        0x2C83,
        0x1CE0,
        0x0CC1,
        0xEF1F,
        0xFF3E,
        0xCF5D,
        0xDF7C,
        0xAF9B,
        0xBFBA,
        0x8FD9,
        0x9FF8,
        0x6E17,
        0x7E36,
        0x4E55,
        0x5E74,
        0x2E93,
        0x3EB2,
        0x0ED1,
        0x1EF0,
    ]

    # Instantiates CRCProcessor class using crc_16 polynomial
    crc_processor = CRCProcessor(
        polynomial=polynomial, initial_crc_value=initial_crc_value, final_xor_value=final_xor_value
    )

    # Verifies that the class generates the expected lookup table for the crc_16 polynomial
    assert np.array_equal(crc_processor.crc_table, np.array(expected_array, dtype=np.uint16))


def test_crc_processor_generate_table_crc_32():
    """Verifies that the generate_crc_table() method of the CRCProcessor class works for CRC32 polynomials."""
    # Defines crc-32 polynomial parameters and test values
    polynomial = np.uint32(0x000000AF)
    initial_crc_value = np.uint32(0x00000000)
    final_xor_value = np.uint32(0x00000000)
    expected_array = [
        0x00000000,
        0x000000AF,
        0x0000015E,
        0x000001F1,
        0x000002BC,
        0x00000213,
        0x000003E2,
        0x0000034D,
        0x00000578,
        0x000005D7,
        0x00000426,
        0x00000489,
        0x000007C4,
        0x0000076B,
        0x0000069A,
        0x00000635,
        0x00000AF0,
        0x00000A5F,
        0x00000BAE,
        0x00000B01,
        0x0000084C,
        0x000008E3,
        0x00000912,
        0x000009BD,
        0x00000F88,
        0x00000F27,
        0x00000ED6,
        0x00000E79,
        0x00000D34,
        0x00000D9B,
        0x00000C6A,
        0x00000CC5,
        0x000015E0,
        0x0000154F,
        0x000014BE,
        0x00001411,
        0x0000175C,
        0x000017F3,
        0x00001602,
        0x000016AD,
        0x00001098,
        0x00001037,
        0x000011C6,
        0x00001169,
        0x00001224,
        0x0000128B,
        0x0000137A,
        0x000013D5,
        0x00001F10,
        0x00001FBF,
        0x00001E4E,
        0x00001EE1,
        0x00001DAC,
        0x00001D03,
        0x00001CF2,
        0x00001C5D,
        0x00001A68,
        0x00001AC7,
        0x00001B36,
        0x00001B99,
        0x000018D4,
        0x0000187B,
        0x0000198A,
        0x00001925,
        0x00002BC0,
        0x00002B6F,
        0x00002A9E,
        0x00002A31,
        0x0000297C,
        0x000029D3,
        0x00002822,
        0x0000288D,
        0x00002EB8,
        0x00002E17,
        0x00002FE6,
        0x00002F49,
        0x00002C04,
        0x00002CAB,
        0x00002D5A,
        0x00002DF5,
        0x00002130,
        0x0000219F,
        0x0000206E,
        0x000020C1,
        0x0000238C,
        0x00002323,
        0x000022D2,
        0x0000227D,
        0x00002448,
        0x000024E7,
        0x00002516,
        0x000025B9,
        0x000026F4,
        0x0000265B,
        0x000027AA,
        0x00002705,
        0x00003E20,
        0x00003E8F,
        0x00003F7E,
        0x00003FD1,
        0x00003C9C,
        0x00003C33,
        0x00003DC2,
        0x00003D6D,
        0x00003B58,
        0x00003BF7,
        0x00003A06,
        0x00003AA9,
        0x000039E4,
        0x0000394B,
        0x000038BA,
        0x00003815,
        0x000034D0,
        0x0000347F,
        0x0000358E,
        0x00003521,
        0x0000366C,
        0x000036C3,
        0x00003732,
        0x0000379D,
        0x000031A8,
        0x00003107,
        0x000030F6,
        0x00003059,
        0x00003314,
        0x000033BB,
        0x0000324A,
        0x000032E5,
        0x00005780,
        0x0000572F,
        0x000056DE,
        0x00005671,
        0x0000553C,
        0x00005593,
        0x00005462,
        0x000054CD,
        0x000052F8,
        0x00005257,
        0x000053A6,
        0x00005309,
        0x00005044,
        0x000050EB,
        0x0000511A,
        0x000051B5,
        0x00005D70,
        0x00005DDF,
        0x00005C2E,
        0x00005C81,
        0x00005FCC,
        0x00005F63,
        0x00005E92,
        0x00005E3D,
        0x00005808,
        0x000058A7,
        0x00005956,
        0x000059F9,
        0x00005AB4,
        0x00005A1B,
        0x00005BEA,
        0x00005B45,
        0x00004260,
        0x000042CF,
        0x0000433E,
        0x00004391,
        0x000040DC,
        0x00004073,
        0x00004182,
        0x0000412D,
        0x00004718,
        0x000047B7,
        0x00004646,
        0x000046E9,
        0x000045A4,
        0x0000450B,
        0x000044FA,
        0x00004455,
        0x00004890,
        0x0000483F,
        0x000049CE,
        0x00004961,
        0x00004A2C,
        0x00004A83,
        0x00004B72,
        0x00004BDD,
        0x00004DE8,
        0x00004D47,
        0x00004CB6,
        0x00004C19,
        0x00004F54,
        0x00004FFB,
        0x00004E0A,
        0x00004EA5,
        0x00007C40,
        0x00007CEF,
        0x00007D1E,
        0x00007DB1,
        0x00007EFC,
        0x00007E53,
        0x00007FA2,
        0x00007F0D,
        0x00007938,
        0x00007997,
        0x00007866,
        0x000078C9,
        0x00007B84,
        0x00007B2B,
        0x00007ADA,
        0x00007A75,
        0x000076B0,
        0x0000761F,
        0x000077EE,
        0x00007741,
        0x0000740C,
        0x000074A3,
        0x00007552,
        0x000075FD,
        0x000073C8,
        0x00007367,
        0x00007296,
        0x00007239,
        0x00007174,
        0x000071DB,
        0x0000702A,
        0x00007085,
        0x000069A0,
        0x0000690F,
        0x000068FE,
        0x00006851,
        0x00006B1C,
        0x00006BB3,
        0x00006A42,
        0x00006AED,
        0x00006CD8,
        0x00006C77,
        0x00006D86,
        0x00006D29,
        0x00006E64,
        0x00006ECB,
        0x00006F3A,
        0x00006F95,
        0x00006350,
        0x000063FF,
        0x0000620E,
        0x000062A1,
        0x000061EC,
        0x00006143,
        0x000060B2,
        0x0000601D,
        0x00006628,
        0x00006687,
        0x00006776,
        0x000067D9,
        0x00006494,
        0x0000643B,
        0x000065CA,
        0x00006565,
    ]

    # Instantiates CRCProcessor class using crc_32 polynomial
    crc_processor = CRCProcessor(
        polynomial=polynomial, initial_crc_value=initial_crc_value, final_xor_value=final_xor_value
    )

    # Verifies that the class generates the expected lookup table for the crc_32 polynomial
    assert np.array_equal(crc_processor.crc_table, np.array(expected_array, dtype=np.uint32))


def test_crc_processor():
    """Tests the functioning of the CRCProcessor class calculate_packet_crc_checksum(), convert_crc_checksum_to_bytes()
    and convert_crc_checksum_to_integer() methods. All tests are using CRC16 polynomial."""

    # Define test data
    test_data_1 = np.array([0x01, 0x02, 0x03, 0x04, 0x05, 0x15], dtype=np.uint8)
    test_data_2 = np.array([0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80], dtype=np.uint8)
    test_data_3 = np.array([0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA], dtype=np.uint8)

    # Instantiates tested class
    polynomial = np.uint16(0x1021)
    initial_crc_value = np.uint16(0xFFFF)
    final_xor_value = np.uint16(0x0000)
    crc_processor = CRCProcessor(polynomial, initial_crc_value, final_xor_value)

    # Tests checksum calculation using 3 data seeds:
    # Test 1
    checksum = crc_processor.calculate_packet_crc_checksum(test_data_1)
    assert checksum == np.uint16(0xF54E)

    # Test 2
    checksum = crc_processor.calculate_packet_crc_checksum(test_data_2)
    assert checksum == np.uint16(0x2B19)

    # Test 3
    checksum = crc_processor.calculate_packet_crc_checksum(test_data_3)
    assert checksum == np.uint16(0xBA4F)
    assert isinstance(checksum, np.uint16)

    # Tests checksum integer-to-byte array conversion
    checksum = crc_processor.calculate_packet_crc_checksum(test_data_1)
    checksum_buffer = crc_processor.convert_crc_checksum_to_bytes(checksum)
    assert np.array_equal(checksum_buffer, np.array([245, 78], dtype=np.uint8))
    assert isinstance(checksum_buffer, np.ndarray)

    # Tests the checksum byte array to integer conversion (extraction from buffer)
    checksum = crc_processor.convert_crc_checksum_to_integer(checksum_buffer)
    assert checksum == np.uint16(0xF54E)
    assert isinstance(checksum, np.uint16)


def test_crc_processor_errors():
    """Tests error handling of calculate_packet_crc_checksum(), convert_crc_checksum_to_bytes() and
    convert_crc_checksum_to_integer() methods of the COBSProcessor class. Also test COBSProcessor initialization errors.
    """
    # Instantiates tested class
    polynomial = np.uint16(0x1021)
    initial_crc_value = np.uint16(0xFFFF)
    final_xor_value = np.uint16(0x0000)
    crc_processor = CRCProcessor(polynomial, initial_crc_value, final_xor_value)

    # Verifies that using unsupported datatype for the class initialization arguments correctly triggers appropriate
    # type errors:
    # Polynomial
    invalid_argument_type = None
    error_message = (
        f"Unsupported 'polynomial' argument type '{type(invalid_argument_type)}' encountered when instantiating "
        f"CRCProcessor class. Use numpy uint8, uint16, or uint32."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = CRCProcessor(invalid_argument_type, initial_crc_value, final_xor_value)

    # Initial CRC Value
    invalid_argument_type = None
    error_message = (
        f"Unsupported 'initial_crc_value' argument type {type(invalid_argument_type)} encountered when "
        f"instantiating CRCProcessor class. Use numpy uint8, uint16, or uint32."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = CRCProcessor(polynomial, invalid_argument_type, final_xor_value)

    # Final XOR Value
    invalid_argument_type = None
    error_message = (
        f"Unsupported 'final_xor_value' argument type {type(invalid_argument_type)} encountered when instantiating "
        f"CRCProcessor class. Use numpy uint8, uint16, or uint32."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = CRCProcessor(polynomial, initial_crc_value, invalid_argument_type)

    # Verifies that using valid, but non-matching types for the class initialization arguments correctly triggers
    # the appropriate type error:
    non_matching_argument = np.uint32(0xFFFF)  # Valid initial value and type, but does not match polynomial
    error_message = (
        "All arguments ('polynomial', 'initial_crc_value', 'final_xor_value') must have the same type when "
        "instantiating CRCProcessor class."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = CRCProcessor(polynomial, non_matching_argument, final_xor_value)

    # Tests invalid buffer input type checksum calculation error
    invalid_input = None
    error_message = (
        f"A uint8 numpy ndarray buffer expected, but instead encountered '{type(invalid_input)}' when calculating the "
        f"CRC checksum for the input buffer."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = crc_processor.calculate_packet_crc_checksum(invalid_input)

    # Tests invalid buffer datatype checksum calculation error
    invalid_buffer_type = np.array([0x01, 0x02, 0x03, 0x04, 0x05], dtype=np.uint16)
    error_message = (
        f"The datatype of the input buffer to be CRC-checksummed ({invalid_buffer_type.dtype}) is not supported. "
        f"Only uint8 (byte) numpy arrays are currently supported as buffer inputs. CODE: 52."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = crc_processor.calculate_packet_crc_checksum(invalid_buffer_type)

    # Tests invalid crc_checksum input type checksum to byte array conversion error
    invalid_input = None
    error_message = (
        f"A uint8, uint16 or uint32 crc_checksum expected, but instead encountered '{type(invalid_input)}', "
        f"when converting the unsigned integer CRC checksum to an array of bytes."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = crc_processor.convert_crc_checksum_to_bytes(invalid_input)

    # Tests invalid buffer input type checksum to integer conversion error
    invalid_input = None
    error_message = (
        f"A uint8 numpy ndarray buffer expected, but instead encountered '{type(invalid_input)}' type when converting "
        f"the array of CRC checksum bytes to the unsigned integer value."
    )
    with pytest.raises(
        TypeError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = crc_processor.convert_crc_checksum_to_integer(invalid_input)

    # Tests invalid buffer dtype checksum to integer conversion error
    invalid_buffer = np.array([0x01, 0x02, 0x03, 0x04, 0x05], dtype=np.uint16)
    error_message = (
        f"The datatype of the input buffer to be converted to the unsigned integer CRC checksum "
        f"({invalid_buffer.dtype}) is not supported. Only uint8 (byte) numpy arrays are currently supported as buffer "
        f"inputs. CODE: 55."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = crc_processor.convert_crc_checksum_to_integer(invalid_buffer)

    # Tests invalid buffer size checksum to integer conversion error
    invalid_buffer = np.array([0x01, 0x02, 0x03, 0x04, 0x05], dtype=np.uint8)  # Correct dtype this time
    error_message = (
        f"The byte-size of the input buffer to be converted to the unsigned integer CRC checksum "
        f"({invalid_buffer.size}) does not match the size required to represent the specified checksum datatype "
        f"({crc_processor.crc_byte_length}). CODE: 56."
    )
    with pytest.raises(
        ValueError,
        match=re.escape(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False)),
    ):
        # noinspection PyTypeChecker
        _ = crc_processor.convert_crc_checksum_to_integer(invalid_buffer)


def test_serial_mock():
    """Jointly tests the successful and error-handling functioning of all SerialMock class methods."""

    # Creates an instance of SerialMock to test
    mock_serial = SerialMock()

    # Tests class initialization
    assert not mock_serial.is_open
    assert mock_serial.tx_buffer == b""
    assert mock_serial.rx_buffer == b""

    # Tests open() method
    mock_serial.open()
    assert mock_serial.is_open

    # Tests close() method
    mock_serial.close()
    assert not mock_serial.is_open

    # Tests write() method
    mock_serial.open()
    mock_serial.write(b"Hello")
    assert mock_serial.tx_buffer == b"Hello"

    # Tests write() method with non-bytes data (expecting TypeError)
    with pytest.raises(TypeError):
        # noinspection PyTypeChecker
        mock_serial.write("Not bytes")

    # Tests read() method
    mock_serial.rx_buffer = b"World"
    data = mock_serial.read(3)
    assert data == b"Wor"
    assert mock_serial.rx_buffer == b"ld"

    # Tests reset_input_buffer() method
    mock_serial.reset_input_buffer()
    assert mock_serial.rx_buffer == b""

    # Tests reset_output_buffer() method
    mock_serial.reset_output_buffer()
    assert mock_serial.tx_buffer == b""

    # Tests in_waiting() method
    mock_serial.rx_buffer = b"Data"
    assert mock_serial.in_waiting == 4

    # Tests out_waiting() method
    mock_serial.tx_buffer = b"Output"
    assert mock_serial.out_waiting == 6

    # Tests methods when the port is not open (expecting Exception)
    mock_serial.close()
    with pytest.raises(Exception):
        mock_serial.write(b"Test")
    with pytest.raises(Exception):
        mock_serial.read()
    with pytest.raises(Exception):
        mock_serial.reset_input_buffer()
    with pytest.raises(Exception):
        mock_serial.reset_output_buffer()
