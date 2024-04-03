// Due to certain issues with reconnecting to teensy boards for running separate test suits, this test suit acts as a
// single centralized hub for running all available tests for all supported classes and methods of the
// SerializedTransferProtocol library. Declare all required tests using separate functions (as needed) and then add the
// tests to be evaluated to the RunUnityTests function at the bottom of this file. Comment unused tests out if needed.

// Dependencies
#include <Arduino.h>                       // For Arduino functions
#include <unity.h>                         // This is the C testing framework, no connection to the Unity game engine
#include "cobs_processor.h"                // COBSProcessor class
#include "crc_processor.h"                 // CRCProcessor class
#include "serialized_transfer_protocol.h"  // SerializedTransferProtocol class
#include "stream_mock.h"                   // StreamMock class required for SerializedTransferProtocol class testing

// This function is called automatically before each test function. Currently not used.
void setUp(void)
{}

// This function is called automatically after each test function. Currently not used.
void tearDown(void)
{}

// Tests COBSProcessor EncodePayload() and DecodePayload() methods.
void TestCOBSProcessor(void)
{
    // Prepares test assets
    uint8_t payload_buffer[256];                         // Initializes test buffer
    memset(payload_buffer, 22, sizeof(payload_buffer));  // Sets all values to 22
    COBSProcessor cobs_processor;                        // Instantiates class object to be tested

    // Creates a test payload using the format: overhead [0], payload [1 to 10], delimiter [11]
    uint8_t initial_packet[12] = {0, 10, 0, 0, 20, 0, 0, 0, 143, 12, 54, 22};
    memcpy(payload_buffer, initial_packet, sizeof(initial_packet));  // Copies the payload into the buffer

    // Expected packet (overhead + payload + delimiter) after encoding. Used to test encoding result
    uint8_t encoded_packet[12] = {2, 10, 1, 2, 20, 1, 1, 4, 143, 12, 54, 0};

    // Expected state of the packet after decoding. They payload is reverted to original
    // state, the overflow is reset to 0, but delimiter byte is not changed. Used to test the decoding result.
    uint8_t decoded_packet[12] = {0, 10, 0, 0, 20, 0, 0, 0, 143, 12, 54, 0};

    uint8_t payload_size         = 10;    // Tested payload size, for payload generated above
    uint8_t packet_size          = 12;    // Tested packet size, for the decoder test
    uint8_t delimiter_byte_value = 0x00;  // Tested delimiter byte value, uses the preferred default of 0

    // Verifies the unencoded packet matches pre-test expectations
    TEST_ASSERT_EQUAL_UINT8_ARRAY(initial_packet, payload_buffer, 11);

    // Verifies that the cobs_status is initialized to the expected standby value
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kStandby),
        cobs_processor.cobs_status
    );

    // Encodes test payload
    uint16_t encoded_size = cobs_processor.EncodePayload(payload_buffer, payload_size, delimiter_byte_value);

    // Verifies the encoding runtime status
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPayloadEncoded),
        cobs_processor.cobs_status
    );

    // Verifies that encoding returned expected payload size (10) + overhead + delimiter (== 12, packet size)
    TEST_ASSERT_EQUAL_UINT16(packet_size, encoded_size);

    // Verifies that the encoded payload matches the expected encoding outcome
    TEST_ASSERT_EQUAL_UINT8_ARRAY(encoded_packet, payload_buffer, 11);

    // Decodes test payload
    uint16_t decoded_size = cobs_processor.DecodePayload(payload_buffer, packet_size, delimiter_byte_value);

    // Verifies the decoding runtime status
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPayloadDecoded),
        cobs_processor.cobs_status
    );

    // Checks that size correctly equals to packet_size - 2 (10, payload_size).
    TEST_ASSERT_EQUAL_UINT16(payload_size, decoded_size);

    // Verifies that decoding reverses the payload back to the original state. Note, this excludes the overhead and
    // the delimiter, as the decoding operation does not alter these values (hence the use of a separate tester array)
    TEST_ASSERT_EQUAL_UINT8_ARRAY(decoded_packet, payload_buffer, 11);

    // Verifies that the non-payload portion of the buffer was not affected by the encoding/decoding cycles
    for (uint16_t i = 12; i < sizeof(payload_buffer); i++)
    {
        // Uses a custom message system similar to Unity Array check to provide the filed index number
        char message[50];  // Buffer for the failure message
        snprintf(message, sizeof(message), "Check failed at index: %d", i);
        TEST_ASSERT_EQUAL_UINT8_MESSAGE(22, payload_buffer[i], message);
    }
}

// Tests error handling for EncodePayload() and DecodePayload() COBSProcessor methods.
void TestCOBSProcessorErrors(void)
{
    // Generates test buffer and sets every value inside to 22
    uint8_t payload_buffer[256];
    memset(payload_buffer, 22, sizeof(payload_buffer));
    payload_buffer[0] = 0;  // Resets the overhead placeholder to 0, otherwise the encoding attempt below will fail

    // Instantiates class object to be tested
    COBSProcessor cobs_processor;

    // Verifies minimum encoding and decoding payload / packet size ranges. Uses standard global buffer of size 256
    // with all values set to 22. Takes ranges from the kCOBSProcessorCodes enumerator class to benefit from the fact
    // all hard-coded settings are centralized and can be modified from one place without separately tweaking source and
    // test code.

    // Verifies that payloads with minimal size are encoded correctly
    uint16_t result = cobs_processor.EncodePayload(
        payload_buffer,
        static_cast<uint8_t>(COBSProcessor::kCOBSProcessorParameters::kMinPayloadSize),
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPayloadEncoded),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMinPacketSize), result);

    // Verifies packets with minimal size are decoded correctly
    result = cobs_processor.DecodePayload(
        payload_buffer,
        static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMinPacketSize),
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPayloadDecoded),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMinPayloadSize), result);

    // Verifies that payloads with maximal size are encoded correctly
    result = cobs_processor.EncodePayload(
        payload_buffer,
        static_cast<uint8_t>(COBSProcessor::kCOBSProcessorParameters::kMaxPayloadSize),
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPayloadEncoded),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMaxPacketSize), result);

    // Verifies that packets with maximal size are decoded correctly
    result = cobs_processor.DecodePayload(
        payload_buffer,
        static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMaxPacketSize),
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPayloadDecoded),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMaxPayloadSize), result);

    // Verifies that unsupported (too high / too low) ranges give expected error codes that can be decoded using the
    // enumerator class. To do so, shifts the payload/packet size 1 value above or below the limit and tests for the
    // correct returned error code.

    // Tests too small payload size encoder error
    result = cobs_processor.EncodePayload(
        payload_buffer,
        static_cast<uint8_t>(COBSProcessor::kCOBSProcessorParameters::kMinPayloadSize) - 1,
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kEncoderTooSmallPayloadSize),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Tests too large payload size encoder error
    result = cobs_processor.EncodePayload(
        payload_buffer,
        static_cast<uint8_t>(COBSProcessor::kCOBSProcessorParameters::kMaxPayloadSize) + 1,
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kEncoderTooLargePayloadSize),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Tests too small packet size decoder error
    result = cobs_processor.DecodePayload(
        payload_buffer,
        static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMinPacketSize) - 1,
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kDecoderTooSmallPacketSize),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Tests too large packet size decoder error
    result = cobs_processor.DecodePayload(
        payload_buffer,
        static_cast<uint16_t>(COBSProcessor::kCOBSProcessorParameters::kMaxPacketSize) + 1,
        0
    );
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kDecoderTooLargePacketSize),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Tests decoder payload (in)validation error codes, issued whenever the payload does not conform to the format
    // expected from COBS encoding. During runtime, the code assumes that the packages were properly encoded using the
    // COBSProcessor class and, therefore, any deviation from the expected format is due to the payload or packet being
    // corrupted during transmission or CRC checking.

    // Resets the shared buffer to default state before running the test to exclude any confounds from the tests above
    memset(payload_buffer, 22, sizeof(payload_buffer));
    payload_buffer[0] = 0;  // Sets the overhead placeholder to 0 which is required for encoding to work

    // Introduces 'jump' variables to be encoded by the call below (since 0 is the delimiter value to be encoded)
    payload_buffer[5]  = 0;
    payload_buffer[10] = 0;

    // Encodes the payload of size 15, inserting a delimiter (0) byte at index 16, generating a packet of size 17
    uint16_t encoded_size = cobs_processor.EncodePayload(payload_buffer, 15, 0);

    // Decodes the packet of size 13 (17-4), which is a valid size. The process should abort before the delimiter at
    // index 16 is reached with the appropriate error code. Tests both the error code and that the decoder that uses a
    // while loop exits the loop as expected instead of overwriting the 'out-of-limits' buffer memory.
    result = cobs_processor.DecodePayload(payload_buffer, 13, 0);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kDecoderUnableToFindDelimiter),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Overwrites encoded jump variable at index 10 with the actual delimiter value. This should trigger the decoder
    // loop to break early, and issue an error code, as it encountered the delimiter before it expected it based on the
    // input packet size
    payload_buffer[10] = 0;

    // Resets the overhead back to the correct value, since the decoder overwrites it to 0 on each call, even if the
    // call produces one of the 'malformed packet' errors
    payload_buffer[0] = 5;

    // Tests delimiter found too early error code
    result = cobs_processor.DecodePayload(payload_buffer, encoded_size, 0);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kDecoderDelimiterFoundTooEarly),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Tests that calling a decoder on a packet with overhead byte set to 0 produces the expected error code
    // In this particular case, the error would correctly prevent calling decoder on the same data twice.
    // Also ensure the error takes precedence over the kDecoderDelimiterFoundTooEarly error.
    result = cobs_processor.DecodePayload(payload_buffer, encoded_size, 0);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPacketAlreadyDecoded),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Tests that calling an encoder on a buffer with overhead placeholder not set to 0 produces an error
    payload_buffer[0] = 5;  // Resets the overhead byte to a non-0 value

    // Tests correct kPayloadAlreadyEncoded error
    result = cobs_processor.EncodePayload(payload_buffer, 15, 0);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kPayloadAlreadyEncoded),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Initializes a small test buffer to test buffer-size related errors
    uint8_t test_buffer[5] = {0, 0, 0, 0, 0};

    // Attempts to encode a payload with size 20 using a buffer with size 5. This is not allowed and should trigger an
    // error
    result = cobs_processor.EncodePayload(test_buffer, 20, 11);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kEncoderPacketLargerThanBuffer),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Same as above, but tests the error for the decoder function
    result = cobs_processor.DecodePayload(test_buffer, 20, 11);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kDecoderPacketLargerThanBuffer),
        cobs_processor.cobs_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);
}

// Tests 8-bit CRC GenerateCRCTable() method of CRCProcessor class.
// Verifies that the table generated programmatically using input polynomial parameters matches static reference values.
// For that, uses an external source to generate the test table. Specifically, https://crccalc.com/ was used here as it
// offers pregenerated lookup tables used by the calculator itself.
void TestCRCProcessorGenerateTable_CRC8(void)
{
    // CRC-8 Table (Polynomial 0x07)
    // Make sure your controller has enough memory for the tested and generated tables. Here, the controller needs to
    // have 256 bytes of memory to store both tables, which should be compatible with most existing boards, including
    // Arduino Uno.
    constexpr uint8_t test_crc_table[256] = {
        0x00, 0x07, 0x0E, 0x09, 0x1C, 0x1B, 0x12, 0x15, 0x38, 0x3F, 0x36, 0x31, 0x24, 0x23, 0x2A, 0x2D, 0x70, 0x77,
        0x7E, 0x79, 0x6C, 0x6B, 0x62, 0x65, 0x48, 0x4F, 0x46, 0x41, 0x54, 0x53, 0x5A, 0x5D, 0xE0, 0xE7, 0xEE, 0xE9,
        0xFC, 0xFB, 0xF2, 0xF5, 0xD8, 0xDF, 0xD6, 0xD1, 0xC4, 0xC3, 0xCA, 0xCD, 0x90, 0x97, 0x9E, 0x99, 0x8C, 0x8B,
        0x82, 0x85, 0xA8, 0xAF, 0xA6, 0xA1, 0xB4, 0xB3, 0xBA, 0xBD, 0xC7, 0xC0, 0xC9, 0xCE, 0xDB, 0xDC, 0xD5, 0xD2,
        0xFF, 0xF8, 0xF1, 0xF6, 0xE3, 0xE4, 0xED, 0xEA, 0xB7, 0xB0, 0xB9, 0xBE, 0xAB, 0xAC, 0xA5, 0xA2, 0x8F, 0x88,
        0x81, 0x86, 0x93, 0x94, 0x9D, 0x9A, 0x27, 0x20, 0x29, 0x2E, 0x3B, 0x3C, 0x35, 0x32, 0x1F, 0x18, 0x11, 0x16,
        0x03, 0x04, 0x0D, 0x0A, 0x57, 0x50, 0x59, 0x5E, 0x4B, 0x4C, 0x45, 0x42, 0x6F, 0x68, 0x61, 0x66, 0x73, 0x74,
        0x7D, 0x7A, 0x89, 0x8E, 0x87, 0x80, 0x95, 0x92, 0x9B, 0x9C, 0xB1, 0xB6, 0xBF, 0xB8, 0xAD, 0xAA, 0xA3, 0xA4,
        0xF9, 0xFE, 0xF7, 0xF0, 0xE5, 0xE2, 0xEB, 0xEC, 0xC1, 0xC6, 0xCF, 0xC8, 0xDD, 0xDA, 0xD3, 0xD4, 0x69, 0x6E,
        0x67, 0x60, 0x75, 0x72, 0x7B, 0x7C, 0x51, 0x56, 0x5F, 0x58, 0x4D, 0x4A, 0x43, 0x44, 0x19, 0x1E, 0x17, 0x10,
        0x05, 0x02, 0x0B, 0x0C, 0x21, 0x26, 0x2F, 0x28, 0x3D, 0x3A, 0x33, 0x34, 0x4E, 0x49, 0x40, 0x47, 0x52, 0x55,
        0x5C, 0x5B, 0x76, 0x71, 0x78, 0x7F, 0x6A, 0x6D, 0x64, 0x63, 0x3E, 0x39, 0x30, 0x37, 0x22, 0x25, 0x2C, 0x2B,
        0x06, 0x01, 0x08, 0x0F, 0x1A, 0x1D, 0x14, 0x13, 0xAE, 0xA9, 0xA0, 0xA7, 0xB2, 0xB5, 0xBC, 0xBB, 0x96, 0x91,
        0x98, 0x9F, 0x8A, 0x8D, 0x84, 0x83, 0xDE, 0xD9, 0xD0, 0xD7, 0xC2, 0xC5, 0xCC, 0xCB, 0xE6, 0xE1, 0xE8, 0xEF,
        0xFA, 0xFD, 0xF4, 0xF3,
    };

    // Instantiates a class object to be tested. The class constructor triggers the table generation function and fills
    // the class-specific public instance of crc_table with calculated CRC values.
    CRCProcessor<uint8_t> crc_processor(0x07, 0x00, 0x00);

    // Verifies that internally created CRC table matches the external table
    TEST_ASSERT_EQUAL_HEX8_ARRAY(test_crc_table, crc_processor.crc_table, 256);
}

// Tests 16-bit CRC GenerateCRCTable() method of CRCProcessor class.
// Verifies that the table generated programmatically using input polynomial parameters matches static reference values.
// For that, uses an external source to generate the test table. Specifically, https://crccalc.com/ was used here as it
// offers pregenerated lookup tables used by the calculator itself.
void TestCRCProcessorGenerateTable_CRC16(void)
{
    // CRC-16/CCITT-FALSE Table (Polynomial 0x1021)
    // Make sure your controller has enough memory for the tested and generated tables. Here, the controller needs to
    // have 1024 bytes of memory to store both tables, which will be a stretch for controllers like Arduino Uno (but
    // not more modern and advanced systems).
    constexpr uint16_t test_crc_table[256] = {
        0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7, 0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD,
        0xE1CE, 0xF1EF, 0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6, 0x9339, 0x8318, 0xB37B, 0xA35A,
        0xD3BD, 0xC39C, 0xF3FF, 0xE3DE, 0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485, 0xA56A, 0xB54B,
        0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D, 0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
        0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC, 0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861,
        0x2802, 0x3823, 0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B, 0x5AF5, 0x4AD4, 0x7AB7, 0x6A96,
        0x1A71, 0x0A50, 0x3A33, 0x2A12, 0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A, 0x6CA6, 0x7C87,
        0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41, 0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
        0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70, 0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A,
        0x9F59, 0x8F78, 0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F, 0x1080, 0x00A1, 0x30C2, 0x20E3,
        0x5004, 0x4025, 0x7046, 0x6067, 0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E, 0x02B1, 0x1290,
        0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256, 0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
        0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405, 0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E,
        0xC71D, 0xD73C, 0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634, 0xD94C, 0xC96D, 0xF90E, 0xE92F,
        0x99C8, 0x89E9, 0xB98A, 0xA9AB, 0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3, 0xCB7D, 0xDB5C,
        0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A, 0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
        0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9, 0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83,
        0x1CE0, 0x0CC1, 0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8, 0x6E17, 0x7E36, 0x4E55, 0x5E74,
        0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
    };

    // Instantiates a class object to be tested. The class constructor triggers the table generation function and fills
    // the class-specific public instance of crc_table with calculated CRC values.
    CRCProcessor<uint16_t> crc_processor(0x1021, 0xFFFF, 0x0000);

    // Verifies that internally created CRC table matches the external table
    TEST_ASSERT_EQUAL_HEX16_ARRAY(test_crc_table, crc_processor.crc_table, 256);
}

// Tests 32-bit CRC GenerateCRCTable() method of CRCProcessor class.
// Verifies that the table generated programmatically using input polynomial parameters matches static reference values.
// For that, uses an external source to generate the test table. Specifically, https://crccalc.com/ was used here as it
// offers pregenerated lookup tables used by the calculator itself.
void TestCRCProcessorGenerateTable_CRC32(void)
{
    // CRC-32/XFER Table (Polynomial 0x000000AF)
    // Make sure your controller has enough memory for the tested and generated tables. Here, the controller needs to
    // have 2048 bytes of memory to store both tables, which will be a stretch for controllers like Arduino Uno (but
    // not more modern and advanced systems).
    constexpr uint32_t test_crc_table[256] = {
        0x00000000, 0x000000AF, 0x0000015E, 0x000001F1, 0x000002BC, 0x00000213, 0x000003E2, 0x0000034D, 0x00000578,
        0x000005D7, 0x00000426, 0x00000489, 0x000007C4, 0x0000076B, 0x0000069A, 0x00000635, 0x00000AF0, 0x00000A5F,
        0x00000BAE, 0x00000B01, 0x0000084C, 0x000008E3, 0x00000912, 0x000009BD, 0x00000F88, 0x00000F27, 0x00000ED6,
        0x00000E79, 0x00000D34, 0x00000D9B, 0x00000C6A, 0x00000CC5, 0x000015E0, 0x0000154F, 0x000014BE, 0x00001411,
        0x0000175C, 0x000017F3, 0x00001602, 0x000016AD, 0x00001098, 0x00001037, 0x000011C6, 0x00001169, 0x00001224,
        0x0000128B, 0x0000137A, 0x000013D5, 0x00001F10, 0x00001FBF, 0x00001E4E, 0x00001EE1, 0x00001DAC, 0x00001D03,
        0x00001CF2, 0x00001C5D, 0x00001A68, 0x00001AC7, 0x00001B36, 0x00001B99, 0x000018D4, 0x0000187B, 0x0000198A,
        0x00001925, 0x00002BC0, 0x00002B6F, 0x00002A9E, 0x00002A31, 0x0000297C, 0x000029D3, 0x00002822, 0x0000288D,
        0x00002EB8, 0x00002E17, 0x00002FE6, 0x00002F49, 0x00002C04, 0x00002CAB, 0x00002D5A, 0x00002DF5, 0x00002130,
        0x0000219F, 0x0000206E, 0x000020C1, 0x0000238C, 0x00002323, 0x000022D2, 0x0000227D, 0x00002448, 0x000024E7,
        0x00002516, 0x000025B9, 0x000026F4, 0x0000265B, 0x000027AA, 0x00002705, 0x00003E20, 0x00003E8F, 0x00003F7E,
        0x00003FD1, 0x00003C9C, 0x00003C33, 0x00003DC2, 0x00003D6D, 0x00003B58, 0x00003BF7, 0x00003A06, 0x00003AA9,
        0x000039E4, 0x0000394B, 0x000038BA, 0x00003815, 0x000034D0, 0x0000347F, 0x0000358E, 0x00003521, 0x0000366C,
        0x000036C3, 0x00003732, 0x0000379D, 0x000031A8, 0x00003107, 0x000030F6, 0x00003059, 0x00003314, 0x000033BB,
        0x0000324A, 0x000032E5, 0x00005780, 0x0000572F, 0x000056DE, 0x00005671, 0x0000553C, 0x00005593, 0x00005462,
        0x000054CD, 0x000052F8, 0x00005257, 0x000053A6, 0x00005309, 0x00005044, 0x000050EB, 0x0000511A, 0x000051B5,
        0x00005D70, 0x00005DDF, 0x00005C2E, 0x00005C81, 0x00005FCC, 0x00005F63, 0x00005E92, 0x00005E3D, 0x00005808,
        0x000058A7, 0x00005956, 0x000059F9, 0x00005AB4, 0x00005A1B, 0x00005BEA, 0x00005B45, 0x00004260, 0x000042CF,
        0x0000433E, 0x00004391, 0x000040DC, 0x00004073, 0x00004182, 0x0000412D, 0x00004718, 0x000047B7, 0x00004646,
        0x000046E9, 0x000045A4, 0x0000450B, 0x000044FA, 0x00004455, 0x00004890, 0x0000483F, 0x000049CE, 0x00004961,
        0x00004A2C, 0x00004A83, 0x00004B72, 0x00004BDD, 0x00004DE8, 0x00004D47, 0x00004CB6, 0x00004C19, 0x00004F54,
        0x00004FFB, 0x00004E0A, 0x00004EA5, 0x00007C40, 0x00007CEF, 0x00007D1E, 0x00007DB1, 0x00007EFC, 0x00007E53,
        0x00007FA2, 0x00007F0D, 0x00007938, 0x00007997, 0x00007866, 0x000078C9, 0x00007B84, 0x00007B2B, 0x00007ADA,
        0x00007A75, 0x000076B0, 0x0000761F, 0x000077EE, 0x00007741, 0x0000740C, 0x000074A3, 0x00007552, 0x000075FD,
        0x000073C8, 0x00007367, 0x00007296, 0x00007239, 0x00007174, 0x000071DB, 0x0000702A, 0x00007085, 0x000069A0,
        0x0000690F, 0x000068FE, 0x00006851, 0x00006B1C, 0x00006BB3, 0x00006A42, 0x00006AED, 0x00006CD8, 0x00006C77,
        0x00006D86, 0x00006D29, 0x00006E64, 0x00006ECB, 0x00006F3A, 0x00006F95, 0x00006350, 0x000063FF, 0x0000620E,
        0x000062A1, 0x000061EC, 0x00006143, 0x000060B2, 0x0000601D, 0x00006628, 0x00006687, 0x00006776, 0x000067D9,
        0x00006494, 0x0000643B, 0x000065CA, 0x00006565,
    };

    // Instantiates a class object to be tested. The class constructor triggers the table generation function and fills
    // the class-specific public instance of crc_table with calculated CRC values.
    CRCProcessor<uint32_t> crc_processor(0x000000AF, 0x00000000, 0x00000000);

    // Verifies that internally created CRC table matches the external table
    TEST_ASSERT_EQUAL_HEX32_ARRAY(test_crc_table, crc_processor.crc_table, 256);
}

// Tests CRCProcessor class CalculatePacketCRCChecksum(), AddCRCChecksumToBuffer() and ReadCRCChecksumFromBuffer()
// methods. Relies on the TestCRCProcessorGenerateTable functions to verify lookup table generation for all supported
// CRCs prior to running this test. All tests here are calibrated for 16-bit 0x1021 polynomial and will not work for
// any other polynomial.
void TestCRCProcessor(void)
{
    // Generates the test buffer of size 8 with an example packet of size 6 and two placeholder values
    uint8_t test_packet[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x15, 0x00, 0x00};

    // Instantiates the class object to be tested, which also generates a crc_table.
    CRCProcessor<uint16_t> crc_processor(0x1021, 0xFFFF, 0x0000);

    // Verifies that the crc_status initializes to the expected value
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kStandby),
        crc_processor.crc_status
    );
    // Runs the checksum generation function on the test packet
    uint16_t result = crc_processor.CalculatePacketCRCChecksum(test_packet, 0, 6);

    // Verifies that the CRC checksum generator returns the expected number and status
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kCRCChecksumCalculated),
        crc_processor.crc_status
    );
    TEST_ASSERT_EQUAL_HEX16(0xF54E, result);

    // Stuffs the CRC checksum into the test buffer
    uint16_t buffer_size = crc_processor.AddCRCChecksumToBuffer(test_packet, 6, result);

    // Verifies that the addition function works as expected and returns the correct used size of the buffer and status
    // code
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kCRCChecksumAddedToBuffer),
        crc_processor.crc_status
    );
    TEST_ASSERT_EQUAL_UINT16(8, buffer_size);

    // Runs the checksum on the packet and the two CRC bytes appended to it
    result = crc_processor.CalculatePacketCRCChecksum(test_packet, 0, 8);

    // Ensures that including CRC checksum into the input buffer correctly returns 0. This is a standard property of
    // CRC checksums often used in-place of direct checksum comparison when CRC-verified payload is checked upon
    // reception for errors.
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kCRCChecksumCalculated),
        crc_processor.crc_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Extracts the CRC checksum from the buffer
    uint16_t extracted_checksum = crc_processor.ReadCRCChecksumFromBuffer(test_packet, 6);

    // Verifies that the checksum is correctly extracted from buffer using the expected value check and status check
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kCRCChecksumReadFromBuffer),
        crc_processor.crc_status
    );
    TEST_ASSERT_EQUAL_HEX16(0xF54E, extracted_checksum);
}

// Tests error handling for CalculatePacketCRCChecksum(), AddCRCChecksumToBuffer() and ReadCRCChecksumFromBuffer()
// of CRCProcessor class.
void TestCRCProcessorErrors(void)
{
    // Generates a small test buffer
    uint8_t test_buffer[5] = {0x01, 0x02, 0x03, 0x04, 0x05};

    // Instantiates the class object to be tested, which also generates a crc_table.
    CRCProcessor<uint16_t> crc_processor(0x1021, 0xFFFF, 0x0000);

    // Attempts to generate a CRC for the buffer above using an incorrect input packet_size of 11. Since this is smaller
    // than the buffer size of 5, the function should return 0 (default error return) and set the crc_status to an error
    // code (this is the critical part tested here).
    uint16_t checksum = crc_processor.CalculatePacketCRCChecksum(test_buffer, 0, 11);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kCalculateCRCChecksumBufferTooSmall),
        crc_processor.crc_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, checksum);

    // Generates the checksum of the test buffer using correct input parameters
    checksum = crc_processor.CalculatePacketCRCChecksum(test_buffer, 0, 5);

    // Verifies that the AddCRCChecksumToBuffer function raises the correct error if the input buffer size is too small
    // to accommodate the sufficient number of bytes to store the crc checksum starting at the start_index. Here, start
    // index of 4 is inside the buffer, but 2 bytes are needed for crc 16 checksum and index 5 is not available, leading
    // to an error.
    uint16_t result = crc_processor.AddCRCChecksumToBuffer(test_buffer, 4, checksum);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kAddCRCChecksumBufferTooSmall),
        crc_processor.crc_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);

    // Same as above, but for the GetCRCChecksumFromBuffer function (same idea, index 5 is needed, but is not available
    // to read the CRC from it).
    result = crc_processor.ReadCRCChecksumFromBuffer(test_buffer, 4);
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCRCProcessorCodes::kReadCRCChecksumBufferTooSmall),
        crc_processor.crc_status
    );
    TEST_ASSERT_EQUAL_UINT16(0, result);
}

// Tests that the StreamMock class methods function as expected. This is a fairly minor, but necessary test to carry out
// prior to testing major SerializedTransferProtocol methods.
void TestStreamMock(void)
{
    // Instantiates the StreamMock class object to be tested. StreamMock mimics the base Stream class, but exposes
    // rx/tx buffer for direct manipulation
    StreamMock stream;

    // Extracts stream buffer size to a local variable
    uint16_t stream_buffer_size = stream.buffer_size;

    // Initializes a buffer to store the test data. Has to initialize an input buffer using uint8_t and an output
    // buffer (for test stream buffers) using int16_t. This is an unfortunate consequence of how the mock class is
    // implemented to support the behavior of the prototype stream class.
    uint8_t test_array_in[10]  = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int16_t test_array_out[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Verifies that the buffers are initialized to expected values (0)
    for (uint16_t i = 0; i < stream_buffer_size; i++)
    {
        TEST_ASSERT_EQUAL_INT16(0, stream.rx_buffer[i]);
        TEST_ASSERT_EQUAL_INT16(0, stream.tx_buffer[i]);
    }

    // Tests available() method. It is expected to return the size of the buffer as the number of available bytes since
    // the buffers are initialized to 0, which is a valid byte-valued for this class.
    int32_t available_bytes = stream.available();  // Have to use int32 for type-safety as method returns plain int
    TEST_ASSERT_EQUAL_INT16(stream_buffer_size, available_bytes);

    // Tests write() method with array input, which transfers the data from the test array to the stream tx buffer
    int16_t data_written = static_cast<int16_t>(stream.write(test_array_in, sizeof(test_array_in)));

    // Verifies that the writing operation was successful
    TEST_ASSERT_EQUAL_INT16_ARRAY(test_array_out, stream.tx_buffer, data_written);  // Checks the tx_buffer state
    TEST_ASSERT_EQUAL_size_t(data_written, stream.tx_buffer_index);

    // Tests write() method using a single-byte input (verifies byte-wise buffer filling)
    int16_t byte_written = static_cast<int16_t>(stream.write(101));

    // Verifies that the addition was successful
    TEST_ASSERT_EQUAL_size_t(data_written + byte_written, stream.tx_buffer_index);
    TEST_ASSERT_EQUAL_INT16(101, stream.tx_buffer[stream.tx_buffer_index - 1]);

    // Tests reset() method, which sets booth buffers to -1 and sets the rx/tx buffer indices to 0
    stream.reset();

    // Verifies that the buffers have been reset to -1
    for (uint16_t i = 0; i < stream_buffer_size; i++)
    {
        TEST_ASSERT_EQUAL_INT16(-1, stream.rx_buffer[i]);
        TEST_ASSERT_EQUAL_INT16(-1, stream.tx_buffer[i]);
    }

    // Also verifies that the tx_index was reset to 0
    TEST_ASSERT_EQUAL_size_t(0, stream.tx_buffer_index);

    // Explicitly overwrites both buffers with test data
    for (uint16_t i = 0; i < sizeof(test_array_in); i++)
    {
        stream.rx_buffer[i] = test_array_out[i];
        stream.tx_buffer[i] = test_array_out[i];
    }

    // Tests flush() function, which, for the mock class, functions as a tx_buffer-specific reset
    stream.flush();

    // Verifies that the tx buffer has been reset to -1
    for (uint16_t i = 0; i < stream_buffer_size; i++)
    {
        TEST_ASSERT_EQUAL_INT16(-1, stream.tx_buffer[i]);
    }

    // Verifies that the flush() method did not modify the rx buffer
    TEST_ASSERT_EQUAL_INT16_ARRAY(test_array_out, stream.rx_buffer, sizeof(test_array_in));

    // Tests peek() method, which should return the value that the current rx_buffer index is pointing at
    int16_t peeked_value = stream.peek();

    // Verifies that the peeked value matches expected value written from the test_array (Should use index 0)
    TEST_ASSERT_EQUAL_INT16(test_array_out[stream.rx_buffer_index], peeked_value);

    // Also verifies that the operation does not consume the value by running it again, expecting the same value as
    // before as a response
    int16_t peeked_value_2 = stream.peek();
    TEST_ASSERT_EQUAL_INT16(peeked_value, peeked_value_2);

    // Tests read() method, which is used to read a byte value from the rx buffer and 'consume' it by advancing the
    // rx_buffer_index
    int16_t read_value = stream.read();

    // Verifies that the consumed value is equal to the expected value peeked above
    TEST_ASSERT_EQUAL_INT16(peeked_value, read_value);

    // Consumes the remaining valid data to reach the invalid portion of the rx buffer
    for (uint8_t i = stream.rx_buffer_index; i < sizeof(test_array_in); i++)
    {
        stream.read();
    }

    // Attempts to consume an invalid value (-1) from the rx_buffer. Attempting to consume an invalid value should
    // return - 1
    read_value = stream.read();

    // Verifies that the method returns -1 when attempting to read invalid data
    TEST_ASSERT_EQUAL_INT16(-1, read_value);

    // Also verifies that peek() method returns -1 when peeking invalid data
    peeked_value = stream.peek();
    TEST_ASSERT_EQUAL_INT16(-1, peeked_value);
}

// Tests WriteData() and ReadData() methods of SerializedTransferProtocol class. The test is performed as a cycle to
// allow reusing test assets. Tests writing and reading a structure, an array and a concrete value. Also, this is the
// only method that verifies that the class variables initialize to the expected constant values and that tests using
// different transmission and reception buffer sizes.
void TestSerializedTransferProtocolBufferManipulation(void)
{
    // Instantiates the mock serial class and the tested SerializedTransferProtocol class
    StreamMock mock_port;
    // Note, uses different maximum payload size for the Rx and Tx buffers
    SerializedTransferProtocol<uint16_t, 254, 160> protocol(mock_port, 0x1021, 0xFFFF, 0x0000, 129, 0, 20000, false);

    // Verifies the performance of payload and buffer size accessor (get) methods
    TEST_ASSERT_EQUAL_UINT16(254, protocol.get_maximum_tx_payload_size());
    TEST_ASSERT_EQUAL_UINT16(160, protocol.get_maximum_rx_payload_size());
    TEST_ASSERT_EQUAL_UINT16(258, protocol.get_tx_buffer_size());
    TEST_ASSERT_EQUAL_UINT16(164, protocol.get_rx_buffer_size());

    // Initializes two test and expected buffers to 0. Have to use two buffers due to using different sizes for
    // reception and transmission buffers
    static constexpr uint16_t tx_buffer_size   = protocol.get_tx_buffer_size();
    static constexpr uint16_t rx_buffer_size   = protocol.get_rx_buffer_size();
    uint8_t expected_tx_buffer[tx_buffer_size] = {0};  // Initializes to 0, which is the expected state of class buffers
    uint8_t expected_rx_buffer[rx_buffer_size] = {0};
    uint8_t test_tx_buffer[tx_buffer_size]     = {0};
    uint8_t test_rx_buffer[rx_buffer_size]     = {0};
    memset(test_tx_buffer, 0, tx_buffer_size);
    memset(test_rx_buffer, 0, rx_buffer_size);
    memset(expected_tx_buffer, 0, tx_buffer_size);
    memset(expected_rx_buffer, 0, rx_buffer_size);

    // Verifies class status, tracker and buffer variable initialization (all should initialize to predicted values)
    // Transmission Buffer
    protocol.CopyTxDataToBuffer(test_tx_buffer);  // Reads _transmission_buffer into the test buffer
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected_tx_buffer, test_tx_buffer, tx_buffer_size);
    protocol.CopyRxDataToBuffer(test_rx_buffer);  // Reads _reception_buffer into the test buffer
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected_rx_buffer, test_rx_buffer, rx_buffer_size);

    // Transfer Status
    uint8_t expected_code = static_cast<uint8_t>(stp_shared_assets::kSerializedTransferProtocolStatusCodes::kStandby);
    TEST_ASSERT_EQUAL_UINT8(expected_code, protocol.transfer_status);

    // Byte in Buffer trackers
    TEST_ASSERT_EQUAL_UINT16(0, protocol.get_bytes_in_transmission_buffer());
    TEST_ASSERT_EQUAL_UINT16(0, protocol.get_bytes_in_reception_buffer());

    // Instantiates some test objects to be written and read from the buffers
    struct TestStruct
    {
        uint8_t byte_value       = 122;
        uint16_t short_value     = 45631;
        uint32_t long_value      = 321123;
        int8_t signed_8b_value   = -55;
        int16_t signed_16b_value = -8213;
        int32_t signed_32b_value = -62312;
    } __attribute__((packed)) test_structure;

    uint16_t test_array[15] = {1, 2, 3, 4, 5, 6, 7, 8, 101, 256, 1234, 7834, 15643, 38123, 65321};
    int32_t test_value      = -765;

    // Writes test objects into the _transmission_buffer
    uint16_t next_index = 0;
    next_index          = protocol.WriteData(test_structure, next_index);
    next_index          = protocol.WriteData(test_array, next_index);
    next_index          = protocol.WriteData(test_value, next_index);

    // Verifies that the buffer status matches the expected status (bytes successfully written)
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kBytesWrittenToBuffer,
        protocol.transfer_status
    );

    // Verifies that transmission bytes tracker matches the value returned by the final write operation
    TEST_ASSERT_EQUAL_UINT16(next_index, protocol.get_bytes_in_transmission_buffer());

    // Verifies that the payload size tracker does not change if one of the already written bytes is overwritten and
    // keeps the same value as achieved by the chain of the write operations above
    uint16_t new_index = protocol.WriteData(test_structure, 0);  // Re-writes the structure to the same place
    TEST_ASSERT_NOT_EQUAL_UINT16(new_index, protocol.get_bytes_in_transmission_buffer());  // Should not be the same
    TEST_ASSERT_EQUAL_UINT16(next_index, protocol.get_bytes_in_transmission_buffer());     // Should be the same

    // Verifies that bytes' tracker matches the value expected given the byte-size of all written objects
    // Combines the sizes (in bytes) of all test objects to come up with the overall payload size
    uint16_t expected_bytes = sizeof(test_structure) + sizeof(test_array) + sizeof(test_value);
    TEST_ASSERT_EQUAL_UINT16(expected_bytes, protocol.get_bytes_in_transmission_buffer());

    // Checks that the _transmission_buffer itself is set to the expected values. For this, overwrites the initial
    // portion of the expected_tx_buffer with the expected values of the _transmission_buffer after data has been written
    // to it.
    expected_tx_buffer[0]  = 0;
    expected_tx_buffer[1]  = 122;
    expected_tx_buffer[2]  = 63;
    expected_tx_buffer[3]  = 178;
    expected_tx_buffer[4]  = 99;
    expected_tx_buffer[5]  = 230;
    expected_tx_buffer[6]  = 4;
    expected_tx_buffer[7]  = 0;
    expected_tx_buffer[8]  = 201;
    expected_tx_buffer[9]  = 235;
    expected_tx_buffer[10] = 223;
    expected_tx_buffer[11] = 152;
    expected_tx_buffer[12] = 12;
    expected_tx_buffer[13] = 255;
    expected_tx_buffer[14] = 255;
    expected_tx_buffer[15] = 1;
    expected_tx_buffer[16] = 0;
    expected_tx_buffer[17] = 2;
    expected_tx_buffer[18] = 0;
    expected_tx_buffer[19] = 3;
    expected_tx_buffer[20] = 0;
    expected_tx_buffer[21] = 4;
    expected_tx_buffer[22] = 0;
    expected_tx_buffer[23] = 5;
    expected_tx_buffer[24] = 0;
    expected_tx_buffer[25] = 6;
    expected_tx_buffer[26] = 0;
    expected_tx_buffer[27] = 7;
    expected_tx_buffer[28] = 0;
    expected_tx_buffer[29] = 8;
    expected_tx_buffer[30] = 0;
    expected_tx_buffer[31] = 101;
    expected_tx_buffer[32] = 0;
    expected_tx_buffer[33] = 0;
    expected_tx_buffer[34] = 1;
    expected_tx_buffer[35] = 210;
    expected_tx_buffer[36] = 4;
    expected_tx_buffer[37] = 154;
    expected_tx_buffer[38] = 30;
    expected_tx_buffer[39] = 27;
    expected_tx_buffer[40] = 61;
    expected_tx_buffer[41] = 235;
    expected_tx_buffer[42] = 148;
    expected_tx_buffer[43] = 41;
    expected_tx_buffer[44] = 255;
    expected_tx_buffer[45] = 3;
    expected_tx_buffer[46] = 253;
    expected_tx_buffer[47] = 255;
    expected_tx_buffer[48] = 255;
    protocol.CopyTxDataToBuffer(test_tx_buffer);  // Copies the _transmission_buffer contents to the test_buffer
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected_tx_buffer, test_tx_buffer, tx_buffer_size);

    // Initializes new test objects, sets all to 0, which is different from the originally used test object values
    struct TestStruct2
    {
        uint8_t byte_value       = 0;
        uint16_t short_value     = 0;
        uint32_t long_value      = 0;
        int8_t signed_8b_value   = 0;
        int16_t signed_16b_value = 0;
        int32_t signed_32b_value = 0;
    } __attribute__((packed)) test_structure_new;

    uint16_t test_array_new[15] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int32_t test_value_new      = 0;

    // Copies the contents of the _transmission_buffer to the _reception_buffer to test reception buffer manipulation
    // (reading)
    bool copied = protocol.CopyTxBufferPayloadToRxBuffer();
    TEST_ASSERT_TRUE(copied);

    // Reads the data from the _reception_buffer into the newly instantiated test objects, resetting them to the original
    // test object values
    uint16_t bytes_read = 0;
    bytes_read          = protocol.ReadData(test_structure_new, bytes_read);

    // Verifies that the bytes-read does NOT match reception bytes tracker, since bytes_in_reception_buffer is not
    // modified by the read method
    TEST_ASSERT_NOT_EQUAL_UINT16(bytes_read, protocol.get_bytes_in_reception_buffer());

    // Continues reading data from the _transmission_buffer
    bytes_read = protocol.ReadData(test_array_new, bytes_read);
    bytes_read = protocol.ReadData(test_value_new, bytes_read);

    // Now should be equal, as the whole payload has been effectively consumed
    TEST_ASSERT_EQUAL_UINT16(bytes_read, protocol.get_bytes_in_reception_buffer());

    // Verifies that the buffer status matches the expected status (bytes successfully read)
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kBytesReadFromBuffer,
        protocol.transfer_status
    );

    // Verifies that the objects read from the buffer are the same as the original objects:
    // Structure (tests field-wise)
    TEST_ASSERT_EQUAL_UINT8(test_structure.byte_value, test_structure_new.byte_value);
    TEST_ASSERT_EQUAL_UINT16(test_structure.short_value, test_structure_new.short_value);
    TEST_ASSERT_EQUAL_UINT32(test_structure.long_value, test_structure_new.long_value);
    TEST_ASSERT_EQUAL_INT8(test_structure.signed_8b_value, test_structure_new.signed_8b_value);
    TEST_ASSERT_EQUAL_INT16(test_structure.signed_16b_value, test_structure_new.signed_16b_value);
    TEST_ASSERT_EQUAL_INT32(test_structure.signed_32b_value, test_structure_new.signed_32b_value);

    // Array
    TEST_ASSERT_EQUAL_UINT16_ARRAY(test_array, test_array_new, 15);

    // Value
    TEST_ASSERT_EQUAL_INT32(test_value, test_value_new);

    // Verifies that the reception buffer (which is basically set to the _transmission_buffer state now) was not
    // altered by the read method runtime
    memcpy(expected_rx_buffer, expected_tx_buffer, rx_buffer_size);  // Copies expected values from tx to rx buffer
    protocol.CopyRxDataToBuffer(test_rx_buffer);  // Sets test_rx_buffer to the actual state of the _reception_buffer
    TEST_ASSERT_EQUAL_UINT8_ARRAY(expected_tx_buffer, test_rx_buffer, rx_buffer_size);
}

// Tests expected error handling by WriteData() and ReadData() methods of SerializedTransferProtocol class. This is a
// fairly minor function, as buffer reading and writing can only fail in a small subset of cases. Uses the same payload
// size for the _reception_buffer and the _transmission_buffer. Note, this function reserves a lot of memory for all of
// its buffers (> 2kB), so it is advised to disable it for lower-end boards like Uno.
void TestSerializedTransferProtocolBufferManipulationErrors(void)
{
    // Initializes the tested class
    StreamMock mock_port;
    // Uses identical rx and tx payload sizes
    SerializedTransferProtocol<uint16_t, 254, 254> protocol(mock_port, 0x1021, 0xFFFF, 0x0000, 129, 0, 20000, false);

    // Initializes a test variable
    uint8_t test_value = 223;

    // Verifies that writing the variable to the last valid index of the payload works as expected and returns a valid
    // payload size and status code
    uint16_t final_payload_index = protocol.WriteData(test_value, 254 - 1);
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kBytesWrittenToBuffer,
        protocol.transfer_status
    );

    // Verifies that attempting to write the variable to an index beyond the payload range results in an error
    uint16_t error_index = protocol.WriteData(test_value, final_payload_index);
    TEST_ASSERT_EQUAL_UINT16(0, error_index);
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kWritePayloadTooSmallError,
        protocol.transfer_status
    );

    // Copies the contents of the _transmission_buffer to the _reception_buffer to test reception buffer manipulation
    // (reading)
    bool copied = protocol.CopyTxBufferPayloadToRxBuffer();
    TEST_ASSERT_TRUE(copied);

    // Verifies that reading from the end of the payload functions as expected
    final_payload_index = protocol.ReadData(test_value, 254 - 1);
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kBytesReadFromBuffer,
        protocol.transfer_status
    );

    // Verifies that attempting to read from an index beyond the payload range results in an error
    error_index = protocol.ReadData(test_value, final_payload_index);
    TEST_ASSERT_EQUAL_UINT16(0, error_index);
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kReadPayloadTooSmallError,
        protocol.transfer_status
    );
}

// Tests major SendData() and ReceiveData() methods of the SerializedTransferProtocol class, alongside all used
// sub-methods (ParsePacket(), ValidatePacket(), ConstructPacket(), private for the class) and auxiliary methods
// (Available(), public, but not frequently used by-themselves). Note, assumes lower level tests have already verified
// the functionality of StreamMock and buffer manipulation methods, which are also used here to facilitate testing.
void TestSerializedTransferProtocolDataTransmission(void)
{
    // Initializes the tested class
    StreamMock mock_port;
    // Uses identical rx and tx payload sizes
    SerializedTransferProtocol<uint16_t, 254, 254> protocol(mock_port, 0x1021, 0xFFFF, 0x0000, 129, 0, 20000, false);

    // Instantiates separate instances of encoder classes used to verify processing results
    COBSProcessor cobs_class;
    // CRC settings HAVE to be the same as used by the SerializedTransferProtocol instance.
    CRCProcessor<uint16_t> crc_class = CRCProcessor<uint16_t>(0x1021, 0xFFFF, 0x0000);

    // Generates the test array to be packaged and 'sent'
    uint8_t test_array[10] = {1, 2, 3, 0, 0, 6, 0, 8, 0, 0};

    // Writes the package into the _transmission_buffer
    protocol.WriteData(test_array, 0);

    // Sends the payload to the Stream buffer. This step consists of first packaging the data inside the
    // payload sector of the _transmission_buffer using COBS encoding and running and saving a CRC checksum for the
    // packaged data into the postamble buffer (private). Then, the data is transferred into the Stream transmission
    // buffer in the following order: preamble with start byte, packetized data, postamble with the CRC checksum.
    // Provided all steps succeed, the method returns 'true'.
    bool sent_status = protocol.SendData();

    // Verifies that the data has been successfully sent to the Stream buffer
    TEST_ASSERT_TRUE(sent_status);
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kPacketSent,
        protocol.transfer_status
    );

    // Manually verifies the contents of the tx_buffer of the StreamMock class to confirm that the data has been
    // processed correctly:

    // Instantiates an array to simulate the _transmission_buffer after the data has been added to it
    uint8_t buffer_array[14] = {0, 1, 2, 3, 0, 0, 6, 0, 8, 0, 0, 0, 0, 0};

    // Simulates COBS encoding the buffer. Note, assumes COBSProcessor methods have been tested prior to running this
    // test. Specifically, targets the 10-value payload starting from index 1. Uses the same delimiter byte value as
    // does the serial protocol class
    uint16_t packet_size = cobs_class.EncodePayload(buffer_array, 10, 0);

    // Calculates the CRC for the COBS-encoded buffer. Also assumes that the CRCProcessor methods have been tested prior
    // to running this test. The CRC calculation includes the overhead byte, the encoded payload and the inserted
    // delimiter byte. Note, the returned checksum depends on the used polynomial type.
    uint16_t crc_checksum = crc_class.CalculatePacketCRCChecksum(buffer_array, 0, packet_size);

    // Adds the CRC to the end of the buffer
    crc_class.AddCRCChecksumToBuffer(buffer_array, packet_size, crc_checksum);

    // Verifies that the packet inside the Stream tx_buffer is the same as the packet simulated above. Has to use some
    // verification heuristics because the tx_buffer is an int16_t array and is assembled piecemeal. Specifically, it is
    // assembled using 2 write() calls that first add the StartByte and then the packet fused with the CRC checksum
    // postamble. These operations cannot be properly simulated here, so instead constant check values are used via a
    // specifically structured 'for'-loop.
    for (uint8_t i = 0; i < 15; i++)
    {
        // Extracts and casts each tx_buffer value from the native int16_t type to the uint8_t type that would be used
        // during production runtime (with the real Stream class)
        uint8_t checked_value = static_cast<uint8_t>(mock_port.tx_buffer[i]);

        // For the very first index inside the tx_buffer, checks the value against the constant start byte value
        if (i == 0)
        {
            TEST_ASSERT_EQUAL_UINT8(129, checked_value);
        }
        // For the second value, checks that is the expected fixed payload size (10, size of the test array)
        else if (i == 1)
        {
            TEST_ASSERT_EQUAL_UINT8(10, checked_value);
        }

        // For all subsequent values, checks i-2 index of the buffer_array against the i index of the tx_buffer. This is
        // to discount for the fact that buffer_array does not have the start_byte or payload_size, but contains the
        // same packet and postamble as the tx_buffer. This assumes that the postamble only contains the CRC-16
        // checksum.
        else
        {
            TEST_ASSERT_EQUAL_UINT8(buffer_array[i - 2], static_cast<uint8_t>(mock_port.tx_buffer[i]));
        }
    }

    // Copies the fully encoded package into the rx_buffer to simulate packet reception and test ReceiveData() method
    // Modifies the copied package to exclude the payload size, as it is only used in outgoing packets.
    mock_port.rx_buffer[0] = mock_port.tx_buffer[0];
    for (uint16_t i = 1; i < 16; i++)
    {
        mock_port.rx_buffer[i] = mock_port.tx_buffer[i + 1];  // +1 allows avoiding copying the payload_size
    }

    // Ensures that the overhead byte copied to the rx_buffer is not zero (that the packet is COBS-encoded).
    TEST_ASSERT_NOT_EQUAL_UINT16(mock_port.rx_buffer[1], 0);

    // Simulates data reception using the rx_buffer of the mock port. Data reception has two distinct sub-steps.
    // The first step is to parse the data from the reception buffer of the Stream class, which consists of byte-wise
    // reading (consuming) the data from the buffer while looking for certain patterns. See the method implementation
    // for more details, but it is sufficient to say that only packets with a certain structure will pass this step.
    // The second step consists of verifying the parsed packet by first passing it through a CRC check and then
    // COBS-decoding the payload out of the packet. If these steps succeed, the method returns 'true'.
    bool receive_status = protocol.ReceiveData();

    // Verifies that the data has been successfully received from the Stream buffer
    TEST_ASSERT_EQUAL_UINT8(
        stp_shared_assets::kSerializedTransferProtocolStatusCodes::kPacketReceived,
        protocol.transfer_status
    );
    TEST_ASSERT_TRUE(receive_status);

    // Verifies that internal class _reception_buffer tracker was set to the payload size
    TEST_ASSERT_EQUAL_UINT16(10, protocol.get_bytes_in_reception_buffer());

    // Verifies that the reverse-processed payload is the same as the original payload array. This is less involved than
    // the forward-conversion since there is no need to generate the CRC value or simulate COBS encoding here. This
    // assumes these methods have been fully tested prior to calling this test
    uint8_t decoded_array[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // Placeholder-initialized
    protocol.ReadData(decoded_array, 0);                         // Reads the data from _transmission_buffer

    // Verifies that the decoded payload fully matches the test payload array contents
    TEST_ASSERT_EQUAL_UINT8_ARRAY(test_array, decoded_array, sizeof(test_array));

    // Verifies that the minor Available() method works as expected. This method returns 'true' if data to parse is
    // available and 'false' otherwise. Since StreamMock class initializes its buffers with zeroes, which is a valid
    // data value, this method should return 'true' even after fully consuming the test payload.
    bool data_available = protocol.Available();
    TEST_ASSERT_TRUE(data_available);

    // Verifies that ResetReceptionBuffer() method works as expected. The reset method is expected to do two
    // things: reset the bytes_in_reception_buffer tracker to 0 and overwrite the overhead_byte placeholder of the
    // _reception_buffer with 0. Since the latter is done during decoding, only the first effect is evaluated below.
    protocol.ResetReceptionBuffer();
    TEST_ASSERT_EQUAL_UINT16(0, protocol.get_bytes_in_reception_buffer());

    // Also verifies ResetTransmissionBuffer() method
    protocol.ResetTransmissionBuffer();
    TEST_ASSERT_EQUAL_UINT16(0, protocol.get_bytes_in_transmission_buffer());

    // Fully resets the mock rx_buffer with -1, which is used as a stand-in for no available data. This is to test the
    // 'false' return portion of the Available() method.
    for (uint16_t i = 0; i < mock_port.buffer_size; i++)
    {
        mock_port.rx_buffer[i] = -1;
    }

    // Verifies that available() correctly returns 'false' if no data is actually available to be read from the
    // Stream class rx_buffer
    data_available = protocol.Available();
    TEST_ASSERT_FALSE(data_available);
}

// Tests the errors and, where applicable, edge cases associated with the SendData() and ReceiveData() methods of the
// SerializedTransferProtocol class. No auxiliary methods are tested here since they do not raise any errors. Note,
// focuses specifically on errors raised by SerializedTransferProtocol class methods, COBS and CRC errors should be
// tested by their respective test functions. Also, does not test errors that are generally impossible to encounter
// without modifying the class code, such as COBS encoding due to incorrect overhead placeholder value error.
// Note, to better accommodate testing on Uno boards, the CRC used for these tests is CRC8. This should not affect the
// tested logic, but will reduce the memory size reserved by these functions
void TestSerializedTransferProtocolDataTransmissionErrors(void)
{
    // Initializes the tested class
    StreamMock mock_port;
    SerializedTransferProtocol<uint16_t, 254, 254> protocol(mock_port, 0x07, 0x00, 0x00, 129, 0, 20000, false);

    // Instantiates crc encoder class separately to generate test data
    CRCProcessor<uint16_t> crc_class = CRCProcessor<uint16_t>(0x07, 0x00, 0x00);

    // Initializes a test payload
    uint8_t test_payload[10] = {1, 2, 3, 4, 0, 0, 7, 8, 9, 10};

    // After the recent class logic update, it is effectively impossible to encounter an error during data
    // sending as there are now local guards against every possible error engineered into the class itself or buffer
    // manipulation methods. As such, simply runs the method sequence here and moves on to testing reception, which
    // can run into errors.
    protocol.WriteData(test_payload);
    protocol.SendData();

    // Prepares to test ReceiveData() method
    // Instantiates component buffers
    uint8_t preamble[1]              = {129};
    uint8_t packet_and_postamble[14] = {5, 1, 2, 3, 4, 3, 6, 7, 3, 9, 10, 0, 0, 0};

    // Calculates and adds packet CRC checksum to the postamble section of the packet_and_postamble array
    uint16_t crc_checksum = crc_class.CalculatePacketCRCChecksum(packet_and_postamble, 0, 12);
    crc_class.AddCRCChecksumToBuffer(packet_and_postamble, 12, crc_checksum);

    // Resets mock port buffers, which sets every buffer variable to -1 to ensure that the buffers are fully cleared
    // before running the tests below
    mock_port.reset();

    // Writes the components to the mock class rx buffer to simulate data reception
    for (uint8_t i = 0; i < 15; i++)
    {
        // Preamble, first byte of the rx_buffer
        if (i == 0) mock_port.rx_buffer[i] = static_cast<int16_t>(preamble[0]);
        // Packet followed by the crc_checksum
        else mock_port.rx_buffer[i] = static_cast<int16_t>(packet_and_postamble[i - 1]);
    }

    // Verifies that the algorithm correctly handles missing start byte error. By default, the algorithm is configured
    // to treat these 'errors' as 'no bytes available for reading' status, which is a non-error status
    mock_port.rx_buffer[0] = 0;  // Removes the start byte
    protocol.ReceiveData();
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kSerializedTransferProtocolStatusCodes::kNoBytesToParseFromBuffer),
        protocol.transfer_status
    );
    mock_port.rx_buffer_index = 0;  // Resets readout index back to 0

    // Re-initializes the protocol class, now setting the flag to allow start byte errors. I know this is MAJORLY
    // wasteful as this reserves another 540 bytes of memory for little reason, but this is the price of late fixes to
    // class logic, I guess. Uno owners: feel free to comment this line and the test below out, these are the only
    // places 'new protocol class is used to make it easy to remove'
    SerializedTransferProtocol<uint16_t, 254, 254> new_protocol(mock_port, 0x07, 0x00, 0x00, 129, 0, 20000, true);

    // Verifies that when Start Bytes are enabled, the algorithm correctly returns the error code.
    new_protocol.ReceiveData();
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kSerializedTransferProtocolStatusCodes::kPacketStartByteNotFoundError),
        new_protocol.transfer_status
    );
    mock_port.rx_buffer[0]    = 129;  // Restores the start byte
    mock_port.rx_buffer_index = 0;    // Resets readout index back to 0

    // Sets the entire rx_buffer to valid non-delimiter byte-values for the test below to work, as it has to consume
    // most of the rx_buffer to run out of the _reception_buffer space of the SerializedTransferProtocol class.
    for (uint16_t i = 15; i < mock_port.buffer_size; i++)
    {
        mock_port.rx_buffer[i] = 11;
    }

    // Verifies that the algorithm correctly handles running out of buffer space due to the missing delimiter byte error
    // not stopping the method runtime at the intended time
    mock_port.rx_buffer[12] = 11;  // Removes the delimiter byte
    protocol.ReceiveData();
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kSerializedTransferProtocolStatusCodes::kPacketOutOfBufferSpaceError),
        protocol.transfer_status
    );
    mock_port.rx_buffer[12]   = 0;  // Restores the delimiter byte value
    mock_port.rx_buffer_index = 0;  // Resets readout index back to 0

    // Verifies that the algorithm correctly handles encountering no valid bytes for a long time as a stale packet
    // error. For that, inserts an invalid value in the middle of the packet, which will be interpreted as not receiving
    // data until he timeout guard kicks-in to break the stale runtime.
    mock_port.rx_buffer[5] = -1;  // Sets byte 5 to an 'invalid' value to simulate not receiving valid bytes at index 5
    protocol.ReceiveData();
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kSerializedTransferProtocolStatusCodes::kPacketTimeoutError),
        protocol.transfer_status
    );
    mock_port.rx_buffer[5]    = packet_and_postamble[4];  // Restores the invalidated byte back to the original value
    mock_port.rx_buffer_index = 0;                        // Resets readout index back to 0

    // Verifies that the algorithm correctly handles the reception staling due to not receiving valid postamble
    // bytes for a long time
    mock_port.rx_buffer[13] = -1;  // Uses as a 'no data' placeholder instead of the first postamble byte value
    protocol.ReceiveData();
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kSerializedTransferProtocolStatusCodes::kPostambleTimeoutError),
        protocol.transfer_status
    );
    // Note, does not restore the CRC byte here, as it is used for the test below
    mock_port.rx_buffer_index = 0;  // Resets readout index back to 0

    // Verifies that the algorithm correctly handles a CRC checksum error (indicates corrupted packets).
    mock_port.rx_buffer[13] = 123;  // Fake CRC byte, overwrites the -1 code from the test above
    protocol.ReceiveData();
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kSerializedTransferProtocolStatusCodes::kCRCCheckFailed),
        protocol.transfer_status
    );
    mock_port.rx_buffer[13]   = static_cast<int16_t>(packet_and_postamble[12]);  // Restores the CRC byte value
    mock_port.rx_buffer_index = 0;                                               // Resets readout index back to 0

    // Verifies that the errors originating from sub-methods (in this case, the COBSProcessor class method) are properly
    // raised and handled by the caller methods of the SerializedTransferProtocol class. Specifically, expects COBS errors
    // to be raised to the main class level and saved to the transfer_status variable. To facilitate this test, sets
    // the packet to a value below the minimal value supported by the COBSProcessor. Note! Assumes default COBSProcessor
    // limits are used here (3 byte minimum for any packet).
    uint8_t small_packet[3] = {0, 0, 0};
    crc_checksum            = crc_class.CalculatePacketCRCChecksum(small_packet, 0, 1);
    crc_class.AddCRCChecksumToBuffer(small_packet, 1, crc_checksum);
    for (uint8_t i = 0; i < 4; i++)
    {
        // Preamble, first byte of the rx_buffer
        if (i == 0) mock_port.rx_buffer[i] = static_cast<int16_t>(preamble[0]);
        // Packet followed by the crc_checksum
        else mock_port.rx_buffer[i] = static_cast<int16_t>(small_packet[i - 1]);
    }

    // Verifies that the process fails as expected. Note, the error code is derived from the kCOBSProcessorCodes
    // enumeration here!
    protocol.ReceiveData();
    TEST_ASSERT_EQUAL_UINT8(
        static_cast<uint8_t>(stp_shared_assets::kCOBSProcessorCodes::kDecoderTooSmallPacketSize),
        protocol.transfer_status
    );
    mock_port.rx_buffer_index = 0;  // Resets readout index back to 0
}

// Specifies the test functions to be executed and controls their runtime. Use this function to determine which tests
// are run during test runtime and in what order. Note, each test function is encapsulated and will run even if it
// depends on other test functions ran before it which fail the tests.
int RunUnityTests(void)
{
    UNITY_BEGIN();

    // COBS Processor
    RUN_TEST(TestCOBSProcessor);
    RUN_TEST(TestCOBSProcessorErrors);

    // CRC Processor
    RUN_TEST(TestCRCProcessorGenerateTable_CRC8);
    RUN_TEST(TestCRCProcessorGenerateTable_CRC16);

// This test requires at least 2048 bytes of RAM to work, so prevents it from being evaluated by boards like Arduino
// Uno. Specifically, uses a static 3kb RAM limit
#if !defined RAMEND >= 0x0BFF
    RUN_TEST(TestCRCProcessorGenerateTable_CRC32);
#endif
    RUN_TEST(TestCRCProcessor);
    RUN_TEST(TestCRCProcessorErrors);

    // Stream Mock
    RUN_TEST(TestStreamMock);

    // Serial Transfer Protocol Write / Read Data
    RUN_TEST(TestSerializedTransferProtocolBufferManipulation);
    RUN_TEST(TestSerializedTransferProtocolBufferManipulationErrors);

    // Serial Transfer Protocol Send / Receive Data
    RUN_TEST(TestSerializedTransferProtocolDataTransmission);
    RUN_TEST(TestSerializedTransferProtocolDataTransmissionErrors);

    return UNITY_END();
}

// This is necessary for the Arduino framework testing to work as expected, which includes teensy. All tests are
// run inside setup function as they are intended to be one-shot tests
void setup()
{
    // Waits ~5 seconds before the Unity test runner establishes connection with a board Serial interface. For teensy,
    // this is less important as instead of using a UART, it uses a USB interface which does not reset the board on
    // connection.
    Serial.begin(115200);
    delay(5000);

    // Runs the required tests
    RunUnityTests();

    // Stops the serial communication interface.
    Serial.end();
}

// Nothing here as all tests are done in a one-shot fashion using 'setup' function above
void loop()
{}
