"""This file stores low-level helper modules that are used by the main SerializedTransferProtocol class to support its
runtime.

At this time, this file includes the following classes:
- COBSProcessor and CRCProcessor modules that are used to process the transferred data before and after transmission.
- SerialMock class used to test SerializedTransferProtocol's transmission and reception behavior.
- ElapsedTimer class, which is a custom perf_counter_ns-based implementation of elapsedMillis C++ library, used to time
serial packet reception delays.
- ZeroMQSerial class, which is a wrapper around the ZeroMQ communication protocol that allows interfacing with
non-microcontroller devices running the Python or C# version of this library.

All methods of CRCProcessor and COBSProcessor classes are implemented using Numba and Numpy to optimize runtime
execution speed. This allows compiling the classes to machine-code (via first translating them to C under-the-hood)
whenever they are first called by the main class and achieves execution speeds that are at least 5 times as fast as
the equivalent pure-python implementation. As a downside, many python features are not supported by numba optimized
classes and, to improve user experience, the compiled classes are wrapped into equally named pure-python classes to
provide a more standard python API. The API is primarily intended for unit testing and users who want to use any of
the helper modules without using the main SerializedTransferProtocol class.

The SerialMock class is a pure-python class whose main job is to 'overload' the methods of the pySerial's Serial
class so that SerializedTransferProtocol can be tested without a properly configured microcontroller (and also
without trying to guess which specific COM port (if any) is available for each user running the tests). It has no
practical use past that specific role.

The ZeroMQSerial class is provided as an ongoing attempt to unify all communication methods under the same convenient
abstraction offered through SerializedTransferProtocol class. ZeroMQ is a popular and, most importantly, lightweight,
fast and flexible protocol implemented in more or less any popular programming language. This makes it a perfect
candidate to serve both for inter-process and cross-machine communication. Overall, the idea is to have microcontrollers
talk to PC via Serial, which is by far the most efficient protocol and the PC to talk to other PCs / processes via the
ZeroMQ connection over TCP (or UDP, ZeroMQ can be flexibly configured to use any backbone).
"""

import textwrap
import time as tm
from typing import Any, Literal, Union, Optional
import zmq
import threading
from collections import deque
from serial import Serial

import numpy as np
from numba import njit, uint8, uint16, uint32
from numba.experimental import jitclass


class COBSProcessor:
    """Provides methods for encoding and decoding data using the Consistent Overhead Byte Stuffing (COBS) scheme.

    See the original paper for the details on COBS methodology and specific data packet layouts:
    S. Cheshire and M. Baker, "Consistent overhead byte stuffing," in IEEE/ACM Transactions on Networking, vol. 7,
    no. 2, pp. 159-172, April 1999, doi: 10.1109/90.769765.

    Notes:
        This class functions as a wrapper that provides a consistent Python API for the internal instance of a
        jit-compiled _COBSProcessor class. This allows achieving python-like experience when using the class while
        simultaneously benefiting from fast machine-compiled code generated through numba jit-optimization. The wrapper
        automatically converts internal class runtime status codes into exception error messages where appropriate to
        notify users about runtime errors.

        For the maximum execution speed, you can access the private methods directly (see SerializedTransferProtocol
        class), although this is highly discouraged.

    Attributes:
        __processor: The private instance of the jit-compiled _COBSProcessor class which actually does all the required
            computations. Private by design and should never be accessed directly.
    """

    def __init__(self) -> None:
        # Instantiates the inner COBS class using the proper spec template
        self.__processor = self.__make_cobs_processor_class()

    def __repr__(self) -> str:
        repr_message = (
            f"COBSProcessor(inner_status={self.__processor.status}, "
            f"max_payload_size={self.__processor.max_payload_size}, "
            f"min_payload_size={self.__processor.min_payload_size}, "
            f"max_packet_size={self.__processor.max_packet_size}, "
            f"min_packet_size={self.__processor.min_packet_size})"
        )
        return repr_message

    def encode_payload(self, payload: np.ndarray, delimiter: np.uint8 = 0) -> np.ndarray:
        """Encodes the input payload into a transmittable packet using COBS scheme.

        Eliminates all instances of the delimiter value from the payload by replacing every such value
        with the distance to the next consecutive instance or, if no more instances are discovered, to the end of the
        payload. Appends an overhead byte value to the beginning of the payload that stores the distance to the first
        instance of the (eliminated) delimiter value or the end of the payload. Appends an unencoded delimiter byte
        value to the end of the payload to mark the end of the packet.

        Notes:
            The encoding produces the following packet structure: [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            payload: The numpy array that stores the payload to be encoded using COBS scheme. Has to use uint8
                datatype and be between 1 and 254 bytes in length.
            delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter.

        Raises:
            TypeError: If the payload or delimiter arguments are not of a correct numpy datatype.

        Returns:
            The packet uint8 numpy array encoded using COBS scheme.
        """

        # Prevents using the method for unsupported input types
        if not isinstance(payload, np.ndarray):
            error_message = (
                f"A numpy ndarray payload expected, but instead encountered '{type(payload)}' when encoding payload "
                f"using COBS scheme."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif not isinstance(delimiter, np.uint8):
            error_message = (
                f"A scalar numpy uint8 (byte) delimiter expected, but instead encountered '{type(delimiter)}' when "
                f"encoding payload using COBS scheme."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Calls the encoding method
        packet = self.__processor.encode_payload(payload, delimiter)

        # Resolves method runtime status to see if an error exception needs to be raised or the packet has been encoded
        # and can be returned to caller
        self.__verify_encoding_outcome(payload)

        # If runtime was successful, returns the packet
        return packet

    def __verify_encoding_outcome(self, payload: np.ndarray) -> None:
        """Verifies that encode_payload() method runtime was successful. If not, raises the appropriate exception.

        Notes:
            The main reason for having this as a separate method is to allow verification following direct calls to
            inner methods that bypass the API wrapper methods.

        Args:
            payload: The input payload array. The payload is used to extract information like it's size and datatype
                properties to make the error messages more informative.

        Raises:
            ValueError: If the parameters of the input arguments, such as the size or the datatype of the payload
                array, are not valid.
            RuntimeError: If the status code returned by the encoder method is not one of the expected values.

        """

        # Success code, verification is complete
        if self.__processor.status == self.__processor.payload_encoded:
            pass

        # Payload too small
        elif self.__processor.status == self.__processor.payload_too_small_error:
            error_message = (
                f"The size of the input payload ({payload.size}) is too small to be encoded using COBS scheme. "
                f"A minimum size of {self.__processor.min_payload_size} elements (bytes) is required. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Payload too large
        elif self.__processor.status == self.__processor.payload_too_large_error:
            error_message = (
                f"The size of the input payload ({payload.size}) is too large to be encoded using COBS scheme. "
                f"A maximum size of {self.__processor.max_payload_size} elements (bytes) is required. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Invalid payload datatype
        elif self.__processor.status == self.__processor.invalid_payload_datatype_error:
            error_message = (
                f"The datatype of the input payload to be encoded using COBS scheme ({payload.dtype}) is not "
                f"supported. Only uint8 (byte) numpy arrays are currently supported as payload inputs. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Unknown status code
        else:
            error_message = (
                f"Unexpected inner _COBSProcessor class status code ({self.__processor.status}) encountered when "
                f"attempting to COBS-encode the input payload. "
                f"CODE: 0."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    def decode_payload(self, packet: np.ndarray, delimiter: np.uint8 = 0) -> np.ndarray:
        """Decodes the COBS-encoded payload from the input packet.

        Traverses the input packet by cyclically advancing by the number of indices obtained from the value of the
        variable evaluated at the beginning of each cycle, starting from the overhead bytes. For all variables other
        than the overhead and the delimiter bytes, overwrites them with the input delimiter value. Removes the overhead
        and the delimiter bytes once the payload has been decoded.

        Notes:
            This method doubles-up as packet corruption detector. Specifically, it expects that the input packet always
            ends with the unencoded delimiter and that there are no unencoded delimiter occurrences amongst the
            traversed variables. Any deviation from this expectation is interpreted as packet corruption.

            Expects the input packets to adhere to the following structure:
            [Overhead] ... [COBS Encoded Payload] ... [Delimiter].

        Args:
            packet: The numpy array that stores COBS-encoded packet. The array should be using uint8 datatype and has to
                be entirely filled with the packet data. The first index (0) should store the overhead byte, and the
                last valid index of the packet should store the unencoded delimiter. The packet should be between 3 and
                256 bytes in length.
            delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter. It is used to
                optimize the decoding flow and to verify the unencoded delimiter at the end of the packet.

        Returns:
            The payload uint8 numpy array decoded from the packet.

        Raises:
            TypeError: If the packet or delimiter arguments are not of a correct numpy datatype.
        """

        # Prevents using the method for unsupported input types
        if not isinstance(packet, np.ndarray):
            error_message = (
                f"A numpy ndarray packet expected, but instead encountered '{type(packet)}' when decoding packet "
                f"using COBS scheme."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif not isinstance(delimiter, np.uint8):
            error_message = (
                f"A scalar numpy uint8 (byte) delimiter expected, but instead encountered '{type(delimiter)}' when "
                f"decoding packet using COBS scheme."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Calls decoding method
        payload = self.__processor.decode_payload(packet, delimiter)

        # Verifies the outcome of decoding method runtime
        self.__verify_decoding_outcome(packet)

        # Returns the decoded payload to caller if verification was successful
        return payload

    def __verify_decoding_outcome(self, packet: np.ndarray) -> None:
        """Verifies that decode_payload() method runtime was successful. If not, raises the appropriate exception.

        Notes:
            The main reason for having this as a separate method is to allow verification following direct calls to
            inner methods that bypass the API wrapper methods.

        Args:
            packet: The input packet array. The packet is used to extract information like it's size and datatype
                properties to make the error messages more informative.

        Raises:
            ValueError: If the parameters of the input arguments, such as the size or the datatype of the packet
                array, are unsupported. Also, if the unencoded delimiter is not found at all or is found before reaching
                the end of the packet, which indicates a corrupted packet.
            RuntimeError: If the status code returned by the decoder method is not one of the expected values.

        """

        # Runtime successful, verification is complete
        if self.__processor.status == self.__processor.payload_decoded:
            pass

        # Packet too small
        elif self.__processor.status == self.__processor.packet_too_small_error:
            error_message = (
                f"The size of the input packet ({packet.size}) is too small to be decoded using COBS scheme. "
                f"A minimum size of {self.__processor.min_packet_size} elements (bytes) is required. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Packet too large
        elif self.__processor.status == self.__processor.packet_too_large_error:
            error_message = (
                f"The size of the input packet ({packet.size}) is too large to be decoded using COBS scheme. "
                f"A maximum size of {self.__processor.max_packet_size} elements (bytes) is required. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Invalid packet datatype
        elif self.__processor.status == self.__processor.invalid_packet_datatype_error:
            error_message = (
                f"The datatype of the input packet to be decoded using COBS scheme ({packet.dtype}) is not supported. "
                f"Only uint8 (byte) numpy arrays are currently supported as packet inputs. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Delimiter isn't found at the end of the packet or 'jumping' does not point at the end of teh packet. Indicates
        # packet corruption.
        elif self.__processor.status == self.__processor.delimiter_not_found_error:
            error_message = (
                f"Attempting to decode the packet using COBS scheme does not result in reaching the unencoded delimiter"
                f"at the end of the packet. This is either because the end-value is not an unencoded delimiter or "
                f"because the traversal process does not point at the final index of the packet. Packet is likely "
                f"corrupted. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Delimiter encountered before reaching the end of the packet. Indicates packet corruption.
        elif self.__processor.status == self.__processor.delimiter_found_too_early_error:
            error_message = (
                f"Unencoded delimiter found before reaching the end of the packet during COBS-decoding sequence. "
                f"Packet is likely corrupted. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Unknown code
        else:
            error_message = (
                f"Unexpected inner _COBSProcessor class status code ({self.__processor.status}) encountered when "
                f"attempting to COBS-decode the input packet. "
                f"CODE: 0."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    @property
    def processor(self):
        """Returns the private jit-compiled _COBSProcessor class instance. This accessor represents a convenient way of
        unwrapping the jit-compiled class, so that the fast methods can be used directly (helpful when you want to use
        them from another jit-method).
        """
        return self.__processor

    @staticmethod
    def __make_cobs_processor_class():
        """A template-like private method that instantiates and returns jit-compiled _COBSProcessor class object.

        Notes:
            Since COBSProcessor does not use any 'template' parameters, this function only exists to maintain
            algorithmic similarity with CRCProcessor class that uses template parameters.

            Contains the numba spec-file as internal module variable, which can be edited to control the datatypes
            used by numba to compile class attributes.

        Returns:
            A fully configured instance of the jit-compiled _COBSProcessor class.
        """

        # The template for the numba compiler to assign specific datatypes to variables used by the COBSProcessor class.
        # This is needed for Numba to properly compile the class to C.
        cobs_spec = [
            ("status", uint8),
            ("max_payload_size", uint8),
            ("min_payload_size", uint8),
            ("max_packet_size", uint16),
            ("min_packet_size", uint8),
            ("standby", uint8),
            ("payload_too_small_error", uint8),
            ("payload_too_large_error", uint8),
            ("invalid_payload_datatype_error", uint8),
            ("payload_encoded", uint8),
            ("packet_too_small_error", uint8),
            ("packet_too_large_error", uint8),
            ("delimiter_not_found_error", uint8),
            ("delimiter_found_too_early_error", uint8),
            ("invalid_packet_datatype_error", uint8),
            ("payload_decoded", uint8),
        ]

        # Spec has to be available at instantiation as numba needs this information to properly compile class. Hence,
        # the class is defined at the same level as the spec list.
        # members.
        @jitclass(cobs_spec)
        class _COBSProcessor:
            """The inner COBSProcessor class that actually implements all method logic.

            Notes:
                This class is optimized using Numba's JIT (Just-In-Time) compilation module to significantly improve
                the execution speed of all class methods. As an unfortunate side effect, this process interferes with
                Python's built-in error handling tools. To provide error-handling capacity, a C-driven approach of
                returning fixed byte error-codes has been implemented. The error codes are available through the class
                attributes below. Each method returns the status (success or error) code by setting the class 'status'
                attribute to the latest runtime code, mimicking the functioning of the class version intended for
                microcontrollers.

            Attributes:
                status: Dynamically updated during runtime to track the latest method's runtime integer byte-code.
                max_payload_size: The maximum size of the payload, in bytes. Due to COBS, cannot exceed 254 bytes.
                min_payload_size: The minimum size of the payload, in bytes. No algorithmic minimum enforced, but does
                    not make sense to have it less than 1 byte.
                max_packet_size: The maximum size of the packet, in bytes. Due to COBS, it cannot exceed 256 bytes
                    (254 payload bytes + 1 overhead + 1 delimiter byte).
                min_packet_size: The minimum size of the packet, in bytes. Due to COBSm cannot be below 3 bytes.
                standby: The integer code used during class initialization (before any method is called).
                payload_too_small_error: The input payload array's size was below the min_payload_size during encoding.
                payload_too_large_error: The input payload array's size was above the max_payload_size during encoding.
                invalid_payload_datatype_error: The input payload array's datatype was not valid for the encoding
                    method (not uint8).
                payload_encoded: Payload has been successfully encoded (into a packet).
                packet_too_small_error: The input packet array's size was below the min_packet_size during decoding.
                packet_too_large_error: The input packet array's size was above the max_packet_size during decoding.
                delimiter_not_found_error: The decoder method did not encounter an unencoded delimiter during its
                    runtime.
                delimiter_found_too_early_error: The decoder method encountered the unencoded delimiter before reaching
                    the end of the packet.
                invalid_packet_datatype_error: The input packet array's datatype was not valid for the decoder method
                    (not uint8).
                payload_decoded: Packet has been successfully decoded into payload.
            """

            def __init__(self) -> None:
                # Constant class parameters (do not modify, they are already optimal for any non-embedded system)
                self.max_payload_size = 254
                self.min_payload_size = 1
                self.max_packet_size = 256
                self.min_packet_size = 3

                # Status codes. Follow a similar approach as the microcontroller library where the codes are unique
                # across the entire library and staty in the range of 11 to 50.
                self.standby = 11
                self.payload_too_small_error = 12
                self.payload_too_large_error = 13
                self.invalid_payload_datatype_error = 14
                self.payload_encoded = 15
                self.packet_too_small_error = 16
                self.packet_too_large_error = 17
                self.delimiter_not_found_error = 18
                self.delimiter_found_too_early_error = 19
                self.invalid_packet_datatype_error = 20
                self.payload_decoded = 21

                self.status = self.standby  # Initializes to standby

            def encode_payload(self, payload: np.ndarray, delimiter: np.uint8 = 0) -> np.ndarray:
                """Encodes the input payload into a transmittable packet using COBS scheme.

                This is a jit method that is very fast but requires strict input / output typing as it is compiled to C.
                If possible, only use this method through a wrapper API to ensure proper error handling.

                Args:
                    payload: The numpy array that stores the payload to be encoded using COBS scheme. Has to use uint8
                        datatype and be between 1 and 254 bytes in length.
                    delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter.

                Returns:
                    The packet uint8 numpy array encoded using COBS scheme, if the method succeeds or an empty
                    uninitialized numpy array otherwise. Use the status code available through the 'status' attribute
                    of the class instance to determine if the method succeeded or failed (in this case, the code
                    provides specific error code).
                """

                # Saves payload size to a separate variable
                # noinspection DuplicatedCode
                size = payload.size

                # Prevents execution if the packet is too small. It is meaningless to send empty packets.
                if size < self.min_payload_size:
                    self.status = self.payload_too_small_error
                    return np.empty(0, dtype=payload.dtype)

                # Prevents execution if the payload is too large. Due to using byte-streams and COBS encoding, the
                # overhead byte can only store a maximum value of 255 and for any payload it should be able to store the
                # distance to the end of the packet. 254 bytes is the maximum size that still fits that requirement once
                # overhead and delimiter are added to the payload.
                elif size > self.max_payload_size:
                    self.status = self.payload_too_large_error
                    return np.empty(0, dtype=payload.dtype)

                # Ensures that the input payload uses uint8 datatype. Since the library uses byte-streams for
                # communication, this is an important prerequisite.
                if payload.dtype is not np.dtype(np.uint8):
                    self.status = self.invalid_payload_datatype_error
                    return np.empty(0, dtype=payload.dtype)

                # Initializes the output array, uses payload size + 2 as size to make space for the overhead and
                # delimiter bytes (see COBS scheme for more details on why this is necessary).
                packet = np.empty(size + 2, dtype=payload.dtype)
                packet[-1] = delimiter  # Sets the last byte of the packet to the delimiter byte value
                packet[
                    1:-1
                ] = payload  # Copies input payload into the packet array, leaving spaces for overhead and delimiter.

                # A tracker variable that is used to calculate the distance to the next delimiter value when an
                # unencoded delimiter is required.
                next_delimiter_position = packet.size - 1  # Initializes to the index of the delimiter value added above

                # Iterates over the payload in reverse and replaces every instance of the delimiter value inside the
                # payload with the distance to the next delimiter value (or the value added to the end of the payload).
                # This process ensures that the delimiter value is only found at the end of the packet and, if delimiter
                # is not 0, potentially also as the overhead byte value. This encodes the payload using COBS scheme.
                for i in range(size - 1, -1, -1):  # Loops over every index of the payload
                    if payload[i] == delimiter:
                        # If any of the payload values match the delimiter value, replaces that value in the packet with
                        # the distance to the next_delimiter_position. This is either the distance to the next encoded
                        # value or the distance to the delimiter value located at the end of the packet.
                        packet[i + 1] = next_delimiter_position - (
                            i + 1
                        )  # +1 is to translate for payload to packet index

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
                self.status = self.payload_encoded
                return packet

            def decode_payload(self, packet: np.ndarray, delimiter: np.uint8 = 0) -> np.ndarray:
                """Decodes the COBS-encoded payload from the input packet.

                This is a jit method that is very fast but requires strict input / output typing as it is compiled to C.
                If possible, only use this method through a wrapper API to ensure proper error handling.

                Args:
                    packet: The numpy array that stores COBS-encoded packet. The array should be using uint8 datatype
                        and has to be entirely filled with the packet data. That means that the first index (0) should
                        store the overhead byte and the last valid index of the packet should store the unencoded
                        delimiter. The packet should be between 3 and 256 bytes in length.
                    delimiter: The numpy uint8 value (0 through 255) that is used as the packet delimiter. It is used to
                        optimize the decoding flow and to verify the unencoded delimiter at the end of the packet.

                Returns:
                    The payload uint8 numpy array decoded from the packet if the method succeeds or an empty
                    uninitialized numpy array otherwise. Use the status code available through the 'status' attribute of
                    the class instance to determine if the method succeeded or failed (in this case, the code provides
                    specific error code).
                """
                # noinspection DuplicatedCode
                size = packet.size  # Extracts packet size for the checks below

                # This is needed due to how this method is used by the main class, where the input to this method
                # happens to be a 'readonly' array. Copying the array removes the readonly flag.
                packet = packet.copy()

                # Prevents execution if the size of the packet is too small. The packet should at minimum have enough
                # space for the overhead byte, one payload byte and the delimiter byte (3 bytes).
                # noinspection DuplicatedCode
                if size < self.min_packet_size:
                    self.status = self.packet_too_small_error
                    return np.empty(0, dtype=packet.dtype)

                # Also prevents execution if the size of the packet is too large. The maximum size is enforced due to
                # how the COBS encoding works, as it requires having at most 255 bytes between the overhead byte and the
                # end of the packet.
                elif size > self.max_packet_size:
                    self.status = self.packet_too_large_error
                    return np.empty(0, dtype=packet.dtype)

                # Ensures that the input packet uses uint8 datatype. Since the library uses byte-streams for
                # communication, this is an important prerequisite.
                if packet.dtype is not np.dtype(np.uint8):
                    self.status = self.invalid_packet_datatype_error
                    return np.empty(0, dtype=packet.dtype)

                # Tracks the currently evaluated variable's index in the packet array. Initializes to 0 (overhead byte
                # index).
                read_index = 0

                # Tracks the distance to the next index to evaluate, relative to the read_index value
                next_index = packet[read_index]  # Reads the distance stored in the overhead byte into the next_index

                # Loops over the payload and iteratively jumps over all encoded values, restoring (decoding) them back
                # to the delimiter value in the process. Carries on with the process until it reaches the end of the
                # packet or until it encounters an unencoded delimiter values. These two conditions should coincide for
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
                            self.status = self.payload_decoded
                            return packet[1:-1]
                        else:
                            # If the delimiter is encountered before reaching the end of the packet, this indicates that
                            # the packet was corrupted during transmission and the CRC-check failed to recognize the
                            # data corruption. In this case, returns an error code.
                            self.status = self.delimiter_found_too_early_error
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
                self.status = self.delimiter_not_found_error
                return np.empty(0, dtype=packet.dtype)

        return _COBSProcessor()


class CRCProcessor:
    """Provides methods for working with CRC checksums used to verify the integrity of transferred data packets.

    For more information on how the CRC checksum works, see the original paper:
    W. W. Peterson and D. T. Brown, "Cyclic Codes for Error Detection," in Proceedings of the IRE, vol. 49, no. 1,
    pp. 228-235, Jan. 1961, doi: 10.1109/JRPROC.1961.287814.

    Notes:
        This class functions as a wrapper that provides a consistent Python API for the internal instance of a
        jit-compiled _CRCProcessor class. This allows achieving python-like experience when using the class while
        simultaneously benefiting from fast machine-compiled code generated through numba jit-optimization. The wrapper
        automatically converts internal class runtime status codes into exception error messages where appropriate to
        notify users about runtime errors.

        For the maximum execution speed, you can access the private methods directly (see SerializedTransferProtocol
        class), although this is highly discouraged.

        To increase runtime speed, this class generates a static CRC lookup table using the input polynomial, which is
        subsequently used to calculate CRC checksums. This statically reserves 256, 512 or 1024 bytes of RAM to store
        the table, which is more or less irrelevant for all modern systems.

        In addition to providing CRC checksum calculation methods, this class also provides methods for converting the
        calculated CRC checksum between unsigned integer format returned by the calculator and the numpy byte array used
        during transmission.

    Attributes:
        __processor: The private instance of the jit-compiled _CRCProcessor class which actually does all the required
            computations. Private by design and should never be accessed directly.

    Args:
        polynomial: The polynomial to use for the generation of the CRC lookup table. Can be provided as an
            appropriately sized HEX number (e.g., 0x1021). Note, currently only non-reversed polynomials of numpy
            uint8, uint16 and uint32 datatypes are supported.
        initial_crc_value: The initial value to which the CRC checksum variable is initialized during calculation.
            This value depends on the chosen polynomial algorithm ('polynomial' argument) and should use the same
            datatype as the polynomial argument. It can be provided as an appropriately sized HEX number (e.g., 0xFFFF).
        final_xor_value: The final XOR value to be applied to the calculated CRC checksum value. This value depends on
            the chosen polynomial algorithm ('polynomial' argument) and should use the same datatype as the polynomial
            argument. It can be provided as an appropriately sized HEX number (e.g., 0x0000).

    Raises:
        TypeError: If any of the class initialization arguments are not of the supported type, and if the arguments are
            not of the same type.
    """

    def __init__(
        self,
        polynomial: Union[np.uint8, np.uint16, np.uint32],
        initial_crc_value: Union[np.uint8, np.uint16, np.uint32],
        final_xor_value: Union[np.uint8, np.uint16, np.uint32],
    ) -> None:
        # Ensures that all inputs use the same valid type. Note, uint64 is currently not supported primarily to maintain
        # implicit compatibility with older AVR boards that do not support uint64 type. That said, both the C++ and this
        # Python codebases are written in a way that will natively scale to uint 64 if this static guard is modified to
        # allow it.
        if not isinstance(polynomial, (np.uint8, np.uint16, np.uint32)):
            error_message = (
                f"Unsupported 'polynomial' argument type '{type(polynomial)}' encountered when instantiating "
                f"CRCProcessor class. Use numpy uint8, uint16, or uint32."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif not isinstance(initial_crc_value, (np.uint8, np.uint16, np.uint32)):
            error_message = (
                f"Unsupported 'initial_crc_value' argument type {type(initial_crc_value)} encountered when "
                f"instantiating CRCProcessor class. Use numpy uint8, uint16, or uint32."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif not isinstance(final_xor_value, (np.uint8, np.uint16, np.uint32)):
            error_message = (
                f"Unsupported 'final_xor_value' argument type {type(final_xor_value)} encountered when instantiating "
                f"CRCProcessor class. Use numpy uint8, uint16, or uint32."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif not (polynomial.dtype == initial_crc_value.dtype == final_xor_value.dtype):
            error_message = (
                "All arguments ('polynomial', 'initial_crc_value', 'final_xor_value') must have the same type when "
                "instantiating CRCProcessor class."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Initializes and compiles the internal _CRCProcessor class. This automatically generates the static CRC lookup
        # table
        self.__processor = self.__make_crc_processor_class(polynomial, initial_crc_value, final_xor_value)

    def __repr__(self) -> str:
        repr_message = (
            f"CRCProcessor(inner_status={self.__processor.status}, "
            f"polynomial={hex(self.__processor.polynomial)}, "
            f"initial_crc_value={hex(self.__processor.initial_crc_value)}, "
            f"final_xor_value={hex(self.__processor.final_xor_value)}, "
            f"crc_byte_length={self.__processor.crc_byte_length})"
        )
        return repr_message

    def calculate_packet_crc_checksum(self, buffer: np.ndarray) -> Union[np.uint8, np.uint16, np.uint32]:
        """Calculates the checksum for the input buffer.

        This method loops over the contents of the buffer and iteratively computes the CRC checksum for the entire
        buffer. Assumes that the buffer is entirely made up of the data to be checksummed.

        Args:
            buffer: The uint8 numpy array that stores the data to be checksummed.

        Returns:
            A numpy uint8, uint16 or uint32 integer (depends on the polynomial datatype that was used during class
            initialization) that stores the calculated CRC checksum value.

        Raises:
            TypeError: If the input buffer is not a numpy array.
            ValueError: If the inner _CRCProcessor class returns an unsupported value for the byte-size of the CRC
                checksum (This is impossible unless the class code is tampered with, so serves as a static guard to aid
                developers).
        """

        # Prevents using the method for unsupported input types
        if not isinstance(buffer, np.ndarray):
            error_message = (
                f"A uint8 numpy ndarray buffer expected, but instead encountered '{type(buffer)}' when calculating the "
                f"CRC checksum for the input buffer."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Calls the appropriate _CRCProcessor method to calculate the crc checksum
        checksum = self.__processor.calculate_packet_crc_checksum(buffer)

        # Verifies that checksum calculation was successful
        self.__verify_checksum_calculation_outcome(buffer)

        # Since other methods expect numpy values, the checksum is explicitly cast to the correct type here. Numba has a
        # limitation, where it prefers python types and casts all outputs to them regardless of the type assigned during
        # numba runtime. This is why the types need to be resolved explicitly at the level of the wrapper.
        if self.__processor.crc_byte_length == 1:
            return np.uint8(checksum)
        elif self.__processor.crc_byte_length == 2:
            return np.uint16(checksum)
        elif self.__processor.crc_byte_length == 4:
            return np.uint32(checksum)
        else:
            error_message = (
                f"Unsupported 'crc_byte_length' value ({self.__processor.crc_byte_length}) encountered when "
                f"calculating the CRC checksum for the input buffer."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    def __verify_checksum_calculation_outcome(self, buffer: np.ndarray) -> None:
        """Verifies that calculate_packet_crc_checksum() method runtime was successful. If not, raises the appropriate
        exception.

        Notes:
            The main reason for having this as a separate method is to allow verification following direct calls to
            inner methods that bypass the API wrapper methods.

        Args:
            buffer: The input buffer that was checksummed. Only used to extract some descriptive information, such as
            its dtype, to make error messages more informative.

        Raises:
            ValueError: If the input buffer numpy array is not using the uint8 datatype.
            RuntimeError: If the status code returned by the crc calculator method is not one of the expected values.
        """
        # Success code, verification successful
        if self.__processor.status == self.__processor.checksum_calculated:
            pass

        # Incorrect buffer datatype
        elif self.__processor.status == self.__processor.calculate_checksum_buffer_datatype_error:
            error_message = (
                f"The datatype of the input buffer to be CRC-checksummed ({buffer.dtype}) is not supported. "
                f"Only uint8 (byte) numpy arrays are currently supported as buffer inputs. "
                f"CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Unexpected status code
        else:
            error_message = (
                f"Unexpected inner _CRCProcessor class status code ({self.__processor.status}) encountered when "
                f"attempting to calculate the CRC checksum for the input buffer. "
                f"CODE: 0."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    def convert_crc_checksum_to_bytes(self, crc_checksum: Union[np.uint8, np.uint16, np.uint32]) -> np.ndarray:
        """Converts the input numpy integer checksum into a byte numpy array.

        This method converts a multibyte CRC checksum into multiple bytes and writes them to a numpy uint8 array
        starting with the highest byte of the checksum.

        Returns:
            A uint8 numpy array entirely filled with the CRC checksum bytes.

        Raises:
            TypeError: If the input crc_checksum is not a numpy uint8, uint16 or uint32 integer.
        """
        # Prevents using the method for unsupported input types
        if not isinstance(crc_checksum, (np.uint8, np.uint16, np.uint32)):
            error_message = (
                f"A uint8, uint16 or uint32 crc_checksum expected, but instead encountered '{type(crc_checksum)}', "
                f"when converting the unsigned integer CRC checksum to an array of bytes."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Calls the appropriate _CRCProcessor method to convert the crc checksum to an array of bytes
        checksum_bytes = self.__processor.convert_crc_checksum_to_bytes(crc_checksum)

        # At the time of writing this method cannot fail, and this is more or less a static check that the returned
        # code matches the success code in case something changes in the future.
        self.__verify_crc_to_bytes_conversion()
        return checksum_bytes

    def __verify_crc_to_bytes_conversion(self) -> None:
        """Verifies that convert_crc_checksum_to_bytes() method runtime was successful. If not, raises the appropriate
        exception.

        Notes:
            The main reason for having this as a separate method is to allow verification following direct calls to
            inner methods that bypass the API wrapper methods.

        Raises:
            RuntimeError: If the status code returned by the checksum converter method is not one of the expected
                values.
        """

        # Success code, verification successful
        if self.__processor.status == self.__processor.checksum_converted_to_bytes:
            pass

        # Unknown status code
        else:
            error_message = (
                f"Unexpected inner _CRCProcessor class status code ({self.__processor.status}) encountered when "
                f"converting the unsigned integer CRC checksum to an array of bytes. CODE: 0."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    def convert_crc_checksum_to_integer(self, buffer: np.ndarray) -> Union[np.uint8, np.uint16, np.uint32]:
        """Converts the input buffer that stores crc checksum bytes to an unsigned numpy integer value.

        This method is used to convert uint8 (byte) numpy arrays to crc checksum integer values. The method assumes
        that the checksum has been converted to bytes starting with the highest byte of the checksum and assumes that
        the buffer is entirely filled with the checksum bytes.

        Returns:
            A numpy uint8, uint16 or uint32 integer (depends on the polynomial datatype that was used during class
            initialization) that stores the converted CRC checksum value.

        Raises:
            TypeError: If the input buffer is not a numpy array.
            ValueError: If the inner _CRCProcessor class returns an unsupported value for the byte-size of the CRC
                checksum (This is impossible unless the class code is tampered with, so serves as a static guard to aid
                 developers).
        """

        # Prevents using the method for unsupported input types
        if not isinstance(buffer, np.ndarray):
            error_message = (
                f"A uint8 numpy ndarray buffer expected, but instead encountered '{type(buffer)}' type when converting "
                f"the array of CRC checksum bytes to the unsigned integer value."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Calls the appropriate _CRCProcessor method to convert the aray of crc checksum bytes to an integer value
        checksum = self.__processor.convert_crc_checksum_to_integer(buffer)

        # Verifies method runtime status
        self.__verify_crc_to_integer_conversion(buffer)

        # Since other methods expect numpy values, the checksum is explicitly cast to the correct type here. Numba has a
        # limitation, where it prefers python types and casts all outputs to them regardless of the type assigned during
        # numba runtime. This is why the types need to be resolved explicitly at the level of the wrapper.
        if self.__processor.crc_byte_length == 1:
            return np.uint8(checksum)
        elif self.__processor.crc_byte_length == 2:
            return np.uint16(checksum)
        elif self.__processor.crc_byte_length == 4:
            return np.uint32(checksum)
        else:
            error_message = (
                f"Unsupported 'crc_byte_length' value ({self.__processor.crc_byte_length}) encountered when "
                f"calculating the CRC checksum for the input buffer."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    def __verify_crc_to_integer_conversion(self, buffer: np.ndarray) -> None:
        """Verifies that convert_crc_checksum_to_integer() method runtime was successful. If not, raises the appropriate
        exception.

        Notes:
            The main reason for having this as a separate method is to allow verification following direct calls to
            inner methods that bypass the API wrapper methods.

        Args:
            buffer: The input buffer that was checksummed. Only used to extract some descriptive information, such as
            its dtype, to make error messages more informative.

        Raises:
            ValueError: If the input buffer numpy array is not using the uint8 datatype or is not exactly the same
                byte-size as needed to store the unsigned integer checksum (this is derived from the polynomial
                byte-size).
            RuntimeError: If the status code returned by the checksum converter method is not one of the expected
                values.
        """

        # Success code, verification successful
        if self.__processor.status == self.__processor.checksum_converted_to_integer:
            pass

        # Invalid buffer datatype
        elif self.__processor.status == self.__processor.convert_checksum_invalid_buffer_datatype_error:
            error_message = (
                f"The datatype of the input buffer to be converted to the unsigned integer CRC checksum "
                f"({buffer.dtype}) is not supported. Only uint8 (byte) numpy arrays are currently supported as buffer "
                f"inputs. CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # The size of the buffer does not match the number of bytes required to represent the checksum datatype
        elif self.__processor.status == self.__processor.convert_checksum_invalid_buffer_size_error:
            error_message = (
                f"The byte-size of the input buffer to be converted to the unsigned integer CRC checksum "
                f"({buffer.size}) does not match the size required to represent the specified checksum datatype "
                f"({self.__processor.crc_byte_length}). CODE: {self.__processor.status}."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Unknown status code
        else:
            error_message = (
                f"Unexpected inner _CRCProcessor class status code ({self.__processor.status}) encountered when "
                f"converting the array of CRC checksum bytes to the unsigned integer value. CODE: 0."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

    @property
    def crc_byte_length(self) -> np.uint8:
        """Returns the variable byte-size used by the class instance to store CRC checksums."""
        return self.__processor.crc_byte_length

    @property
    def crc_table(self) -> np.ndarray:
        """Returns the CRC lookup table used by the class instance during checksum calculation."""
        return self.__processor.crc_table

    @property
    def processor(self):
        """Returns the private jit-compiled _CRCProcessor class instance. This accessor represents a convenient way of
        unwrapping the jit-compiled class, so that the fast methods can be used directly (helpful when you want to use
        them from another jit-method).
        """
        return self.__processor

    @property
    def polynomial(self) -> Union[np.uint8, np.uint16, np.uint32]:
        """Returns the polynomial used by the class instance during checksum calculation."""
        return self.__processor.polynomial

    @property
    def initial_crc_value(self) -> Union[np.uint8, np.uint16, np.uint32]:
        """Returns the initial value used by the class instance during checksum calculation."""
        return self.__processor.initial_crc_value

    @property
    def final_xor_value(self) -> Union[np.uint8, np.uint16, np.uint32]:
        """Returns the final XOR value used by the class instance during checksum calculation."""
        return self.__processor.final_xor_value

    @staticmethod
    def __make_crc_processor_class(
        polynomial: Union[np.uint8, np.uint16, np.uint32],
        initial_crc_value: Union[np.uint8, np.uint16, np.uint32],
        final_xor_value: Union[np.uint8, np.uint16, np.uint32],
    ):
        """Instantiates an appropriately configured CRCProcessor class to work with the polynomial of the requested
        type.

        Notes:
            This setup method mimics the operation of the CRCProcessor's template method used in the microcontroller
            version of this library. Since Numba is used to speed up class methods for the PC version of the library, it
            has to be provided with appropriately typed class attributes (polynomial, crc_table, etc.) at compile-time.
            To make that possible, this template-like function written in python resolves the necessary dependencies and
            initializes and returns the properly configured CRCProcessor class instance.

            This method relies on the fact that the input polynomial and other supporting arguments are type-checked
            by the initializer method of the wrapper CRCProcessor class and, as such, has no mechanisms for checking
            these inputs for validity or cross-compatibility.

        Args:
            polynomial: The polynomial to use for the generation of the CRC lookup table. Can be provided as an
                appropriately sized HEX number (e.g., 0x1021). Note, currently only non-reversed polynomials of numpy
                uint8, uint16 and uint32 datatype are supported.
            initial_crc_value: The initial value to which the CRC checksum variable is initialized during calculation.
                This value depends on the chosen polynomial algorithm ('polynomial' argument) and, as such, should use
                the same datatype as the polynomial argument. It can be provided as an appropriately sized HEX number
                (e.g., 0xFFFF).
            final_xor_value: The final XOR value to be applied to the calculated CRC checksum value. This value depends
                on the chosen polynomial algorithm ('polynomial' argument) and, as such, should use the same datatype as
                the polynomial argument. It can be provided as an appropriately sized HEX number (e.g., 0x0000).

        Returns:
            A fully configured CRCProcessor class instance. The class is configured to work with a specific
            datatype (uint8, uint16, or uint32) of the polynomial and, by extension, crc_checksums. Using the class with
            the crc checksums of an unsupported type is likely to result in undefined behavior.
        """

        # Converts the input polynomial type from numpy to numba format so that it can be used in the spec array below
        if polynomial.dtype is np.dtype(np.uint8):
            crc_type = uint8
        elif polynomial.dtype is np.dtype(np.uint16):
            crc_type = uint16
        elif polynomial.dtype is np.dtype(np.uint32):
            crc_type = uint32
        # Generally this is redundant due to a static guard inside the wrapper class initialization function. However,
        # this may be helpful if the class ever needs to be adjusted to support uint64 polynomials as this template
        # method needs to be adjusted in addition to the static guard. The code inside class methods will scale
        # automatically
        else:
            error_message = (
                f"Unsupported 'polynomial' type ({polynomial.dtype}) encountered when resolving the inner "
                f"_CRCProcessor jit-compiled class specification."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # The template for the numba compiler to assign specific datatypes to variables used by CRCProcessor class.
        crc_spec = [
            ("status", uint8),
            ("polynomial", crc_type),
            ("initial_crc_value", crc_type),
            ("final_xor_value", crc_type),
            ("crc_byte_length", uint8),
            ("crc_table", crc_type[:]),
            ("standby", uint8),
            ("calculate_checksum_buffer_datatype_error", uint8),
            ("checksum_calculated", uint8),
            ("checksum_converted_to_bytes", uint8),
            ("convert_checksum_invalid_buffer_datatype_error", uint8),
            ("convert_checksum_invalid_buffer_size_error", uint8),
            ("checksum_converted_to_integer", uint8),
            ("calculate_and_append_checksum_buffer_datatype_error", uint8),
            ("checksum_calculated_and_appended_to_buffer", uint8),
        ]

        # Uses the spec from above to define and compile the appropriate CRCProcessor class instance
        @jitclass(crc_spec)
        class _CRCProcessor:
            """The inner CRCProcessor class that actually implements all method logic.

            Notes:
                This class is optimized using Numba's JIT (Just-In-Time) compilation module to significantly improve
                the execution speed of all class methods. As an unfortunate side effect, this process interferes with
                Python's built-in error handling tools. To provide error-handling capacity, a C-driven approach of
                returning fixed byte error-codes has been implemented. The error codes are available through the class
                attributes below. Each method returns the status (success or error) code by setting the class 'status'
                attribute to the latest runtime code, mimicking the functioning of the class version intended for
                microcontrollers.

            Attributes:
                status: Stores the last-called method's runtime status code.
                polynomial: Stores the polynomial used for the CRC checksum calculation. Only used for the wrapper class
                    __repr__ method as the inner class immediately converts the polynomial to a static crc_table upon
                    initialization.
                initial_crc_value: Stores the initial value used for the CRC checksum calculation. Specifically, this is
                    the value that the CRC checksum variable is initialized to at the beginning of each
                    calculate_packet_crc_checksum() method runtime.
                final_xor_value: Stores the final XOR value used for the CRC checksum calculation. Specifically, this is
                    the value that the CRC checksum variable is XORed with prior to being returned to caller at the end
                    of each calculate_packet_crc_checksum() method runtime.
                crc_byte_length: Stores the length of the CRC polynomial in bytes. This is used across most methods of
                    the class to automatically scale processing to the number of bytes used to store the CRC checksum
                    value.
                crc_table: The array that stores the CRC lookup table. The lookup table is used to speed up CRC checksum
                    calculation by pre-computing the checksum value for each possible byte-value (from 0 through 255:
                    256 values total). The table is filled automatically during class instantiation and reserves 256,
                    512 or 1024 bytes of RAM for the entire lifetime of the class, depending on the numpy datatype of
                    the polynomial.
                standby: The integer code used during class initialization (before any method is called).
                calculate_checksum_buffer_datatype_error: The buffer provided to the calculate_packet_crc_checksum()
                    method was not of the required uint8 numpy datatype.
                checksum_calculated: The CRC checksum has been successfully calculated.
                checksum_converted_to_bytes: The CRC checksum has been successfully converted to an uint8 numpy array.
                convert_checksum_invalid_buffer_datatype_error: The buffer provided to the
                    convert_crc_checksum_to_integer() method was not of the required uint8 numpy datatype.
                convert_checksum_invalid_buffer_size_error: The buffer provided to the convert_crc_checksum_to_integer()
                    method was not of the byte-size required to store the byte-converted crc checksum value.
                checksum_converted_to_integer: The CRC checksum has been successfully converted from an uint8 numpy
                    array of bytes to an appropriate numpy unsigned integer (uint8, uint16 or uint32).

            Args:
                _polynomial: The polynomial to use for the generation of the CRC lookup table. Can be provided as an
                    appropriately sized HEX number (e.g., 0x1021). Note, currently only non-reversed polynomials are
                    supported.
                _polynomial_size: The size of the polynomial in bytes. This is used to support the manipulations
                    necessary to calculate the CRC checksum and add / read it from storage buffers.
                _initial_crc_value: The initial value to which the CRC checksum variable is initialized during
                    calculation. This value is based on the polynomial parameter. Can be provided as an appropriately
                    sized HEX number (e.g., 0xFFFF).
                _final_xor_value: The final XOR value to be applied to the calculated CRC checksum value. This value is
                    based on the polynomial parameter. Can be provided as an appropriately sized HEX number
                    (e.g., 0x0000).
            """

            def __init__(
                self,
                _polynomial: Union[np.uint8, np.uint16, np.uint32],
                _polynomial_size: np.uint8,
                _initial_crc_value: Union[np.uint8, np.uint16, np.uint32],
                _final_xor_value: Union[np.uint8, np.uint16, np.uint32],
            ) -> None:
                # Local variables
                self.polynomial = _polynomial
                self.initial_crc_value = _initial_crc_value
                self.final_xor_value = _final_xor_value
                self.crc_byte_length = _polynomial_size
                self.crc_table = np.empty(256, dtype=crc_type)  # Initializes to empty for efficiency

                # Static status_codes
                self.standby = 51  # The code used right after class initialization (before any other method is called)
                self.calculate_checksum_buffer_datatype_error = 52
                self.checksum_calculated = 53
                self.checksum_converted_to_bytes = 54
                self.convert_checksum_invalid_buffer_datatype_error = 55
                self.convert_checksum_invalid_buffer_size_error = 56
                self.checksum_converted_to_integer = 57

                self.status = self.standby  # Dynamically updated to track the latest method runtime status

                # Calls table generation method that generates a lookup table based on the target polynomial
                # parameters and iteratively sets each variable inside the crc_table placeholder to the calculated
                # values.
                self.generate_crc_table(_polynomial=_polynomial)

            # noinspection DuplicatedCode
            def calculate_packet_crc_checksum(self, buffer: np.ndarray) -> Union[np.uint8, np.uint16, np.uint32]:
                """Calculates the checksum for the (entire) input buffer.

                This is a jit method that is very fast but requires strict input / output typing as it is compiled to C.
                If possible, only use this method through a wrapper API to ensure proper error handling.

                Notes:
                    While error runtimes always return 0, any 0-value returned by this method is potentially a valid
                    value. To determine if the method runtime was successful or failed, use 'status' class attribute.
                    The returned value is not meaningful until it is verified using the status code!

                Args:
                    buffer: The uint8 numpy array that stores the data to be checksummed.

                Returns:
                    A numpy uint8, uint16 or uint32 integer (depends on the polynomial datatype that was used during
                    class initialization) that represents the calculated CRC checksum value. Also sets the 'status'
                    class attribute to communicate the status of the method's runtime.
                """

                # Verifies that the buffer is using an appropriate datatype (uint8). This method is intended to work
                # with buffers storing byte-serialized data, so explicitly controls for that here.
                if buffer.dtype is not np.dtype(np.uint8):
                    self.status = self.calculate_checksum_buffer_datatype_error
                    return np.uint8(0)

                # Initializes the checksum. The datatype is already correct as it is inferred from the initial_crc_value
                # datatype
                crc_checksum = self.initial_crc_value

                # Loops over each byte inside the buffer and iteratively calculates CRC checksum for the buffer
                for byte in buffer:
                    # Calculates the index to retrieve from CRC table. To do so, combines the high byte of the CRC
                    # checksum with the (possibly) modified (corrupted) data_byte using bitwise XOR.
                    table_index = (crc_checksum >> (8 * (self.crc_byte_length - 1))) ^ byte

                    # Extracts the byte-specific CRC value from the table using the result of the operation above. The
                    # retrieved CRC value from the table is then XORed with the checksum that is shifted back to the
                    # original position to generate an updated checksum.
                    crc_checksum = self.make_polynomial_type((crc_checksum << 8) ^ self.crc_table[table_index])

                # The Final XOR operation may or may not actually be used (depending on the polynomial). The default
                # polynomial 0x1021 has it set to 0x0000 (0), so it is actually not used. Other polynomials may require
                # this step, so it is kept here for compatibility reasons. The exact algorithmic purpose of the XOR
                # depends on the specific polynomial used.
                crc_checksum ^= self.final_xor_value

                # Sets the status to indicate runtime success and returns calculated checksum to the caller.
                self.status = self.checksum_calculated
                return self.make_polynomial_type(crc_checksum)

            def convert_crc_checksum_to_bytes(self, crc_checksum: Union[np.uint8, np.uint16, np.uint32]) -> np.ndarray:
                """Converts the input checksum value into a numpy array of bytes.

                This is a jit method that is very fast but requires strict input / output typing as it is compiled to C.
                If possible, only use this method through a wrapper API to ensure proper error handling.

                Returns:
                    A uint8 numpy array entirely filled with the CRC checksum bytes. Also sets the 'status' class
                    attribute to communicate method runtime status.
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
                self.status = self.checksum_converted_to_bytes
                return buffer

            def convert_crc_checksum_to_integer(self, buffer: np.ndarray) -> Union[np.uint8, np.uint16, np.uint32]:
                """Converts the crc checksum stored in the (entire!) input buffer to an unsigned numpy integer value.

                This is a jit method that is very fast but requires strict input / output typing as it is compiled to C.
                If possible, only use this method through a wrapper API to ensure proper error handling.

                Notes:
                    While error runtimes always return 0, any 0-value returned by this method is potentially a valid
                    value. To determine if the method runtime was successful or failed, use 'status' class attribute.
                    The returned value is not meaningful until it is verified using the status code!

                Returns:
                    A numpy uint8, uint16 or uint32 integer (depends on the polynomial datatype that was used during
                    class initialization) that represents the converted CRC checksum value. Also sets the 'status' class
                    attribute to communicate method runtime status.
                """

                # Verifies that the input buffer uses an appropriate (uint8) datatype. This method is intended to decode
                # CRC checksum values from serialized byte-streams and will not work properly with any other data types.
                if buffer.dtype is not np.dtype(np.uint8):
                    self.status = self.convert_checksum_invalid_buffer_datatype_error
                    # Note, 0 is a valid value. The only way to know if it comes from a successful or failed runtime is
                    # to check the class 'status' attribute that communicates the latest runtime success or error code.
                    return np.uint8(0)

                # Ensures that the buffer size exactly matches the number of bytes required to store the CRC checksum.
                if buffer.size != self.crc_byte_length:
                    self.status = self.convert_checksum_invalid_buffer_size_error
                    return np.uint8(0)

                # Precreates the variable tos tore the extracted checksum and initializes it to zero
                extracted_crc = self.make_polynomial_type(0)

                # Loops over the input buffer and extracts the CRC checksum from the bytes inside the buffer. Assumes
                # the buffer is entirely filled with the checksum bytes and uses crc_byte_length to constrain processing
                # to the exact number of bytes required.
                for i in range(self.crc_byte_length):
                    # Constructs the CRC checksum from the buffer, starting from the most significant byte and moving
                    # towards the least significant byte. This matches the process of how it was converted to bytes by
                    # the convert_crc_checksum_to_bytes() or an equivalent microcontroller method.
                    extracted_crc |= self.make_polynomial_type(buffer[i] << (8 * (self.crc_byte_length - i - 1)))

                # Returns the extracted CRC checksum to caller and sets the status to communicate runtime success.
                self.status = self.checksum_converted_to_integer
                return extracted_crc

            def generate_crc_table(self, _polynomial: Union[np.uint8, np.uint16, np.uint32]) -> None:
                """Uses the polynomial specified during class instantiation to compute the CRC checksums for each
                possible uint8 (byte) value.

                The method updates the precompiled empty crc_table with polynomial-derived CRC values. Note, the
                crc_table has to be initialized correctly for this method to work properly. Use
                make_crc_processor_class() wrapper class method to initialize the class and the crc_table as it
                automatically resolves all type dependencies. Additionally, this method is ONLY intended to be called by
                the class initialization method. Do not use this method outside the class initialization context!

                Notes:
                    While the PC is sufficiently fast to work without a pregenerated table, this method is used to
                    maintain algorithmic similarity to the version of the library used for microcontrollers. Also, using
                    a static table is still faster even for PCs.

                Args:
                    _polynomial: The polynomial to use for the generation of the CRC lookup table. Can be provided as an
                        appropriately sized HEX number (e.g., 0x1021). Note, currently only non-reversed polynomials are
                        supported.
                """

                # Determines the number of bits in the CRC type
                crc_bits = np.uint8(self.crc_byte_length * 8)

                # Determines the Most Significant Bit (MSB) mask based on the CRC type
                msb_mask = self.make_polynomial_type(np.left_shift(1, crc_bits - 1))

                # Iterates over each possible value of a byte variable
                for byte in np.arange(256, dtype=np.uint8):
                    # Casts crc to the appropriate type based on the polynomial type
                    crc = self.make_polynomial_type(byte)

                    # Shifts the CRC value left by the appropriate number of bits based on the CRC type to align the
                    # initial value to the highest byte of the CRC variable.
                    if crc_bits > 8:
                        crc <<= crc_bits - 8

                    # Loops over each of the 8 bits making up the byte-value being processed
                    for _ in range(8):
                        # Checks if the top bit (MSB) is set
                        if crc & msb_mask:
                            # If the top bit is set, shifts the crc value left to bring the next bit into the top
                            # position, then XORs it with the polynomial. This simulates polynomial division where bits
                            # are checked from top to bottom.
                            crc = self.make_polynomial_type((crc << 1) ^ _polynomial)
                        else:
                            # If the top bit is not set, simply shifts the crc value left. This moves to the next bit
                            # without changing the current crc value, as division by polynomial wouldn't modify it.
                            crc <<= 1

                    # Adds the calculated CRC value for the byte to the storage table using byte-value as the key
                    # (index). This value is the remainder of the polynomial division of the byte (treated as a
                    # CRC-sized number), by the CRC polynomial.
                    self.crc_table[byte] = crc

            def make_polynomial_type(self, value: Any) -> Union[np.uint8, np.uint16, np.uint32]:
                """Converts the input value to the appropriate numpy unsigned integer type based on the class instance
                polynomial datatype.

                This is a minor helper method designed to be used exclusively by other class methods. It allows
                resolving typing issues originating from the fact that, at the time of writing, numba is unable to use
                '.itemsize' and other properties of scalar numpy types.

                Notes:
                    The datatype of the polynomial is inferred based on the byte-length of the polynomial as either
                    uint8, uint16 or uint32 (uses 'crc_byte_length' attribute of the class).

                Args:
                    value: The value to convert to the appropriate type.

                Returns:
                    The value converted to the appropriate numpy unsigned integer datatype
                """
                if self.crc_byte_length == 1:
                    return np.uint8(value)
                elif self.crc_byte_length == 2:
                    return np.uint16(value)
                elif self.crc_byte_length == 4:
                    return np.uint32(value)
                # Empty errors are currently supported by Numba. Not the most informative way of doing this, but this
                # error should never occur unless the entire class is changed to add support for more polynomial types,
                # there this error will remind the developers to adjust this method.
                else:
                    raise RuntimeError()

        # Returns fully configured CRCProcessor class instance
        return _CRCProcessor(
            _polynomial=polynomial,
            _polynomial_size=np.uint8(polynomial.itemsize),
            _initial_crc_value=initial_crc_value,
            _final_xor_value=final_xor_value,
        )


class SerialMock:
    """Simulates the methods of pySerial.Serial class used by SerializedTransferProtocol class to support unit-testing.

    This class only provides the methods that are either helpful for testing (like resetting the Mock class buffers)
    or are directly used by the SerializedTransferProtocol class (reading and writing data, opening / closing the port,
    etc.). For example, since SerializedTransferProtocol class does not use readlines() method, this class does not
    offer its mock implementation.

    Notes:
        This class is used in place of the actual Serial class to enable unit-testing of main class methods without
        the confounds of a third-party library. Additionally, it makes the test suite for this library truly platform-
        and user-configuration-agnostic, since there is no longer a need provide a valid Serial port during class
        instantiation when the Mock class is used.

        Also, unlike its prototype, this class exposes the rx_ and tx_ buffers while using similar logic for adding data
        to the buffers as the real Serial class. This allows using this class to fully test and verify all
        SerializedTransferProtocol class methods and expect them to behave identically during real runtime.

        Since all communication protocols share the same interface, this class is also perfectly adequate for imitating
        the functioning of the ZeroMQSerial class.

    Attributes:
        is_open: A boolean flag that tracks the state of the serial port.
        tx_buffer: A buffer that stores the data to be sent over the serial port.
        rx_buffer: A buffer that stores the data received from the serial port.
    """

    def __init__(self) -> None:
        self.is_open = False
        self.tx_buffer = b""
        self.rx_buffer = b""

    def __repr__(self) -> str:
        repr_message = f"StreamMock(open={self.is_open})"
        return repr_message

    def open(self) -> None:
        """If 'is_open' flag is False, switches it to True. Simulates the effect of successful 'open' method calls."""
        if not self.is_open:
            self.is_open = True

    def close(self) -> None:
        """If 'is_open' flag is True, switches it to False. Simulates the effect of successful 'close' method calls."""
        if self.is_open:
            self.is_open = False

    def write(self, data: bytes) -> None:
        """Writes the input data (stored as a 'bytes' object) to the tx_buffer buffer.

        Args:
            data: The data to be written to the output buffer. Has to be stored as a 'bytes' python object.

        Raises:
            TypeError: If the input data is not in bytes' format.
            Exception: If the mock serial port is not open.
        """
        if self.is_open:
            if isinstance(data, bytes):
                self.tx_buffer += data
            else:
                raise TypeError("Data must be a 'bytes' object")
        else:
            raise Exception("Mock serial port is not open")

    def read(self, size=1) -> bytes:
        """Reads the requested 'size' number of bytes from the rx_buffer and returns them as 'bytes' object.

        Args:
            size: The number of bytes to be read from the rx_buffer.

        Returns:
            The requested number of bytes from the rx_buffer as a 'bytes' object.

        Raises:
            Exception: If the mock serial port is not open.
        """
        if self.is_open:
            data = self.rx_buffer[:size]
            self.rx_buffer = self.rx_buffer[size:]
            return data
        else:
            raise Exception("Mock serial port is not open")

    def reset_input_buffer(self) -> None:
        """Resets the input buffer to an empty byte array.

        Raises:
            Exception: If the mock serial port is not open.
        """
        if self.is_open:
            self.rx_buffer = b""
        else:
            raise Exception("Mock serial port is not open")

    def reset_output_buffer(self) -> None:
        """Resets the output buffer to an empty byte array.

        Raises:
            Exception: If the mock serial port is not open.
        """
        if self.is_open:
            self.tx_buffer = b""
        else:
            raise Exception("Mock serial port is not open")

    @property
    def in_waiting(self) -> int:
        """Returns the number of bytes currently stored in the rx_buffer."""
        return len(self.rx_buffer)

    @property
    def out_waiting(self) -> int:
        """Returns the number of bytes currently stored in the tx_buffer."""
        return len(self.tx_buffer)

class SerializedConnection:
    def __init__(
        self,
        address_or_port: str,
        backend: Literal["pyserial", "zeromq", "mock"],
        baudrate: Optional[int] = None,
        connection_mode: Optional[Literal["host", "client"]] = None,
        timeout: Optional[int | float] = 0,
        local_buffer_size: int = 4096,
        force_open: bool = True,
        auto_open: bool = True,
    ) -> None:
        # Ensures the backend argument type is valid
        if not isinstance(backend, str):
            error_message = (
                f"Invalid 'backend' argument type ({type(backend)}) encountered when initializing SerialConnection "
                f"class object. A string option is expected. Supported options are: 'pyserial', 'zeromq' and 'mock'."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Resolves backend. This is by far the most important step of the initialization as it solely determine the
        # connection type
        if backend.lower() == "pyserial":
            self.__backend_code = self.pyserial_backend_code
            self.__backend = Serial(port=address_or_port, baudrate=baudrate, timeout=timeout)

        elif backend.lower == "zeromq":
            self.__backend_code = self.zeromq_backend_code
            self.__backend = ZeroMQSerial(address=address_or_port, connection_mode=connection_mode, timeout=timeout)
            if auto_open:
                self.__backend.open()

        elif backend.lower() == "mock":
            self.__backend_code = self.mock_backend_code
            self.__backend = SerialMock()

        # If the backend could not be resolved, this means an unsupported option was provided, so issues an error
        else:
            error_message = (
                f"Unsupported 'backend' argument value ({backend}) encountered when initializing SerialConnection "
                f"class object. Supported options are: 'pyserial', 'zeromq' and 'mock'."
            )
            raise ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Stores configuration parameters to class attributes
        self.__address = address_or_port
        self.__baudrate = baudrate
        self.__connection_mode = connection_mode
        self.__timeout = timeout

        # Initializes local attributes
        # noinspection PyTypeChecker
        self.__timer = ElapsedTimer("s")
        self.__circular_buffer = np.zeros(local_buffer_size, dtype=np.uint8)
        self.__occupied_buffer_size = 0
        self.__stop_flag = False
        self.__is_open = False

        # Initializes the necessary threading elements to control data reception
        self.__lock = threading.Lock()  # Lock object to synchronize _receiver thread with other class methods
        self.__receiver_thread = threading.Thread(target=self.__receiver)  # Instantiates the receiver thread.
        self.__receiver_thread.daemon = True  # Ensures the thread terminates with the main thread

        self.pyserial_backend_code = 1
        self.zeromq_backend_code = 2
        self.mock_backend_code = 3

        # If the class is initialized with force_open flag, ensures the port is closed before re-establishing
        # connection. This forces anything connected to the port to be ungracefully terminated, emptying the connection
        # slot for the client to connect. This option is very helpful on Windows which in testing has been found to be
        # somewhat unresponsive to graceful port release requests. Note, this only applies to Serial class, as
        # ZeroMQSerial currently does not have a mechanism (and neither it really needs one) to force connections.
        if force_open:
            self.__backend.close()

        # If the class is initialized with the auto_open flag, calls open() at the end of initialization. Otherwise,
        # open() has to be called manually before using read() or write() methods of the class.
        if auto_open:
            self.open()

    def __repr__(self):
        pass

    def __del__(self):
        pass

    def __receiver(self):
        """Extracts (receives) the data from the bundled backend reception buffer.

        This method is designed to be executed via a thread. After initiation, it runs until terminated by stop_flag.
        During runtime, it polls the state of the backend and if data is available to be received, writes it into the
        class circular buffer.

        Notes:
            This implementation is notably more complex and probably slower than a 'need-based' access, but it allows
            for runtime amortization when used as part of a pipeline. Since this library uses a lot of C-code linking
            (via Numba and other libraries), using a thread to continuously transfer data from the backend buffer to
            a local buffer allows to efficiently move data when the main thread is busy doing other C-operations. For
            complex pipelines, this is likely to result in better overall execution speeds. Also, using a centralized
            access approach allows standardizing backend access API, which is always fairly desirable.
        """
        event = threading.Event()  # This is used as a more accurate, GIL-releasing timer than sleep().

        # Runs until stop flag is triggered by the close() or __del__ method.
        while not self.__stop_flag:
            # Uses the same method to receive pySerial and SerialMock data as they have an identical API.
            if self.__backend_code == self.pyserial_backend_code or self.__backend_code == self.mock_backend_code:
                data = self.__receive_serial()

            # Uses a different interface for ZeroMQSerial backbone.
            elif self.__backend_code == self.zeromq_backend_code:
                data = self.__receive_serial()

            # Static guard against unknown backend codes. Should not really happen as this is explicitly controller
            # during initialization, but may be helpful for developers adding new backends
            else:
                error_message = (
                    f"Unexpected __backend_code ({self.__backend_code}) encountered while attempting to receive "
                    f"serialized data from connection backend. Manual intervention is required to resolve the error."
                )
                raise RuntimeError(error_message)

            # If data was received, as indicated by the size of the returned data array being above 0, adds it to the
            # end of the circular buffer. If the buffer does not have space to appends the data, discards elements from
            # the beginning of the buffer to make space.
            if data.size != 0:
                # Acquires the lock to ensure the buffer is not being accessed by another method (such as buffer reader
                # method).
                with self.__lock:
                    self.__occupied_buffer_size = self.__add_to_buffer(
                        self.__circular_buffer, data, self.__occupied_buffer_size
                    )

            # Blocks between 5 and 11 us. This is way more precise than the sleep() (that cannot resolve anything
            # shorter than 16 ms on Windows) and releases GIL, but not as good as the busy-wait using nanosecond timers.
            # That said, it's main goal is to ensure this method is delayed enough to let other threads acquire the GIL,
            # so this small floating delay works for this purpose. Note, slower systems may experience slower refresh
            # times here!
            event.wait(timeout=1e-6)

    def __receive_serial(self) -> np.ndarray:
        """Extracts the data stored in the reception buffer of a pySerial or SerialMock backend.

        Since SerialMock is implemented with identical bindings to the pySerial's Serial class, the data received by
        these classes can be accessed using the same API. Therefore, this method can be used to access the real and
        'mock' serial ports as a universal binding. This method is designed to be executed perpetually via a
        continuously running __receiver thread.

        Returns:
            A numpy array view generated using the received bytes object buffer, if there were bytes to receive. An
            empty numpy array if there were no bytes to receive.
        """
        available_bytes = self.__backend.in_waiting  # Checks if bytes are available
        if available_bytes:
            # If bytes are available, reads all available bytes
            read_bytes = self.__backend.read(available_bytes)

            # Converts the read bytes from 'bytes' to numpy as this is the format used by the rest of the library. Note,
            # this conversion generates a 'readonly' representation as it uses a memory view of the original buffer,
            # which makes it relatively fast
            return np.frombuffer(buffer=read_bytes, dtype=np.uint8)
        else:
            # If no bytes are available, returns an empty representation to indicate no bytes were read
            return np.empty(0, dtype=uint8)

    def __receive_zeromq(self) -> np.ndarray:
        """Extracts the data stored in the reception buffer of a ZeroMQSerial backend.

        This method uses the recv() method of the ZeroMQ backend to access the reception buffer. Uses an error-based
        mechanism which either returns the available bytes or raises an error that is suppressed locally (transformed
        into 'no available bytes' return value). This method is designed to be executed perpetually via a continuously
        running __receiver thread.

        Returns:
            A numpy array view generated using the received bytes object buffer, if there were bytes to receive. An
            empty numpy array if there were no bytes to receive.
        """
        try:
            # Attempts to receive the data. Using the NOBLOCK flag means that the method returns immediately: either
            # with the received data OR with zmq.Again exception if no data is available for reception
            message = self.__socket.recv(flags=zmq.NOBLOCK)
            return np.frombuffer(buffer=message, dtype=np.uint8)

        except zmq.Again:
            # If no bytes were available, returns an empty representation to indicate no bytes were read
            return np.empty(0, dtype=uint8)

    @staticmethod
    @njit(nogil=True, cache=True)
    def __add_to_buffer(buffer: np.ndarray, data: np.ndarray, occupied_buffer_size: int):
        """Appends input data to the end of the input circular buffer.

        This method simulates the behavior of the circular buffers found in many low-end Serial port controllers. It
        adds incoming data in a FIFO fashion to the buffer, discarding the oldest data in the case of an overflow.

        Notes:
            If the method runs out of space at the end of the buffer, it discards the necessary amount of elements from
            the beginning of the buffer (oldest elements), until enough space is made to accommodate the data. If the
            size of the input data array is larger than the size of the buffer, the buffer is replaced with the end
            portion of the data buffer (with the size that fits inside the buffer). This latter case is NOT an intended
            use case, and, due to JIT compilation, may require additional compilation time if encountered during
            runtime.

        Args:
            buffer: The buffer to which the data will be appended. This should be the class circular_buffer instance.
            data: The numpy array that stores the data to be appended. This should be the uint8 array generated using
                the received 'bytes' object as buffer.
            occupied_buffer_size: The portion of the buffer (relative to index 0) currently occupied with buffer.

        Returns:
            The size of the buffer that is occupied with data following the addition of the input data.

        """
        # The intended inputs to this method are generated using np.frombuffer, which makes them read-only. Copying the
        # data into a new numpy object before running the operations below prevents potential 'readonly access denied'
        # errors. Doing the copying inside the jit-compiled method saves execution time.
        local_data = data.copy()

        # If the buffer is not fully occupied and the available space is sufficeint to accommodate the input data,
        # simply appends the data to the end of the buffer. This is expected to be the most common case and, as such, it
        # is evaluated first
        if data.size + occupied_buffer_size <= buffer.size:
            buffer[occupied_buffer_size : occupied_buffer_size + data.size] = data
            occupied_buffer_size += data.size

        # The next likely case: if the buffer does not have sufficeint unoccupied space to accommodate the data, but
        # the size of the data is less than the size of the buffer, discards the oldest data (from the beginning of the
        # buffer) until sufficient amount of space is freed up.
        elif data.size < buffer.size:
            required_space = (data.size + occupied_buffer_size) - buffer.size  # Calculates extra space requirement

            # Shifts the buffer by the required space to the left. This moves the 'excessive' data from the front of the
            # buffer to the end of the buffer, where it will be overwritten with new data (and, therefore, discarded).
            buffer[:] = np.roll(buffer, -required_space)

            # Overwrites the tail end of the buffer with data
            buffer[-data.size :] = data
            occupied_buffer_size = buffer.size

        # The most unlikely case: If data size is larger than or equal to buffer size, keeps only the most recent data
        # that fits into the buffer. This effectively replaces the contents of the buffer with the end portion of the
        # data array (or the whole data array).
        else:
            buffer[:] = data[-buffer.size :]
            occupied_buffer_size = buffer.size

        # Returns the recalculated occupied size tracker so that it can be used in future calls of this method
        return occupied_buffer_size

    @property
    def in_waiting(self) -> int:
        """Returns the number of bytes currently stored inside the class reception buffer."""
        with self.__lock:  # Locks the reception thread before reading the array length
            # Since the buffer is implemented as a deque object, it does not have to necessarily store bytes. That said,
            # the class is explicitly designed to communicate with other class instances from these libraries which
            # always communicated via byte-streams, so it is assumed that the buffer is a bytes buffer.
            return len(self.__circular_buffer)

    @property
    def name(self) -> str:
        """Returns the port name (address) of the ZeroMQ socket used by the class."""
        return self.__port

    def read(self, size: int = 1) -> bytes:
        """Reads the requested number of bytes from the class circular buffer and returns them as 'bytes' object.

        If the requested number of bytes is not available and timeout is above 0, waits for the requested timeout number
        of seconds (using threading event sleep approach). Regardless of the timeout, the method will not read any bytes
        unless the requested number of bytes is available after the timeout loop resolution.

        Notes:
            This method is analogous to the read() method of pySerial Serial class, although it behaves slightly
            differently with respect to timeout.

            The method will NOT work unless ZeroMQ connection was established by calling the open() method. If the class
            is initialized with auto_open flag set to True, the connection has been opened during class initialization.
            Otherwise, this needs to be done manually.

        Args:
            size: The number of bytes to read from the circular buffer.

        Returns:
            A 'bytes' python object that stored acquired bytes. When the class timeout is set to 0 and not enough
            bytes are available, returns an empty 'bytes' object. When the class timeout is set to a number above 0, but
            sufficient number of bytes does not become available before timeout, also returns an empty 'bytes' object.

        Raises:
            RuntimeError: If this method is called before the connection has been established by calling the open()
                method.
        """

        # Prevents interfacing with a closed port.
        if not self.__is_open:
            error_message = (
                f"Unable to read data from a closed connection. Use open() method to open the zmq connection prior "
                f"to calling this method."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        event = threading.Event()  # This is used as a more accurate, GIL-releasing timer than sleep().
        bytes_available = True  # Instantiates a flag used to track if enough bytes are available

        # Blocks until enough bytes are available for reading. When timeout is disabled (set to 0), the method will
        # return immediately. Otherwise, it will wait for that number of SECONDS to acquire the data.
        self.__timer.reset()  # Resets the interval timer
        while self.in_waiting < size:
            if self.__timer.elapsed > self.__timeout:
                bytes_available = False
                break
            # Sleeps 30 us every check cycle. The receiver thread runs at 20 us intervals, so this should be sufficient
            # to de-stack the two and make it possible to receive data and execute this wait loop.
            event.wait(timeout=30e-6)

        if bytes_available:
            # Acquires the lock and retrieves the data
            with self.__lock:
                # Inlines most data acquisition steps by using list comprehension. Specifically, this loops over the
                # circular buffer and retrieves the requested number of bytes from the front of the buffer. The
                # algorithm relies on the while loop above to ensure the requested bytes are available and will not
                # try to retrieve anything if requested number of bytes is not available. Converts the formed list into
                # a bytes object and returns to caller
                return bytes([self.__circular_buffer.popleft() for _ in range(size)])

        # If bytes are not available, returns an empty bytes object to indicate nothing was read as the requested
        # quantity of bytes is not available
        else:
            return bytes()

    def write(self, data: bytes) -> None:
        """Sends the input data over the connected socket.

        Notes:
            The write method itself queues the zmq frame to be sent via an internal IO thread at a later time, much
            like the pySerial write() method does not actually 'flush' the buffer. Since flushing in a ZMQ context is
            more or less irrelevant (just like it is for Windows serial context), this method is not implemented.

            The method will NOT work unless ZeroMQ connection was established by calling the open() method. If the class
            is initialized with auto_open flag set to True, the connection has been opened during class initialization.
            Otherwise, this needs to be done manually.

        Args:
            data: A python 'bytes' object that holds the binary data to be sent.

        Raises:
            RuntimeError: If this method is called before the connection has been established by calling the open()
                method.
            TypeError: If the input data is not a 'bytes' object.
        """

        # Prevents interfacing with a closed port.
        if not self.__is_open:
            error_message = (
                f"Unable to write data to a closed connection. Use open() method to open the zmq connection prior "
                f"to calling this method."
            )
            raise RuntimeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Prevents sending incorrectly typed data
        elif not isinstance(data, bytes):
            error_message = (
                f"Invalid 'data' object ({type(data)}) encountered when writing data to the socket buffer. Data should "
                f"be a python 'bytes' object."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        self.__socket.send(data)

    def clear_buffer(self) -> None:
        """A utility method that completely clears the reception circular buffer of the class."""
        with self.__lock:
            self.__circular_buffer.clear()


class ZeroMQSerial:
    """Establishes and facilitates bidirectional communication with another local or remote process running a
    compatible ZeroMQ binding.

    This class mimics the pySerial's Serial class in purpose and application. It establishes an exclusive pairing with
    another ZeroMQ instance using the input address and protocol (port). This allows bidirectionally communicating
    with any process on the same local or remote machine. Anything that can run ZeroMQ (and most things can) will be
    able to communicate with an instance of this class.

    Notes:
        This class is designed to be paired up with a pySerial Serial class communicating with microcontrollers. It
        shares most of its bindings with how Serial class works, which allows SerializedTransferProtocol class to
        use these communication backends interchangeably. This allows using the same high-level API across the library
        to communicate with microcontrollers and other applications (for example Unity game-engine).

    Args:
        address: The protocol and full address of the port to connect to, eg: "tcp://127.0.0.1:5555".
        connection_mode: The 'role' of the class instance. Available options are 'host' and 'client'. Since the class
            uses the 'pair' socket mode, there can only be one host and one client for each unique address and port. The
            socket that is instantiated first should be set as a host and the socket that is instantiated second should
            be set as a 'client'. The only difference between 'host' and 'client' for this communication mode is the
            order of instantiation.
        timeout: The number of seconds(!) to wait for the requested number of bytes to become available, when reading
            data from the class reception buffer. At minimum this can be set to 0 to completely disable waiting.
            Defaults to 0.
        buffer_size: The size of the circular reception buffer. The buffer is implemented as a deque object and, if
            allowed to fill, will start discarding data from the beginning of the queue to accept more data. This size
            determines the number of bytes that, at a maximum, can occupy the buffer before it starts discarding data.
            Adjust this size to the minimum size appropriate for your use case. Defaults to 4096 (~4 kb).

    Attributes:
        __port: The address used to establish the socket connection. Stores the input address provided by the user as
            class initialization argument so that it can be used by the open() method.
        __connection_mode: Stores the mode to be used by the thread when establishing connection. This is also used by
            the open() method to establish the connection.
        __timeout: Stores the timeout (in seconds) to be used when waiting for data to become available for reading.
            This is used by the read() method.
        __context: Initializes to a placeholder. Following open() method call, stores the context used to generate
            sockets. Maintained until the close() method is used to terminate it.
        __socket: Initializes to a placeholder. Following open() method call, the socket is used to send and receive
            data over the provided port address. Maintained until the close() method is used to terminate it.
        __timer: An instance of the ElapsedTimer class (configured to use second precision), used to time data
            reading delays (if timeout is above 0).
        __circular_buffer: An instance of the deque object, which is used as a temporary buffer to which the receiver
            method continuously appends the data received from the connected socket. Uses 'buffer_size' class
            initialization argument to determine the maximum number of bytes to store in buffer.
        __is_open: A boolean flag that tracks whether the open() method has been used to establish the connection. If
            this flag is set to 'False', read() and write() method cannot be used.
        __stop_flag: A boolean flag used by the close() method to shut down the receiver thread. This allows using
            close() method to gracefully shut down the connection without deleting the class.
        __lock: A lock object used to synchronize the receiver thread with other class methods that need to access
            the circular_buffer to prevent race_conditions and other synchronization issues of multithreaded execution.
        __receiver_thread: A thread that continuously runs '__receiver' method until terminated via the __stop_flag
            attribute. The thread is used to continuously monitor and fetch data transmitted by the connected socket,
            which allows the class to function like a pySerial Serial binding does, where it constantly polls and moves
            the received data to the local reception buffer.

    Raises:
        TypeError: If any of the class initialization arguments are not of a correct type.
        ValueError: If any of the class initialization arguments are set to an incorrect value.
    """

    def __init__(
        self,
        address: str,
        connection_mode: Literal["host", "client"],
        timeout: int | float = 0,
    ) -> None:
        # Verifies that the arguments conform to allowed types and values
        if not isinstance(address, str):
            error_message = (
                f"Invalid 'address' argument type ({type(address)}) encountered when initializing ZeroMQSerial class "
                f"object. A string address and port in the format 'tcp://127.0.0.1:5555' is expected."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif connection_mode.lower() not in ("host", "client"):
            error_message = (
                f"Invalid 'connection_mode' argument value ({connection_mode}) encountered when initializing "
                f"ZeroMQSerial class object. Currently, only 'host' or 'client' are accepted as valid argument values."
            )
            ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif not isinstance(timeout, (int, float)):
            error_message = (
                f"Invalid 'timeout' argument type ({type(timeout)}) encountered when initializing ZeroMQSerial class "
                f"object. Only integer and float inputs are accepted as valid argument."
            )
            raise TypeError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))
        elif timeout < 0:
            error_message = (
                f"Invalid 'timeout' argument value ({timeout}) encountered when initializing ZeroMQSerial class "
                f"object. Timeout has to be equal to or greater than 0."
            )
            ValueError(textwrap.fill(error_message, width=120, break_long_words=False, break_on_hyphens=False))

        # Stores configuration parameters to class attributes
        self.__port = address
        self.__connection_mode = connection_mode
        self.__timeout = timeout

        # Initializes local attributes
        self.__context = None
        self.__socket = None
        self.__is_open = False

    def __del__(self) -> None:
        self.close()  # Ensures the connection is properly terminated before the class is deleted

    def open(self) -> None:
        """Connects to the requested port (address) using the requested mode (host or client).

        This method has to be called prior to calling most other methods of the class, as it is required to actually
        establish the socket connection to the provided address. This method allows instantiating the class without
        immediately establishing the connection, which is advantageous in many contexts.

        Raises:
            ZMQError: Typically if the provided port is not valid or already has another host or client (depending on
                the __connection_mode attribute) connected to it.
        """
        # Initializes connection socket
        self.__context = zmq.Context()  # Instantiates a socket generator method
        self.__socket = self.__context.socket(zmq.PAIR)  # Generates a socket

        # Connects to the created socket depending on the connection mode. Note, since the socket is used in Pair mode,
        # re-binding an already hosted address or connecting to an already connected host will trigger a ZeroMQ error.
        # This makes the class behave exactly like an exclusive Serial port does.
        if self.__connection_mode == "host":
            # Hosts BIND to a socket and start listening for other connections
            self.__socket.bind(self.__port)
        else:
            # Clients CONNECT to the host that has bound the socket
            self.__socket.connect(self.__port)

        # Sets the __is_open flag to indicate that the connection has been opened, and it is safe to use other class
        # methods.
        self.__is_open = True

    def close(self) -> None:
        """Terminates the connection to the port (address).

        This method can be used to disconnect from the port connected to by the open() call. Note, this method
        terminates the socket and the receiver thread, and it is safe to garbage-collect the class afterward (in fact,
        close() is used inside the __del__ method). Closing the connection renders most class methods unusable until the
        connection is opened again.

        Notes:
            Calling this method clears any data currently stored inside the circular_buffer in addition to other
            changes. Make sure all important data is consumed before close() is called.
        """
        # If the connection is not open in the first place, there is no need to close it
        if not self.__is_open:
            return

        # Shuts down the connection by closing the socket and terminating the context
        self.__socket.close()
        self.__context.term()

        # Also toggles the stop flag to terminate the receiver thread and waits until the thread joins
        self.__stop_flag = True
        self.__receiver_thread.join()

        # Clears the circular buffer
        self.clear_buffer()

        # Sets the flag to indicate that the connection has been closed and needs to be re-opened before most other
        # class methods can be used again
        self.__is_open = False
