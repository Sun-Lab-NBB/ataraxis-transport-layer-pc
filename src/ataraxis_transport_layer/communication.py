from .transport_layer import SerialTransportLayer
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional
from enum import Enum
from ataraxis_base_utilities import console


class SerialProtocols(Enum):
    """Stores the protocol codes used in data transmission between the PC and the microcontroller over the serial port.

    Each transmitted message starts with the specific protocol code from is enumeration that instructs the receiver on
    how to process the rest of the data payload. The contents of this enumeration have to mach across all used systems.
    """

    COMMAND = np.uint8(1)
    """The protocol used by messages that communicate commands to be executed by the target microcontroller Kernel or
    Module. Commands trigger direct manipulation of the connected hardware, such as engaging breaks or spinning motors. 
    Currently, only the PC can send the commands to the microcontroller."""
    PARAMETERS = np.uint8(2)
    """The protocol used by messages that allow changing the runtime-addressable parameters of the target 
    microcontroller Kernel or Module. For example, this message would be used to adjust the motor speed or the 
    sensitivity of a lick sensor. Currently, only the PC can send updated parameters data to the microcontroller."""
    DATA = np.uint8(3)
    """The protocol used by messages that communicate the microcontroller-originating data. This protocol is used to 
    both send data and communicate runtime errors. Currently, only the microcontroller can send data messages to the 
    PC."""
    RECEPTION = np.uint8(4)
    """The service protocol used by the microcontroller to optionally acknowledge the reception of a Command or 
    Parameters message. This is used to ensure the delivery of critical messages to the microcontroller and, currently, 
    this feature is only supported by Command and Parameters messages."""
    IDENTIFICATION = np.uint8(5)
    """The service protocol used by the microcontroller to respond to the identification request sent by the PC. This 
    is typically used during the initial system architecture setup to map controllers with specific microcode versions 
    to the USB ports they use for communication with the PC."""


@dataclass()
class CommandMessage:
    """The payload structure used by the outgoing Command messages.

    This structure is used to both issue commands to execute and terminate (end) the currently active and all queued
    commands (by setting command variable to 0). Overall, it aggregates and pre-packages the command data into a bytes'
    array to improve transmission speeds during time-critical runtimes.

    Attributes:
        module_type: The type-code of the module to which the command is addressed.
        module_id: The specific module ID within the broader module family specified by module_type.
        command: The unique code of the command to execute. Note, 0 is not a valid command code and will instead be
            interpreted as an instruction to forcibly terminate (stop) any currently running command of the
            addressed Module or Kernel.
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code
            back to the sender PC upon successfully processing the received command. This is to notify the sender
            that the command was received intact, ensuring message delivery. Setting this argument to 0 disables
            delivery assurance.
        noblock: Determines whether the command runs in blocking or non-blocking mode. If set to false, the
            controller runtime will block in-place for any sensor- or time-waiting loops during command execution.
            Otherwise, the controller will run other commands concurrently, while waiting for the block to complete.
        cycle: Determines whether the command is executed once or repeatedly cycled with a certain periodicity.
            Together with cycle_delay, this allows triggering both one-shot and cyclic command runtimes.
        cycle_delay: The period of time, in microseconds, to delay before repeating (cycling) the command. This is
            only used if the cycle flag is True.
        packed_data: Stores the packed attribute data. After this class is instantiated, all attribute values are packed
            into a byte numpy array, which is the preferred TransportLayer format. This allows 'pre-packing' the data at
            the beginning of each time-sensitive runtime to. Do not overwrite this attribute manually!
    """

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = 0
    noblock: np.bool = True
    cycle: np.bool = False
    cycle_delay: np.uint32 = 0
    packed_data: Optional[NDArray[np.uint8]] = None

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""

        # Packages the input data into a byte numpy array. Prepends the 'command' protocol code to the packaged data.
        self.packed_data = np.empty(11, dtype=np.uint8)  # Empty initialization is a bit faster than zeros.
        self.packed_data[0:7] = [
            SerialProtocols.COMMAND.value,
            self.module_type,
            self.module_id,
            self.return_code,
            self.command,
            self.noblock,
            self.cycle,
        ]
        self.packed_data[7:11] = np.frombuffer(self.cycle_delay, dtype=np.uint8)

    def __repr__(self) -> str:
        """Returns a string representation of the CommandMessage object."""
        message = (
            f"CommandMessage(module_type={self.module_type}, module_id={self.module_id}, command={self.command}, "
            f"return_code={self.return_code}, noblock={self.noblock}, cycle={self.cycle} "
            f"cycle_delay={self.cycle_delay} us)."
        )
        return message


@dataclass
class ParametersMessage:
    """The payload structure used by the outgoing Parameters messages.

    This structure is used to update addressable runtime parameters of microcontroller Kernel and Modules. Regardless
    of the structure used to store the parameters on the microcontroller, this class expects them to be provided as a
    tuple of numpy scalars. These scalars are automatically serialized into a byte array for efficient transmission
    during time-critical runtimes and can then be deserialized into the required data format by the microcontroller.

    Attributes:
        module_type: The type-code of the module to which the parameters are addressed.
        module_id: The specific module ID within the broader module family specified by module_type.
        parameter_data: A tuple of parameter values to send. Each value will be serialized into bytes and sequentially
            written into the payload Parameters object memory. Subsequently, they will be deserialized by the
            microcontroller in the same order as they were written.
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code
            back to the sender PC upon successfully processing the received parameters. This is to notify the sender
            that the parameters were received intact, ensuring message delivery. Setting this argument to 0 disables
            delivery assurance.
        packed_data: Stores the packed attribute data. After this class is instantiated, all attribute values are packed
            into a byte numpy array, which is the preferred TransportLayer format. This allows 'pre-packing' the data at
            the beginning of each time-sensitive runtime. Do not overwrite this attribute manually!
        parameters_size: The size, in bytes, of the serialized parameters data object. This is calculated automatically
             during data packaging and on the microcontroller interpreted as the 'object_size' variable.
    """

    module_type: np.uint8
    module_id: np.uint8
    parameter_data: tuple[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool, ...]
    return_code: np.uint8 = 0
    packed_data: Optional[NDArray[np.uint8]] = None
    parameters_size: np.uint8 = 0

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""

        # Converts parameters to byte arrays
        byte_parameters = [np.frombuffer(np.array([param]), dtype=np.uint8).copy() for param in self.parameter_data]

        # Calculates total size of serialized parameters
        self.parameters_size = np.uint8(sum(param.size for param in byte_parameters))

        # Pre-allocates the full array with exact size (header and parameters)
        self.packed_data = np.empty(5 + self.parameters_size, dtype=np.uint8)

        # Packs the header data into the precreated array
        self.packed_data[0:5] = [
            SerialProtocols.COMMAND.value,
            self.module_type,
            self.module_id,
            self.return_code,
            self.parameters_size,
        ]

        # Loops over and sequentially packs parameter data into the array.
        current_position = 5
        for param_bytes in byte_parameters:
            param_size = param_bytes.size
            self.packed_data[current_position : current_position + param_size] = param_bytes
            current_position += param_size

    def __repr__(self) -> str:
        """Returns a string representation of the ParametersMessage object."""
        message = (
            f"ParametersMessage(module_type={self.module_type}, module_id={self.module_id}, "
            f"return_code={self.return_code}, parameter_object_size={self.parameters_size} bytes)."
        )
        return message


@dataclass()
class IdentificationMessage:
    """The payload structure used by the incoming Identification messages.

    This structure is used to parse the incoming service messages that communicate the unique ID of the microcontroller.
    These messages are sent in response to the 'Identification' Kernel-addressed command and allow mapping controllers
    to interface USB ports. Additionally, since controller IDs are hardcoded in microcontroller firmware, these codes
    help track which AtaraxisMicroController (AXMC) firmware is running on the given controller.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        controller_id: The unique ID of the microcontroller. This ID is hardcoded in the microcontroller firmware
            and helps track which AXMC firmware is running on the given controller.
    """

    message: NDArray[np.uint8]
    controller_id: np.uint8

    def __repr__(self) -> str:
        """Returns a string representation of the IdentificationMessage object."""
        message = f"IdentificationMessage(controller_id_code={self.controller_id})."
        return message


@dataclass
class ReceptionMessage:
    """The payload structure used by the incoming Reception messages.

    This structure is used to parse the incoming service messages that return the reception code originally sent as
    part of the outgoing Command or Parameters messages. This is used to verify that the controller has received and
    successfully parsed a particular message sent by PC, ensuring critical message delivery.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        reception_code: The reception code originally sent as part of the outgoing Command or Parameters messages.
    """

    message: NDArray[np.uint8]
    reception_code: np.uint8

    def __repr__(self) -> str:
        """Returns a string representation of the ReceptionMessage object."""
        message = f"ReceptionMessage(reception_code={self.reception_code})."
        return message


@dataclass
class DataMessage:
    """The payload structure used by the incoming Data messages.

    This structure is used to parse the incoming data messages, which are used to communicate microcontroller-recorded
    data and runtime errors to the PC. Unlike other currently supported message structures, this structure is not able
    to fully parse the incoming message at instantiation, as each data message contains a unique data object that
    differs for each module-command-event combination. Use the parse_data_object() Communication class method
    to parse the data object if you need to know its value(s) during runtime.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        module_type: The type-code of the module which sent the data message.
        module_id: The specific module ID within the broader module family specified by module_type.
        command: The unique code of the command that was executed by the module that sent the data message.
        event: The unique byte-code of the event that prompted sending the data message. Event codes are unique within
            each executed command cycle (each command can have multiple data events).
        object_size: The size of the serialized data object transmitted with the message, in bytes.
    """

    message: NDArray[np.uint8]
    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    event: np.uint8
    object_size: np.uint8

    def __repr__(self):
        message = (
            f"DataMessage(module_type={self.module_type}, module_id={self.module_id}, command={self.command}, "
            f"event={self.event}, object_size={self.object_size})."
        )
        return message


class SerialCommunication:
    """Wraps a specialized SerialTransportLayer instance and exposes methods that allow communicating with a
    connected microcontroller system running project Ataraxis microcode.

    This class is built on top of the SerialTransportLayer and is designed to provide a default communication
    interface. It provides a set of predefined message structures designed to efficiently integrate with the existing
    Ataraxis Micro Controller (AxMC) codebase. Overall, this class provides a stable API that can be used to communicate
    with any AXMC system.

    Notes:
        This class is explicitly designed to use the same parameters as the Communication class used by the
        microcontroller version of this library. Do not modify this class unless you know what you are doing.
        Modifications to this class will likely also require modifying some or all of the core classes that manage
        microcontroller runtime.

    Args:
        usb_port: The name of the USB port to connect to, e.g.: 'COM3' or '/dev/ttyUSB0'. You can use the
            list_available_ports() class method to get a list of discoverable serial port names.
        baudrate: The baudrate to be used to communicate with the Microcontroller. Should match the value used by
            the microcontroller for UART ports, ignored for USB ports. Note, the appropriate baudrate for any UART-using
            microcontroller is usually 115200.
        maximum_transmitted_payload_size: The maximum size of the payload that can be transmitted in a single data
            message. This is used to optimize memory usage and prevent overflowing the microcontroller's buffer.
            The default value is 254, which is a typical size for most microcontroller systems.

    Attributes:
        _transport_layer: A SerialTransportLayer instance that exposes the low-level methods that handle bidirectional
            communication with the microcontroller.
        data_message: A DataMessage instance used to store incoming data payloads.
        identification_message: An IdentificationMessage instance used to store incoming controlled ID payloads.
        reception_message: An ReceptionMessage instance used to store incoming message reception code payloads.
        data_object_index: Stores the index of the data object in the received DataMessage payloads. This is needed
            as data messages are processed in two steps: the first extracts the header structure that stores the ID
            information, and the second is used to specifically read the stored data object. This is similar to how
            the microcontrollers handle Parameters messages.
        _service_message_prototype: The numpy array prototype used to optimize receiving service (e.g.: Identification)
            messages.
    """

    def __init__(
        self,
        usb_port: str,
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
    ) -> None:
        # Specializes the TransportLayer to mostly match a similar specialization carried out by the microcontroller
        # Communication class.
        self._transport_layer = SerialTransportLayer(
            port=usb_port,
            baudrate=baudrate,
            polynomial=np.uint16(0x1021),
            initial_crc_value=np.uint16(0xFFFF),
            final_crc_xor_value=np.uint16(0x0000),
            maximum_transmitted_payload_size=maximum_transmitted_payload_size,
            minimum_received_payload_size=2,  # Protocol (1) and Service Message byte-code (1), 2 bytes total
            start_byte=129,
            delimiter_byte=0,
            timeout=20000,
            test_mode=False,
        )

        # Pre-initializes structures used for processing received data
        self.data_message = DataMessage(
            message=np.empty(7, dtype=np.uint8),
            module_type=np.uint8(0),
            module_id=np.uint8(0),
            command=np.uint8(0),
            event=np.uint8(0),
            object_size=np.uint8(0),
        )
        self.identification_message = IdentificationMessage(
            message=np.empty(2, dtype=np.uint8),
            controller_id=np.uint8(0),
        )
        self.reception_message = ReceptionMessage(
            message=np.empty(2, dtype=np.uint8),
            reception_code=np.uint8(0),
        )
        self.data_object_index = 6

        # Unlike DataMessages, all service messages (Identification and Reception) have the same fixed size of 2 bytes.
        # Therefore, it is possible to only initialize the prototype for reading these messages once, during class
        # initialization.
        self._service_message_prototype = np.empty(2, np.uint8)

    @staticmethod
    def list_available_ports() -> tuple[dict[str, int | str], ...]:
        """Provides the information about each serial port addressable through the pySerial library.

        This method is intended to be used for discovering and selecting the serial port 'names' to use with this
        class.

        Returns:
            A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
            port.
        """
        # The method itself is defined in TransportLayer class, this wrapper just calls that method
        return SerialTransportLayer.list_available_ports()

    def send_message(self, message: CommandMessage | ParametersMessage) -> None:
        """Packages the input Command or Parameters message data into a payload and sends it to the connected
        microcontroller.

        This method transmits any outgoing message format to the microcontroller. To do so, it relies on every valid
        message structure exposing a packed_data attribute, that contains the serialized payload data to be sent.
        Overall, this method is a wrapper around the SerialTransportLayer's write_data() and send_data() methods.

        Args:
            message: The Command or Parameters message to send to the microcontroller.
        """
        # Writes the pre-packaged data into the transmission buffer.
        self._transport_layer.write_data(data_object=message.packed_data)

        # Constructs and sends the data message to the connected system.
        self._transport_layer.send_data()

    def receive_message(self) -> Optional[DataMessage | IdentificationMessage | ReceptionMessage]:
        """Receives the incoming message from the connected microcontroller and parses it into one of the pre-allocated
        class message attributes.

        This method receives all valid incoming message structures. To do so, it uses the protocol code, assumed to be
        stored in the first variable of each payload, to determine how to parse the data.

        Notes:
            This method does not fully parse incoming DataMessages, which requires knowing the prototype for the
            included data object. For data that only needs to be logged, it is more efficient to not extract the
            specific data object during online processing (the data is logged as serialized bytes payloads). For data
            messages whose' objects are used during runtime, call the extract_data_object() method as necessary to parse
            the data object value(s).

        Returns:
            An instance of DataMessage, IdentificationMessage, or ReceptionMessage structures that contain the extracted
            data, or None, if no message was received.

        Raises:
            ValueError: If the received protocol code is not recognized.

        """
        # Attempts to receive the data message. If there is no data to receive, returns None. This is a non-error,
        # no-message return case.
        if not self._transport_layer.receive_data():
            return None

        # If the data was received, first reads the protocol code, expected to be found as the first value of every
        # incoming payload. The protocol is a byte-value, so uses np.uint8 prototype.
        protocol, next_index = self._transport_layer.read_data(np.uint8(0), start_index=0)

        # Since received data is logged as serialized payloads, precreates the array necessary to extract the entire
        # received payload. Since TransportLayer known how many payload bytes it received, this property is used to
        # determine the prototype array size. For data messages, the array has to be initialized de-novo for each
        # method runtime due to the varying size of the data object.
        message_prototype = np.empty(self._transport_layer.bytes_in_reception_buffer, dtype=np.uint8)

        # Uses the extracted protocol value to determine the type of the received message and process the received data.
        if protocol == SerialProtocols.DATA.value:
            # Note, for Data messages, this is not the entire Data message. If data object,
            # extract_data_object() method needs to be called next
            message_data = self._transport_layer.read_data(message_prototype, start_index=0)
            return False, 0
        elif protocol == SerialProtocols.RECEPTION.value:
            self.reception_message
            self.reception_message.reception_code, _ = self._transport_layer.read_data(np.uint8(0), start_index=0)
            return True, protocol
        elif protocol == SerialProtocols.IDENTIFICATION.value:
            self.identification_message.controller_id, _ = self._transport_layer.read_data(np.uint8(0), start_index=0)
            return True, protocol
        else:
            message = (
                f"Unable to recognize the protocol code {protocol} of the received message. Currently, only the codes "
                f"available through the Protocols enumeration are supported."
            )
            console.error(message, error=ValueError)


# def extract_data_object(
#         self,
#         prototype_object: np.unsignedinteger[Any] | np.signedinteger[Any] | np.floating[Any] | np.bool |
#                           NDArray[Any],
# ) -> np.unsignedinteger[Any] | np.signedinteger[Any] | np.floating[Any] | np.bool | NDArray[Any]:
#     """Reconstructs the data object from the serialized data bytes using the provided prototype.
#
#     This step completed data message reception by processing the additional message data. This has to be carried
#     out separately, as data object structure is not known until the exact object prototype is determined based on
#     the module-command-event ID information extracted from the message.
#
#     Args:
#         prototype_object: The prototype object that will be used to format the extracted data. The appropriate
#             prototype for data extraction depends on the data format used by the sender module and has to be
#             determined individually for each received data message. Currently, only numpy scalar or array prototypes
#             are supported.
#
#     Raises:
#         ValueError: If the size of the extracted data does not match the object size declared in the data message.
#     """

# # If the provided prototype's byte-size does not match the object size, raises an error.
# if self.object_size != prototype_object.nbytes:
#     message = (
#         "Unable to extract the requested data object from the received message payload. The size of the object "
#         f"declared by the incoming data message {self.object_size} does not match the size of the provided "
#         f"prototype {prototype_object.size} (in bytes). This may indicate that the data was corrupted in "
#         f"transmission."
#     )
#     console.error(message, error=ValueError)
#
# # Reconstructs the prototype object using the serialized data
# target_dtype = prototype_object.dtype
# data_object = np.frombuffer(self.data_object, dtype=target_dtype)
#
# # If the object is a one-element array, casts it to a scalar numpy type
# if data_object.size == 1:
#     return data_object[0].copy()
#
# # Otherwise, returns the object as a numpy array.
# return data_object.copy()
