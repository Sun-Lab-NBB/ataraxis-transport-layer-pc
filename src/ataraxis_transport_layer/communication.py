from .transport_layer import SerialTransportLayer
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional
from enum import Enum
from ataraxis_base_utilities import console


@dataclass()
class CommandMessage:
    """The payload structure used by the outgoing Command messages.

    Currently, only the PC can send command messages. This structure is used to both issue commands to execute and
    terminate (end) a currently active and all queued commands (by setting command to 0).

    Attributes:
        module_type: The type-code of the module to which the command is addressed.
        module_id: The specific module ID within the broader module family specified by module_type.
        command: The unique code of the command to execute. Note, 0 is not a valid command code and will instead be
            interpreted as an instruction to forcibly terminate (stop) any currently running command of the
            addressed Module or Kernel.
        return_code: When this argument is set to a value other than 0, the microcontroller will send this code
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
            the beginning of each time-sensitive runtime. Do not overwrite this attribute manually!
    """
    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = 0
    noblock: np.bool = True
    cycle: np.bool = False
    cycle_delay: np.uint32 = 0
    packed_data: Optional[NDArray[np.uint8]] = None

    def __post_init__(self):
        """Packs the data into the numpy array to optimize future transmission speed."""

        # Packages the input data into a byte numpy array. Prepends the 'command' protocol code to the packaged data.
        self.packed_data = np.empty(11, dtype=np.uint8)
        self.packed_data[0] = Protocols.COMMAND.value
        self.packed_data[1] = self.module_type
        self.packed_data[2] = self.module_id
        self.packed_data[3] = self.return_code
        self.packed_data[4] = self.command
        self.packed_data[5] = self.noblock
        self.packed_data[6] = self.cycle
        self.packed_data[7] = self.cycle_delay


@dataclass
class DataMessage:
    module_type: np.uint8 = np.uint8(0)
    module_id: np.uint8 = np.uint8(0)
    command: np.uint8 = np.uint8(0)
    event: np.uint8 = np.uint8(0)
    object_size: np.uint8 = np.uint8(0)

    def __repr__(self):
        message = (
            f"DataMessage(module_type={self.module_type}, module_id={self.module_id}, command={self.command}, "
            f"event={self.event}, object_size={self.object_size})."
        )
        return message


@dataclass
class IdentificationMessage:
    controller_id: np.uint8 = np.uint8(0)


@dataclass
class ReceptionMessage:
    reception_code: np.uint8 = np.uint8(0)


class Protocols(Enum):
    """Stores currently supported protocol codes used in data transmission.

    Each transmitted message starts with the specific protocol code used to instruct the receiver on how to process the
    rest of the data payload. The contents of this enumeration have to mach across all used systems.
    """

    COMMAND = np.uint8(1)
    """The protocol used by messages that communicate a command to be executed by the target microcontroller Kernel or
    Module class instance. Commands trigger direct manipulation of the connected hardware, such as engaging breaks or
    spinning motors. Currently, only the PC can send the commands to the microcontroller."""
    PARAMETERS = np.uint8(2)
    """The protocol used by messages that allow changing the runtime-addressable parameters of the target 
    microcontroller Kernel or Module class instance. For example, this message would be used to adjust the motor speed
    or the sensitivity of a lick sensor. Currently, only the PC can set the parameters of the microcontroller."""
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


class SerialCommunication:
    """Specializes an instance of the SerialTransportLayer and exposes methods that allow communicating with a
    connected microcontroller system running project Ataraxis code.

    This class is built on top of the SerialTransportLayer. It provides a set of predefined message structures designed
    to efficiently integrate with the existing Ataraxis Micro Controller (AxMC) codebase. Overall, this class provides a
    stable API that can be used to communicate with any AXMC system.

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
        _transport_layer: An instance of the SerialTransportLayer class used for communication with the microcontroller.
        data_message: A DataMessage instance used to optimize processing received DataMessages.
        identification_message: An IdentificationMessage instance used to optimize processing received service messages
            that communicate connected controller ID code.
        reception_message: A ReceptionMessage instance used to optimize processing received service messages used to
            acknowledge the reception of PC-sent command or parameters.
        data_object_index: Stores the index of the data object in the received DataMessage payloads. This is needed
            as data messages are processed in two steps: the first extracts the header structure that stores the ID
            information, and the second is used to specifically read the stored data object. This is similar to how
            the microcontrollers handle Parameters messages.

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
            minimum_received_payload_size=2,  # Protocol (1) and Service code (1)
            start_byte=129,
            delimiter_byte=0,
            timeout=20000,
            test_mode=False,
        )

        self.data_message = DataMessage()
        self.identification_message = IdentificationMessage()
        self.reception_message = ReceptionMessage()

        self.data_object_index = 6

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

    def send_command_message(self, message: CommandMessage) -> None:
        """Packages the input data into a Command message and sends it to the connected microcontroller.

        This method can be used to issue commands to specific hardware Modules of the microcontroller or the Kernel
        class that manages the microcontroller modules.

        Args:
            message: The Command message to send to the microcontroller.
        """
        # Writes the packaged data into the transmission buffer.
        self._transport_layer.write_data(data_object=message.packed_data)

        # Constructs and sends the data message to the connected system.
        self._transport_layer.send_data()

    def send_parameter_message(
            self,
            module_type: np.uint8,
            module_id: np.uint8,
            parameter_object: Any,
            return_code: np.uint8 = np.uint8(0),
    ) -> None:
        """Packages the input data into a Parameters message and sends it to the connected microcontroller.

        This method can be used to send parameters to specific hardware Modules of the microcontroller or the Kernel
        class that manages the microcontroller modules.

        Notes:
            It is expected that all addressable parameters of a given module are set at the same time, as they are often
            stored in a structure or array. The receiver will overwrite the host structure memory with the received data
            in one step. Therefore, make sure that the parameter_object is configured to represent the desired values
            for all parameters stored in this object on the microcontroller.

        Args:
            module_type: The type-code of the module to which the parameters are addressed.
            module_id: The specific module ID within the broader module family specified by module_type.
            parameter_object: The data object containing the parameter to be sent to the microcontroller. Currently,
                this argument only supports numpy scalar or array inputs, or dataclasses that contain only numpy
                scalars or array fields.
            return_code: When this argument is set to a value other than 0, the microcontroller will send this code
                back to the sender PC upon successfully processing the received parameters. This is to notify the sender
                that the parameters were received intact, ensuring message delivery. Setting this argument to 0 disables
                delivery assurance.
        """

        # Ensures the transmission_buffer is cleared
        self.reset_transmission_state()

        # Packages the header data for the parameter message and writes it to the transmission buffer
        packed_data = np.array(
            [
                Protocols.PARAMETERS.value,
                module_type,
                module_id,
                return_code,
            ],
            dtype=np.uint8,
        )

        # noinspection PyTypeChecker
        next_index = self._transport_layer.write_data(data_object=packed_data)

        size_index = next_index
        object_index = next_index + 1
        next_index = self._transport_layer.write_data(data_object=parameter_object, start_index=object_index)

        self._transport_layer.write_data(data_object=np.uint8(next_index - object_index), start_index=size_index)

        self._transport_layer.send_data()

    def receive_message(self) -> tuple[bool, DataMessage | IdentificationMessage | ReceptionMessage | int]:
        """Receives and processes the message from the connected microcontroller.

        This method determines the type of the received message and extracts message data into the appropriate
        structure.

        Notes:
            The Data messages also require extract_data_object() method to be called to extract the data object. This
            method will only extract the data message header structure necessary to identify the sender and the type
            of the transmitted data object.

        Returns:
            An instance of DataMessage, IdentificationMessage, or ReceptionMessage structures that contain the extracted
            data, or None, if no message was received.

        Raises:
            ValueError: If the received protocol code is not recognized.

        """
        # Attempts to receive the data message. If there is no data to receive, returns None
        if not self._transport_layer.receive_data():
            return False, 0

        # If the data was received, first reads the protocol code, that is expected to be the first value of every
        # message payload
        protocol = np.uint8(0)
        protocol, next_index = self._transport_layer.read_data(protocol, start_index=0)

        data: np.uint8 | NDArray[np.uint8]

        # Uses the protocol to determine the type of the received message and read the data
        if protocol == Protocols.DATA.value:
            # Note, for Data messages, this is not the entire Data message. To process the data object,
            # extract_data_object() method needs to be called next
            # data = self._transport_layer.read_data(np.uint8(0), start_index=0)  # TODO
            return False, 0
        elif protocol == Protocols.RECEPTION.value:
            self.reception_message.reception_code, _ = self._transport_layer.read_data(np.uint8(0), start_index=0)
            return True, protocol
        elif protocol == Protocols.IDENTIFICATION.value:
            self.identification_message.controller_id, _ = self._transport_layer.read_data(np.uint8(0), start_index=0)
            return True, protocol
        else:
            message = (
                f"Unable to recognize the protocol code {protocol} of the received message. Currently, only the codes "
                f"available through the Protocols enumeration are supported."
            )
            console.error(message, error=ValueError)

    def extract_data_object(self, data_message: DataMessage, prototype_object: Any) -> Any:
        """Extracts the data object from the last received message and formats it to match the structure of the
        provided prototype object.

        This method completes receiving the DataMessage by extracting the transmitted data object. It should be called
        after the receive_message(0 method returns a DataMessage header structure.

        Args:
            data_message: The DataMessage structure that stores the necessary ID information to extract the data object.
                This structure is returned by the receive_message() method.
            prototype_object: The prototype object that will be used to format the extracted data. This object depends
                on the specific data format used by the sender module and has to be determined by the user for each
                received data message.

        Raises:
            ValueError: If the size of the extracted data does not match the object size declared in the data message.
        """

        # Attempts to extract the data and save it into the format that matches the prototype object
        extracted_object, next_index = self._transport_layer.read_data(
            prototype_object, start_index=self._data_object_index
        )

        # Verifies that the size of the extracted object matches the size declared in the data message
        extracted_size = next_index - self._data_object_index

        if extracted_size != data_message.object_size:
            message = (
                "Unable to extract the requested data object from the received message payload. The size of the object "
                f"declared by the incoming data message {data_message.object_size} does not match the factual size of "
                f"the extracted data {extracted_size} (in bytes). This may indicate a data corruption."
            )
            console.error(message, error=ValueError)
