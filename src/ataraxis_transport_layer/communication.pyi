from enum import Enum
from typing import Any
from dataclasses import dataclass

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray

from .transport_layer import SerialTransportLayer as SerialTransportLayer

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
    return_code: np.uint8 = ...
    noblock: np.bool = ...
    cycle: np.bool = ...
    cycle_delay: np.uint32 = ...
    packed_data: NDArray[np.uint8] | None = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""

@dataclass
class DataMessage:
    module_type: np.uint8 = ...
    module_id: np.uint8 = ...
    command: np.uint8 = ...
    event: np.uint8 = ...
    object_size: np.uint8 = ...
    def __repr__(self) -> str: ...

@dataclass
class IdentificationMessage:
    controller_id: np.uint8 = ...

@dataclass
class ReceptionMessage:
    reception_code: np.uint8 = ...

class Protocols(Enum):
    """Stores currently supported protocol codes used in data transmission.

    Each transmitted message starts with the specific protocol code used to instruct the receiver on how to process the
    rest of the data payload. The contents of this enumeration have to mach across all used systems.
    """

    COMMAND: Incomplete
    PARAMETERS: Incomplete
    DATA: Incomplete
    RECEPTION: Incomplete
    IDENTIFICATION: Incomplete

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

    _transport_layer: Incomplete
    data_message: Incomplete
    identification_message: Incomplete
    reception_message: Incomplete
    data_object_index: int
    def __init__(self, usb_port: str, baudrate: int = 115200, maximum_transmitted_payload_size: int = 254) -> None: ...
    @staticmethod
    def list_available_ports() -> tuple[dict[str, int | str], ...]:
        """Provides the information about each serial port addressable through the pySerial library.

        This method is intended to be used for discovering and selecting the serial port 'names' to use with this
        class.

        Returns:
            A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
            port.
        """
    def send_command_message(self, message: CommandMessage) -> None:
        """Packages the input data into a Command message and sends it to the connected microcontroller.

        This method can be used to issue commands to specific hardware Modules of the microcontroller or the Kernel
        class that manages the microcontroller modules.

        Args:
            message: The Command message to send to the microcontroller.
        """
    def send_parameter_message(
        self, module_type: np.uint8, module_id: np.uint8, parameter_object: Any, return_code: np.uint8 = ...
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
