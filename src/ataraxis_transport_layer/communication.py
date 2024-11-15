"""This module provides the SerialCommunication and UnityCommunication classes, as well as helper message layout
structures, that enable communication between various project Ataraxis systems.

Currently, SerialCommunication is used to interface between the PC and the MicroController, while UnityCommunication is
used to interface between python PC code and Unity game engine running Virtual Environment tasks.
"""
from queue import Queue
from typing import Any, Union
import datetime
from datetime import timezone
from dataclasses import field, dataclass
from multiprocessing import Queue as MPQueue

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
import paho.mqtt.client as mqtt
from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import NestedDictionary

from .transport_layer import SerialTransportLayer, list_available_ports


@dataclass(frozen=True)
class SerialProtocols:
    """Stores the protocol codes used in data transmission between the PC and the microcontroller over the serial port.

    Each sent and received message starts with the specific protocol code from this class that instructs the receiver on
    how to process the rest of the data payload. The codes available through this class have to match the contents of
    the kProtocols Enumeration available from the AtaraxisMicroController library (communication_assets namespace).

    Attributes:
        kUndefined: Not a valid protocol code. This is used to initialize the Communication class of the
            microcontroller.
        kRepeatedModuleCommand: Protocol for sending Module-addressed commands that should be repeated
            (executed recurrently).
        kOneOffModuleCommand: Protocol for sending Module-addressed commands that should not be repeated
            (executed only once).
        kDequeueModuleCommand: Protocol for sending Module-addressed commands that remove all queued commands
            (including recurrent commands).
        kKernelCommand: Protocol for sending Kernel-addressed commands. All Kernel commands are always non-repeatable
            (one-shot).
        kModuleParameters: Protocol for sending Module-addressed parameters. This relies on transmitting arbitrary
            sized parameter objects likely to be unique for each module type (family).
        kKernelParameters: Protocol for sending Kernel-addressed parameters. The parameters transmitted via these
            messages will be used to overwrite the global parameters shared by the Kernel and all Modules of the
            microcontroller (global runtime parameters).
        kModuleData: Protocol for receiving Module-sent data or error messages that include an arbitrary data object in
            addition to event state-code.
        kKernelData: Protocol for receiving Kernel-sent data or error messages that include an arbitrary data object in
            addition to event state-code.
        kModuleState: Protocol for receiving Module-sent data or error messages that do not include additional data
            objects.
        kKernelState: Protocol for receiving Kernel-sent data or error messages that do not include additional data
            objects.
        kReceptionCode: Protocol used to ensure that the microcontroller has received a previously sent command or
            parameter message. Specifically, when an outgoing message includes a reception_code, this code is
            transmitted back to the PC using this service protocol to acknowledge message reception.
        kIdentification: Protocol used to identify the controller connected to a particular USB port. This service
            protocol is used by the controller that receives the 'Identify' Kernel-addressed command.
    """

    kUndefined: np.uint8 = np.uint8(0)
    kRepeatedModuleCommand: np.uint8 = np.uint8(1)
    kOneOffModuleCommand: np.uint8 = np.uint8(2)
    kDequeueModuleCommand: np.uint8 = np.uint8(3)
    kKernelCommand: np.uint8 = np.uint8(4)
    kModuleParameters: np.uint8 = np.uint8(5)
    kKernelParameters: np.uint8 = np.uint8(6)
    kModuleData: np.uint8 = np.uint8(7)
    kKernelData: np.uint8 = np.uint8(8)
    kModuleState: np.uint8 = np.uint8(9)
    kKernelState: np.uint8 = np.uint8(10)
    kReceptionCode: np.uint8 = np.uint8(11)
    kIdentification: np.uint8 = np.uint8(12)

    def write_protocol_codes(self, code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the 'communication.protocols' section of the core_codes_map dictionary with data.

        The message protocols determine incoming and outgoing message payload structures and allow optimizing certain
        forms of communication by limiting the payload length. Knowing the meaning (and structure) of each message
        protocol is essential for being able to deserialize the logged data, which is stored as raw byte-serialized
        message payloads.

        Make sure this method matches the actual state of the communication_assets::kProtocols enumeration from the
        AtaraxisMicroController library!

        Args:
            code_dictionary: The dictionary to be filled with communication prototype codes.

        Returns:
            The updated dictionary with communication protocol codes information filled.
        """
        section = "communication.protocols.kUndefined"
        description = (
            "Not a valid protocol code. This is used to initialize the Communication class of the microcontroller."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kUndefined)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=False)

        section = "communication.protocols.kRepeatedModuleCommand"
        description = "Protocol for sending Module-addressed commands that should be repeated (executed recurrently)."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kRepeatedModuleCommand)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=True)

        section = "communication.protocols.kOneOffModuleCommand"
        description = "Protocol for sending Module-addressed commands that should not be repeated (executed only once)."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kOneOffModuleCommand)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=True)

        section = "communication.protocols.kDequeueModuleCommand"
        description = (
            "Protocol for sending Module-addressed commands that remove all queued commands (including recurrent "
            "commands)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kDequeueModuleCommand)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=True)

        section = "communication.protocols.kKernelCommand"
        description = (
            "Protocol for sending Kernel-addressed commands. All Kernel commands are always non-repeatable (one-shot)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kKernelCommand)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=True)

        section = "communication.protocols.kModuleParameters"
        description = (
            "Protocol for sending Module-addressed parameters. This relies on transmitting arbitrary sized parameter "
            "objects likely to be unique for each module type (family)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kModuleParameters)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=True)

        section = "communication.protocols.kKernelParameters"
        description = (
            "Protocol for sending Kernel-addressed parameters. The parameters transmitted via these messages will be "
            "used to overwrite the global parameters shared by the Kernel and all Modules of the microcontroller "
            "(global runtime parameters)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kKernelParameters)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=True)

        section = "communication.protocols.kModuleData"
        description = (
            "Protocol for receiving Module-sent data or error messages that include an arbitrary data object in "
            "addition to event state-code."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kModuleData)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=False)

        section = "communication.protocols.kKernelData"
        description = (
            "Protocol for receiving Kernel-sent data or error messages that include an arbitrary data object in "
            "addition to event state-code."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kKernelData)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=False)

        section = "communication.protocols.kModuleState"
        description = (
            "Protocol for receiving Module-sent data or error messages that do not include additional data objects."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kModuleState)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=False)

        section = "communication.protocols.kKernelState"
        description = (
            "Protocol for receiving Kernel-sent data or error messages that do not include additional data objects."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kKernelState)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=False)

        section = "communication.protocols.kReceptionCode"
        description = (
            "Protocol used to ensure that the microcontroller has received a previously sent command or parameter "
            "message. Specifically, when an outgoing message includes a reception_code, this code is transmitted back "
            "to the PC using this service protocol to acknowledge message reception."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kReceptionCode)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=False)

        section = "communication.protocols.kIdentification"
        description = (
            "Protocol used to identify the controller connected to a particular USB port. This service protocol is "
            "used by the controller that receives the 'Identify' Kernel-addressed command"
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kIdentification)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.outgoing", value=False)

        return code_dictionary


@dataclass(frozen=True)
class SerialPrototypes:
    """Stores the protocol codes used in data transmission between the PC and the microcontroller over the serial port.

    Prototype codes are used by Data messages (Kernel and Module) to communicate the structure (prototype) that can be
    used to deserialize the included data object. Transmitting these codes with the message ensures that the receiver
    has the necessary information to decode the data without doing any additional processing. In turn, this allows to
    'inline' the reception procedure to efficiently decode the data object in-place.

    Notes:
        While the use of byte-code limits the number of mapped prototypes to 255 (256 if 0 is made a valid value), this
        number should be enough to support many unique runtime configurations.

    Attributes:
        kOneUnsignedByte: The prototype code for a single uint8_t value.
        _kOneUnsignedBytePrototype: Stores the prototype for kOneUnsignedByte code.
        kTwoUnsignedBytes: The prototype code for an array of two uint8_t values.
        _kTwoUnsignedBytesPrototype: Stores the prototype for kTwoUnsignedBytes code.
        kThreeUnsignedBytes: The prototype code for an array of three uint8_t values.
        _kThreeUnsignedBytesPrototype: Stores the prototype for kThreeUnsignedBytes code.
        kFourUnsignedBytes: The prototype code for an array of four uint8_t values.
        _kFourUnsignedBytesPrototype: Stores the prototype for kFourUnsignedBytes code.
        kOneUnsignedLong: The prototype code for a single uint32_t value.
        _kOneUnsignedLongPrototype: Stores the prototype for kOneUnsignedLong code.
        kOneUnsignedShort: The prototype code for a single uint16_t value.
        _kOneUnsignedShortPrototype: The prototype for kOneUnsignedShort code.
    """

    kOneUnsignedByte: np.uint8 = np.uint8(1)
    _kOneUnsignedBytePrototype: np.uint8 = field(default_factory=lambda: np.uint8(0))

    kTwoUnsignedBytes: np.uint8 = np.uint8(2)
    _kTwoUnsignedBytesPrototype: NDArray[Any] = field(default_factory=lambda: np.empty(shape=2, dtype=np.uint8))

    kThreeUnsignedBytes: np.uint8 = np.uint8(3)
    _kThreeUnsignedBytesPrototype: NDArray[Any] = field(default_factory=lambda: np.empty(shape=3, dtype=np.uint8))

    kFourUnsignedBytes: np.uint8 = np.uint8(4)
    _kFourUnsignedBytesPrototype: NDArray[Any] = field(default_factory=lambda: np.empty(shape=4, dtype=np.uint8))

    kOneUnsignedLong: np.uint8 = np.uint8(5)
    _kOneUnsignedLongPrototype: np.uint32 = field(default_factory=lambda: np.uint32(0))

    kOneUnsignedShort: np.uint8 = np.uint8(6)
    _kOneUnsignedShortPrototype: np.uint16 = field(default_factory=lambda: np.uint16(0))

    def get_prototype(self, code: np.uint8) -> Union[NDArray[np.uint8], np.uint8, np.uint16, np.uint32, None]:
        """Returns the prototype object associated with the input prototype code.

        The prototype object returned by this method can be passed to the reading method of the SerialTransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this dataclass.

        Args:
            code: The prototype byte-code to retrieve the prototype for.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type. If the input code is not one of
            the supported codes, returns None to indicate a matching error.
        """
        if code == self.kOneUnsignedByte:
            return self._kOneUnsignedBytePrototype
        if code == self.kTwoUnsignedBytes:
            return self._kTwoUnsignedBytesPrototype
        if code == self.kThreeUnsignedBytes:
            return self._kThreeUnsignedBytesPrototype
        if code == self.kFourUnsignedBytes:
            return self._kFourUnsignedBytesPrototype
        if code == self.kOneUnsignedLong:
            return self._kOneUnsignedLongPrototype
        if code == self.kOneUnsignedShort:
            return self._kOneUnsignedShortPrototype
        return None

    def write_prototype_codes(self, code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the 'communication.prototypes' section of the core_codes_map dictionary with data.

        The prototypes are used to optimize data message reception and parsing on the PC by providing the necessary
        information as part of the received message.

        Make sure this method matches the actual state of the communication_assets::kPrototypes enumeration from the
        AtaraxisMicroController library!

        Args:
           code_dictionary: The dictionary to be filled with communication prototype codes.

        Returns:
           The updated dictionary with communication prototype codes information filled.
        """
        section = "communication.prototypes.kOneUnsignedByte"
        description = "A single uint8_t value."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kOneUnsignedByte)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.prototype", value=self._kOneUnsignedBytePrototype)

        section = "communication.prototypes.kTwoUnsignedBytes"
        description = "An array made up of two uint8_t values."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kTwoUnsignedBytes)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.prototype", value=self._kTwoUnsignedBytesPrototype)

        section = "communication.prototypes.kThreeUnsignedBytes"
        description = "An array made up of three uint8_t values."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kThreeUnsignedBytes)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype", value=self._kThreeUnsignedBytesPrototype
        )

        section = "communication.prototypes.kFourUnsignedBytes"
        description = "An array made up of four uint8_t values."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kFourUnsignedBytes)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype", value=self._kFourUnsignedBytesPrototype
        )

        section = "communication.prototypes.kOneUnsignedLong"
        description = "A single uint32_t value."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kOneUnsignedLong)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.prototype", value=self._kOneUnsignedLongPrototype)

        section = "communication.prototypes.kOneUnsignedShort"
        description = "A single uint16_t value."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=self.kOneUnsignedShort)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.prototype", value=self._kOneUnsignedShortPrototype)

        return code_dictionary


# Instantiates protocols and prototypes classes to be used by all other classes
prototypes = SerialPrototypes()
protocols = SerialProtocols()


@dataclass(frozen=True)
class RepeatedModuleCommand:
    """Instructs the addressed Module to run the specified command repeatedly (recurrently).

    Attributes:
        module_type: The type-code of the module to which the command is addressed.
        module_id: The specific module ID within the broader module family specified by module_type.
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code back
            to the PC upon successfully processing the received command. This is to notify the sender that the command
            was received intact, ensuring message delivery. Setting this argument to 0 disables delivery assurance.
        command: The unique code of the command to execute. Note, 0 is not a valid command code.
        noblock: Determines whether the command runs in blocking or non-blocking mode. If set to false, the
            controller runtime will block in-place for any sensor- or time-waiting loops during command execution.
            Otherwise, the controller will run other commands concurrently, while waiting for the block to complete.
        cycle_delay: The period of time, in microseconds, to delay before repeating (cycling) the command.
        packed_data: Stores the packed attribute data. During class initialization, all attribute values are packed
            into a byte numpy array, which is the preferred data format for TransportLayer to function with the highest
            efficiency. In turn, this allows 'pre-packing' the data before time-critical runtimes to optimize the
            communication speed during these runtimes. Do not overwrite this attribute manually!
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = np.uint8(0)
    noblock: np.bool_ = np.bool(True)
    cycle_delay: np.uint32 = np.uint32(0)
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=protocols.kRepeatedModuleCommand)

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(10, dtype=np.uint8)
        packed[0:6] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
            self.command,
            self.noblock,
        ]
        packed[6:10] = np.frombuffer(self.cycle_delay.tobytes(), dtype=np.uint8)
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""
        message = (
            f"RepeatedModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock}, cycle_delay={self.cycle_delay} us)."
        )
        return message


@dataclass(frozen=True)
class OneOffModuleCommand:
    """Instructs the addressed Module to run the specified command exactly once (non-recurrently).

    Attributes:
        module_type: The type-code of the module to which the command is addressed.
        module_id: The specific module ID within the broader module family specified by module_type.
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code back
            to the PC upon successfully processing the received command. This is to notify the sender that the command
            was received intact, ensuring message delivery. Setting this argument to 0 disables delivery assurance.
        command: The unique code of the command to execute. Note, 0 is not a valid command code.
        noblock: Determines whether the command runs in blocking or non-blocking mode. If set to false, the
            controller runtime will block in-place for any sensor- or time-waiting loops during command execution.
            Otherwise, the controller will run other commands concurrently, while waiting for the block to complete.
        packed_data: Stores the packed attribute data. During class initialization, all attribute values are packed
            into a byte numpy array, which is the preferred data format for TransportLayer to function with the highest
            efficiency. In turn, this allows 'pre-packing' the data before time-critical runtimes to optimize the
            communication speed during these runtimes. Do not overwrite this attribute manually!
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = np.uint8(0)
    noblock: np.bool_ = np.bool(True)
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=protocols.kOneOffModuleCommand)

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(6, dtype=np.uint8)
        packed[0:6] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
            self.command,
            self.noblock,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the OneOffModuleCommand object."""
        message = (
            f"OneOffModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, return_code={self.return_code}, "
            f"noblock={self.noblock})."
        )
        return message


@dataclass(frozen=True)
class DequeueModuleCommand:
    """Instructs the addressed Module to clear (empty) its command queue.

    Attributes:
        module_type: The type-code of the module to which the command is addressed.
        module_id: The specific module ID within the broader module family specified by module_type.
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code back
            to the PC upon successfully processing the received command. This is to notify the sender that the command
            was received intact, ensuring message delivery. Setting this argument to 0 disables delivery assurance.
        packed_data: Stores the packed attribute data. During class initialization, all attribute values are packed
            into a byte numpy array, which is the preferred data format for TransportLayer to function with the highest
            efficiency. In turn, this allows 'pre-packing' the data before time-critical runtimes to optimize the
            communication speed during these runtimes. Do not overwrite this attribute manually!
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    module_type: np.uint8
    module_id: np.uint8
    return_code: np.uint8 = np.uint8(0)
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=protocols.kDequeueModuleCommand)

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(4, dtype=np.uint8)
        packed[0:4] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""
        message = (
            f"kDequeueModuleCommand(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code})."
        )
        return message


@dataclass(frozen=True)
class KernelCommand:
    """Instructs the Kernel to run the specified command exactly once.

    Currently, the Kernel only supports blocking one-off commands.

    Attributes:
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code back
            to the PC upon successfully processing the received command. This is to notify the sender that the command
            was received intact, ensuring message delivery. Setting this argument to 0 disables delivery assurance.
        command: The unique code of the command to execute. Note, 0 is not a valid command code.
        packed_data: Stores the packed attribute data. During class initialization, all attribute values are packed
            into a byte numpy array, which is the preferred data format for TransportLayer to function with the highest
            efficiency. In turn, this allows 'pre-packing' the data before time-critical runtimes to optimize the
            communication speed during these runtimes. Do not overwrite this attribute manually!
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    command: np.uint8
    return_code: np.uint8 = np.uint8(0)
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=protocols.kKernelCommand)

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        packed = np.empty(3, dtype=np.uint8)
        packed[0:3] = [
            self.protocol_code,
            self.return_code,
            self.command,
        ]
        object.__setattr__(self, "packed_data", packed)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelCommand object."""
        message = (
            f"KernelCommand(protocol_code={self.protocol_code}, command={self.command}, "
            f"return_code={self.return_code})."
        )
        return message


@dataclass(frozen=True)
class ModuleParameters:
    """Instructs the addressed Module to overwrite its custom parameters object with the included object data.

    Attributes:
        module_type: The type-code of the module to which the command is addressed.
        module_id: The specific module ID within the broader module family specified by module_type.
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code back
            to the PC upon successfully processing the received command. This is to notify the sender that the command
            was received intact, ensuring message delivery. Setting this argument to 0 disables delivery assurance.
        parameter_data: A tuple of parameter values to send. Each value will be serialized into bytes and sequentially
            packed into the data object included with the message. Subsequently, the microcontroller will deserialize
            the written parameters in the same order as they were written. Each parameter value has to use the
            appropriate scalar numpy type.
        packed_data: Stores the packed attribute data. During class initialization, all attribute values are packed
            into a byte numpy array, which is the preferred data format for TransportLayer to function with the highest
            efficiency. In turn, this allows 'pre-packing' the data before time-critical runtimes to optimize the
            communication speed during these runtimes. Do not overwrite this attribute manually!
        parameters_size: Stores the size of the parameter object in bytes. This is calculated automatically during data
            packing and is mostly used for debugging purposes.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    module_type: np.uint8
    module_id: np.uint8
    parameter_data: tuple[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool, ...]
    return_code: np.uint8 = np.uint8(0)
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    parameters_size: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=protocols.kModuleParameters)

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        # Converts scalar parameter values to byte arrays (serializes them)
        byte_parameters = [np.frombuffer(np.array([param]), dtype=np.uint8).copy() for param in self.parameter_data]

        # Calculates the total size of serialized parameters in bytes and adds it to the parameters_size attribute
        parameters_size = np.uint8(sum(param.size for param in byte_parameters))
        object.__setattr__(self, "parameters_size", parameters_size)

        # Pre-allocates the full array with exact size (header and parameters object)
        packed_data = np.empty(4 + parameters_size, dtype=np.uint8)

        # Packs the header data into the precreated array
        packed_data[0:4] = [
            self.protocol_code,
            self.module_type,
            self.module_id,
            self.return_code,
        ]

        # Loops over and sequentially appends parameter data to the array.
        current_position = 4
        for param_bytes in byte_parameters:
            param_size = param_bytes.size
            packed_data[current_position : current_position + param_size] = param_bytes
            current_position += param_size

        # Writes the constructed packed data object to the packed_data attribute
        object.__setattr__(self, "packed_data", packed_data)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleParameters object."""
        message = (
            f"ModuleParameters(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, return_code={self.return_code}, "
            f"parameter_object_size={self.parameters_size} bytes)."
        )
        return message


@dataclass(frozen=True)
class KernelParameters:
    """Instructs the Kernel to update the shared DynamicRuntimeParameters object with included data.

    This structure is shared by the Kernel and all custom Modules and is used to set global runtime parameters, such as
    the global lock on sending hardware activation signals. Since this message always targets the same structure, the
    attributes of this class are configured to exactly match the fields of the addressed parameters structure.

    Attributes:
        action_lock: Enables running the controller logic without physically issuing commands, which is especially
            helpful for testing and debugging. Specifically, when this flag is set to True, all hardware-connected
            pins are blocked from writing (emitting HIGH signals). This has no effect on sensor and TTL pins.
        ttl_lock: Same as action_lock, but specifically controls output TTL (Transistor-to-Transistor Logic) pin
            activity. This has no effect on sensor and non-ttl hardware-connected pins.
        return_code: When this attribute is set to a value other than 0, the microcontroller will send this code back
            to the PC upon successfully processing the received command. This is to notify the sender that the command
            was received intact, ensuring message delivery. Setting this argument to 0 disables delivery assurance.
        packed_data: Stores the packed attribute data. During class initialization, all attribute values are packed
            into a byte numpy array, which is the preferred data format for TransportLayer to function with the highest
            efficiency. In turn, this allows 'pre-packing' the data before time-critical runtimes to optimize the
            communication speed during these runtimes. Do not overwrite this attribute manually!
        parameters_size: Stores the size of the parameter object in bytes. This is calculated automatically during data
            packing and is mostly used for debugging purposes.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    action_lock: np.bool
    ttl_lock: np.bool
    return_code: np.uint8 = np.uint8(0)
    packed_data: NDArray[np.uint8] | None = field(init=False, default=None)
    parameters_size: NDArray[np.uint8] | None = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=protocols.kKernelParameters)

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
        # Packs the data into the numpy array. Since parameter count and type are known at initialization, this uses a
        # fixed packing protocol.
        packed_data = np.empty(4, dtype=np.uint8)
        packed_data[0:4] = [self.protocol_code, self.return_code, self.action_lock, self.ttl_lock]

        # Overwrites uninitialized class attributes with data determined above
        object.__setattr__(
            self, "parameters_size", packed_data.nbytes - 2
        )  # -2 to account for protocol and return code
        object.__setattr__(self, "packed_data", packed_data)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelParameters object."""
        message = (
            f"KernelParameters(protocol_code={self.protocol_code}, return_code={self.return_code}, "
            f"parameter_object_size={self.parameters_size} bytes)."
        )
        return message


class ModuleData:
    """Communicates the event state-code of the sender Module and includes an additional data object.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        module_type: The type-code of the module which sent the data message.
        module_id: The specific module ID within the broader module family specified by module_type.
        command: The unique code of the command that was executed by the module that sent the data message.
        event: The unique byte-code of the event that prompted sending the data message. The event-code only needs to
            be unique with respect to the module_type and command combination.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes class.
        _transport_layer: Stores the reference to the TransportLayer class.
    """

    def __init__(self, transport_layer: SerialTransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = protocols.kModuleData

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.module_type: np.uint8 = np.uint8(0)
        self.module_id: np.uint8 = np.uint8(0)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)
        self.data_object: np.unsignedinteger[Any] | NDArray[Any] = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.

        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the static header data from the extracted message
        self.module_type = self.message[1]
        self.module_id = self.message[2]
        self.command = self.message[3]
        self.event = self.message[4]

        # Parses the prototype code and uses it to retrieve the prototype object from the prototypes dataclass instance
        prototype = prototypes.get_prototype(code=self.message[5])

        # If prototype retrieval fails, raises ValueError
        if prototype is None:
            message = (
                f"Invalid prototype code {self.message[5]} encountered when extracting the data object from "
                f"the received ModuleData message sent my module {self.module_id} of type {self.module_type}. All "
                f"data prototype codes have to be available from the SerialPrototypes class to be resolved."
            )
            console.error(message, ValueError)
        else:
            # Otherwise, uses the retrieved prototype to parse the data object
            self.data_object, _ = self._transport_layer.read_data(prototype, start_index=6)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleData object."""
        message = (
            f"ModuleData(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, event={self.event}, "
            f"data_object={self.data_object})."
        )
        return message


class KernelData:
    """Communicates the event state-code of the Kernel and includes an additional data object.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        command: The unique code of the command that was executed by the Kernel when it sent the data message.
        event: The unique byte-code of the event that prompted sending the data message. The event-code only needs to
            be unique with respect to the executed command code.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes class.
        _transport_layer: Stores the reference to the TransportLayer class.
    """

    def __init__(self, transport_layer: SerialTransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = protocols.kKernelData

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)
        self.data_object: np.unsignedinteger[Any] | NDArray[Any] = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the static header data from the extracted message
        self.command = self.message[1]
        self.event = self.message[2]

        # Parses the prototype code and uses it to retrieve the prototype object from the prototypes dataclass instance
        prototype = prototypes.get_prototype(code=self.message[3])

        # If the prototype retrieval fails, raises ValueError.
        if prototype is None:
            message = (
                f"Invalid prototype code {self.message[3]} encountered when extracting the data object from "
                f"the received KernelData message. All data prototype codes have to be available from the "
                f"SerialPrototypes class to be resolved."
            )
            console.error(message, ValueError)

        else:
            # Otherwise, uses the retrieved prototype to parse the data object
            self.data_object, _ = self._transport_layer.read_data(prototype, start_index=4)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelData object."""
        message = (
            f"KernelData(protocol_code={self.protocol_code}, command={self.command}, event={self.event}, "
            f"data_object={self.data_object})."
        )
        return message


class ModuleState:
    """Communicates the event state-code of the sender Module.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        module_type: The type-code of the module which sent the state message.
        module_id: The specific module ID within the broader module family specified by module_type.
        command: The unique code of the command that was executed by the module that sent the state message.
        event: The unique byte-code of the event that prompted sending the state message. The event-code only needs to
            be unique with respect to the module_type and command combination.
    """

    def __init__(self, transport_layer: SerialTransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = protocols.kModuleState

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.module_type: np.uint8 = np.uint8(0)
        self.module_id: np.uint8 = np.uint8(0)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.module_type = self.message[1]
        self.module_id = self.message[2]
        self.command = self.message[3]
        self.event = self.message[4]

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleState object."""
        message = (
            f"ModuleState(module_type={self.module_type}, module_id={self.module_id}, command={self.command}, "
            f"event={self.event})."
        )
        return message


class KernelState:
    """Communicates the event state-code of the Kernel.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        command: The unique code of the command that was executed by the Kernel when it sent the state message.
        event: The unique byte-code of the event that prompted sending the state message. The event-code only needs to
            be unique with respect to the executed command code.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    def __init__(self, transport_layer: SerialTransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = protocols.kKernelState

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.command: np.uint8 = np.uint8(0)
        self.event: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.command = self.message[1]
        self.event = self.message[2]

    def __repr__(self) -> str:
        """Returns a string representation of the KernelState object."""
        message = f"KernelState(command={self.command}, event={self.event})."
        return message


class ReceptionCode:
    """Identifies the connected microcontroller by communicating its unique byte id-code.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        reception_code: The reception code originally sent as part of the outgoing Command or Parameters messages.
    """

    def __init__(self, transport_layer: SerialTransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = protocols.kReceptionCode

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.reception_code: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.reception_code = self.message[1]

    def __repr__(self) -> str:
        """Returns a string representation of the ReceptionCode object."""
        message = f"ReceptionCode(reception_code={self.reception_code})."
        return message


class Identification:
    """Identifies the connected microcontroller by communicating its unique byte id-code.

    For the ID codes to be unique, they have to be manually assigned to the Kernel class of each concurrently
    used microcontroller.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its
    lifetime to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        controller_id: The unique ID of the microcontroller. This ID is hardcoded in the microcontroller firmware
            and helps track which AXMC firmware is running on the given controller.

    """

    def __init__(self, transport_layer: SerialTransportLayer) -> None:
        # Initializes non-placeholder attributes.
        self.protocol_code: np.uint8 = protocols.kIdentification

        # Initializes placeholder attributes. These fields are overwritten with data when update_message_data() method
        # is called.
        self.message: NDArray[np.uint8] = np.empty(1, dtype=np.uint8)
        self.controller_id: np.uint8 = np.uint8(0)

        # Saves transport_layer reference into an attribute.
        self._transport_layer = transport_layer

    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
        # First, uses the payload size to read the entire message into the _message field. The whole message is stored
        # separately from parsed data to simplify data logging
        payload_size = self._transport_layer.bytes_in_reception_buffer
        # noinspection PyTypeChecker
        self.message, _ = self._transport_layer.read_data(data_object=np.empty(payload_size, dtype=np.uint8))

        # Parses the message data
        self.controller_id = self.message[1]

    def __repr__(self) -> str:
        """Returns a string representation of the Identification object."""
        message = f"Identification(controller_id={self.controller_id})."
        return message


class SerialCommunication:
    """Wraps a SerialTransportLayer class instance and exposes methods that allow communicating with the connected
    microcontroller running AtaraxisMicroController firmware.

    This class is built on top of the SerialTransportLayer and is designed to provide a default communication
    interface for project Ataraxis systems. It provides a set of predefined message structures designed to efficiently
    integrate with the existing Ataraxis Micro Controller (AXMC) codebase. Overall, this class provides a stable API
    that can be used to communicate with any AXMC system.

    Notes:
        This class is explicitly designed to use the same parameters as the Communication class used by the
        microcontroller. Do not modify this class unless you know what you are doing. Modifications to this class will
        likely also require modifying some or all of the core classes that manage microcontroller runtime.

        Due to the use of many non-pickleable classes, this class should be initialized by the process that intends to
        use it. For the canonical purpose of communication with microcontrollers, this means the class should be
        initialized by their remote process that runs the communication loop (see MicroControllerInterface
        implementation for details).

    Args:
        usb_port: The name of the USB port to connect to, e.g.: 'COM3' or '/dev/ttyUSB0'. You can use the
            list_available_ports() class method to get a list of discoverable serial port names.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
            This queue is used to buffer and pipe data to be logged to the logger cores. This class is designed to
            log all received and sent data.
        source_id: The unique integer ID of the source that produces the data. It is expected that the id-code of the
            microcontroller, whose MicroControllerInterface initializes this class, is used as the input ID. This
            explicitly links all data managed by the SerialCCommunication class instance with the connected
            microcontroller.
        baudrate: The baudrate to be used to communicate with the Microcontroller. Should match the value used by
            the microcontroller for UART ports, ignored for USB ports. The appropriate baudrate for many UART-using
            microcontrollers is usually 115200.
        maximum_transmitted_payload_size: The maximum size of the payload that can be transmitted in a single data
            message. This is used to optimize memory usage and prevent overflowing the microcontroller's buffer.
            The default value is 254, which is a typical size for most microcontroller systems.
        verbose: Determines whether to print sent and received data messages to console. This is used during debugging
            and should be disabled during production runtimes. Make sure the console is enabled if this flag is
            enabled, the class itself does NOT enable the console.

    Attributes:
        _transport_layer: A SerialTransportLayer instance that exposes the low-level methods that handle bidirectional
            communication with the microcontroller.
        _module_data: Stores the last received ModuleData message.
        _kernel_data: Stores the last received KernelData message.
        _module_state: Stores the last received ModuleState message.
        _kernel_state: Stores the last received KernelState message.
        _identification: Stores the last received Identification message.
        _reception_code: Stores the last received ReceptionCode message.
        _timestamp_timer: The PrecisionTimer instance used to stamp incoming and outgoing data as it is logged.
        _source_id: Stores the unique integer-code to use as the logged data source.
        _logger_queue: Stores the multiprocessing Queue that buffers and pipes the data to the Logger process(es).
        _object_counter: Stores a monotonically incrementing counter that helps tracks the order of logged data objects.
        _verbose: Stores the verbose flag.
    """

    def __init__(
        self,
        usb_port: str,
        logger_queue: MPQueue,  # type: ignore
        source_id: int,
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
        *,
        verbose: bool = False,
    ) -> None:
        # Initializes the TransportLayer to mostly match a similar specialization carried out by the microcontroller
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

        # Pre-initializes the structures used to parse and store received message data.
        self._module_data = ModuleData(self._transport_layer)
        self._kernel_data = KernelData(self._transport_layer)
        self._module_state = ModuleState(self._transport_layer)
        self._kernel_state = KernelState(self._transport_layer)
        self._identification = Identification(self._transport_layer)
        self._reception_code = ReceptionCode(self._transport_layer)

        # Initializes the trackers used to id-stamp data sent to the logger via the logger_queue. All data sent
        # in this way has to be bundled with a unique source_id, which, for this class, is expected to be the id of the
        # bundled microcontroller. Also, it should be bundled with the monotonically increasing data object counter
        # to preserve the object sequence and timestamp values. Moreover, the data itself should be serialized to a
        # bytes' array.
        self._timestamp_timer = PrecisionTimer("us")
        self._source_id = source_id
        self._logger_queue = logger_queue
        self._object_counter = 0

        # Constructs a timezone-aware stamp using UTC time. This creates a reference point for all later delta time
        # readouts.
        onset = datetime.datetime.now(timezone.utc)
        self._timestamp_timer.reset()  # Immediately resets the timer to make it as close as possible to the onset time

        # Converts the timestamp to numpy datetime64 with microsecond precision and casts it to a bytes' array to make
        # it compatible with the format expected by the logger
        onset_numpy = np.datetime64(onset.isoformat(timespec="microseconds"), "us")
        onset_serial = np.frombuffer(buffer=onset_numpy.tobytes(), dtype=np.uint8)

        # Logs the timestamp as a bytes' array and 0 as object and timestamp values. This pattern is universally
        # interpreted as the 'onset time' log. All further timestamps will be treated as integer time deltas relative
        # to the input time in microseconds.
        self._logger_queue.put((source_id, self._object_counter, 0, onset_serial))
        self._verbose = verbose

    def __repr__(self) -> str:
        """Returns a string representation of the SerialCommunication object."""
        # noinspection PyProtectedMember
        return f"SerialCommunication(usb_port={self._transport_layer._port})"

    @staticmethod
    def list_available_ports() -> tuple[dict[str, int | str], ...]:
        """Provides the information about each serial port addressable through the class (via pySerial library).

        This method is intended to be used for discovering and selecting the serial port names to use with this
        class.

        Returns:
            A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
            port.
        """
        # The method itself is defined in transport_layer module, this wrapper just calls that method.
        return list_available_ports()

    def send_message(
        self,
        message: (
            RepeatedModuleCommand
            | OneOffModuleCommand
            | DequeueModuleCommand
            | KernelCommand
            | KernelParameters
            | ModuleParameters
        ),
    ) -> None:
        """Packages the input command or parameters message and sends it to the connected microcontroller.

        This method transmits any outgoing message to the microcontroller. To do so, it relies on every valid message
        structure exposing a packed_data attribute, that contains the serialized payload data to be sent.
        Overall, this method is a wrapper around the SerialTransportLayer's write_data() and send_data() methods.

        Args:
            message: The command or parameters message to send to the microcontroller.
        """

        # Static guard that prevents sending messages whose data has not been packed. Generally, it should be impossible
        # to encounter this scenario though.
        if message.packed_data is None:
            return

        # Writes the pre-packaged data into the transmission buffer.
        self._transport_layer.write_data(data_object=message.packed_data)

        # Constructs and sends the data message to the connected system.
        self._transport_layer.send_data()
        stamp = self._timestamp_timer.elapsed  # Stamps transmission time.

        self._log_data(stamp, message.packed_data, output=True)

    def receive_message(
        self,
    ) -> ModuleData | ModuleState | KernelData | KernelState | Identification | ReceptionCode | None:
        """Receives the incoming message from the connected microcontroller and parses it into the matching class
        message attribute.

        This method receives all valid incoming message structures. To do so, it uses the protocol code, assumed to be
        stored in the first variable of each payload, to determine how to parse the data.

        Notes:
            To optimize overall runtime speed, this class creates message structures for all supported messages at
            initialization and overwrites the appropriate message attribute with the data extracted from each received
            message payload. This method than returns the reference to the overwritten class attribute. Therefore,
            it is advised to copy or finish working with the structure returned by this method before receiving another
            message. Otherwise, it is possible that the received message will be used to overwrite the data of the
            previously referenced structure, leading to the loss of unprocessed / unsaved data.

        Returns:
            A reference the parsed message structure instance stored in class attributes, or None, if no message was
            received.

        Raises:
            ValueError: If the received message uses an invalid (unrecognized) message protocol code.

        """
        # Attempts to receive the data message. If there is no data to receive, returns None. This is a non-error,
        # no-message return case.
        if not self._transport_layer.receive_data():
            return None
        stamp = self._timestamp_timer.elapsed  # Stamps reception time.

        # If the data was received, first reads the protocol code, expected to be found as the first value of every
        # incoming payload. The protocol is a byte-value, so uses np.uint8 prototype.
        protocol, _ = self._transport_layer.read_data(np.uint8(0), start_index=0)

        # Uses the extracted protocol value to determine the type of the received message and process the received data.
        # All supported message structure classes expose an API method that allows them to process and parse the message
        # payload.
        if protocol == protocols.kModuleData:
            self._module_data.update_message_data()
            self._log_data(stamp, self._module_data.message, output=False)
            return self._module_data

        if protocol == protocols.kKernelData:
            self._kernel_data.update_message_data()
            self._log_data(stamp, self._kernel_data.message, output=False)
            return self._kernel_data

        if protocol == protocols.kModuleState:
            self._module_state.update_message_data()
            self._log_data(stamp, self._module_state.message, output=False)
            return self._module_state

        if protocol == protocols.kKernelState:
            self._kernel_state.update_message_data()
            self._log_data(stamp, self._kernel_state.message, output=False)
            return self._kernel_state

        if protocol == protocols.kReceptionCode:
            self._reception_code.update_message_data()
            self._log_data(stamp, self._reception_code.message, output=False)
            return self._reception_code

        if protocol == protocols.kIdentification:
            self._identification.update_message_data()
            self._log_data(stamp, self._identification.message, output=False)
            return self._identification

        # If the protocol code is not resolved by any conditional above, it is not valid. Terminates runtime with a
        # ValueError
        message = (
            f"Invalid protocol code {protocol} encountered when attempting to parse a message received from the "
            f"microcontroller. All incoming messages have to use one of the valid message protocol code available "
            f"from the SerialProtocols dataclass."
        )
        console.error(message, error=ValueError)
        # Fallback to appease mypy
        raise ValueError(message)  # pragma: no cover

    def _log_data(self, timestamp: int, data: NDArray[np.uint8], *, output: bool = False) -> None:
        """Bundles the input data with ID variables and sends it to be saved to disk by the DataLogger class.

        Args:
            timestamp: The value of the timestamp timer 'elapsed' property for the logged data.
            data: The byte-serialized message payload that was sent or received.
            output: Determines whether the logged data was sent or received. This is only used if the class is
                initialized in verbose mode.
        """
        self._object_counter += 1  # Increments the logged object counter

        # Packages the data to be logged into the appropriate tuple format (with ID variables)
        data_log = (self._source_id, self._object_counter, timestamp, data)

        # Sends the data to the logger
        self._logger_queue.put(data_log)

        if self._verbose:
            if output:
                message = f"Source {self._source_id}, object {self._object_counter}. Sent data: {data}"
            else:
                message = f"Source {self._source_id}, object {self._object_counter}. Received data: {data}"

            console.echo(message=message, level=LogLevel.SUCCESS)


class UnityCommunication:
    """Internally binds an MQTT client and exposes methods for communicating with Unity game engine running one of the
    Ataraxis-compatible tasks.

    This class is intended to be used together with SerialCommunication class to transfer data between microcontrollers
    and Unity game engine using the Gimbl library to manage the Virtual Reality (VR) experimental sessions. Usually,
    both communication classes will be managed by the same process (core) that handles the necessary transformations to
    bridge MQTT and Serial communication protocols sued by this library.

    Notes:
        In the future, this class may be phased out in favor of a unified communication protocol that would use
        Zero-MQ binding instead of MQTT to transmit byte-serialized payloads.

    Args:
        ip: The ip address used by Unity-MQTT binding to create channels.
        port: The port used by Unity-MQTT binding to create MQTT channels.
        monitored_topics: The list of MQTT topics which the class instance should subscribe to and monitor for incoming
            data.

    Attributes:
        _broker: Stores the IP address of the MQTT broker.
        _port: Stores the port used by the broker's TCP socket.
        _connected: Tracks whether the class instance is currently connected to the MQTT broker.
        _subscribe_topics: Stores the names of the topics used by the class instance to subscribe to data from Unity.
        _output_queue: A threading queue used to buffer incoming Unity-sent data inside the local process.
        _client: Stores the initialized mqtt Client class instance that carries out the communication.

    Raises:
        ValueError: If the port or ip arguments are not valid.
    """

    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 1883,
        monitored_topics: None | tuple[str, ...] = None,
    ) -> None:
        # Ensures that ip and port formats are valid:
        if not isinstance(ip, str) or len(ip.split(".")) != 4:
            message = (
                f"Unable to initialize a UnityCommunication class instance. A '.'-delimited string in the format "
                f"'127.0.0.1' is expected as 'ip' argument, but instead encountered {ip} of type "
                f"{type(ip).__name__}."
            )
            console.error(message=message, error=ValueError)
        if port < 0:
            message = (
                f"Unable to initialize a UnityCommunication class instance. A positive integer expected as 'port' "
                f"argument, but instead encountered {port} of type {type(port).__name__}."
            )
            console.error(message=message, error=ValueError)

        self._broker: str = ip
        self._port: int = port
        self._connected = False
        self._subscribe_topics: tuple[str, ...] = monitored_topics if monitored_topics is not None else tuple()

        # Initializes the queue to buffer incoming data. The queue may not be used if the class is not configured to
        # receive any data, but this is a fairly minor inefficiency.
        self._output_queue: Queue = Queue()  # type: ignore

        # Initializes the MQTT client. Note, it needs to be connected before it can send and receive messages!
        self._client: mqtt.Client = mqtt.Client(protocol=mqtt.MQTTv5, transport="tcp")

    def __repr__(self) -> str:
        """Returns a string representation of the UnityCommunication object."""
        return (
            f"UnityCommunication(broker_ip={self._broker}, socket_port={self._port}, connected={self._connected}, "
            f"subscribed_topics={self._subscribe_topics}"
        )

    def __del__(self) -> None:
        """Ensures proper resource release when the class instance is garbage-collected."""
        self.disconnect()

    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:
        """The callback function used to receive data from MQTT.

        When passed to the client, this function will be called each time a new message is received. This function
        will then record the message topic and payload and put them into the output_queue for the data to be consumed
        by external processes.

        Args:
            _client: The MQTT client that received the message. Currently not used.
            _userdata: Custom user-defined data. Currently not used.
            message: The received MQTT message.
        """

        # Whenever a reward message is received, puts True into the reward queue
        self._output_queue.put_nowait((message.topic, message.payload))

        # Add more statements as necessary to extend the class to support other types of incoming data.

    def connect(self) -> None:
        """Connects to the requested MQTT channels and sets up the necessary callback routines.

        This method has to be called to initialize communication. If this method is called, the disconnect() method
        should be called before garbage-collecting the class to ensure proper resource release.

        Notes:
            If this class instance subscribes (listens) to any topics, it will start a perpetually active thread
            with a listener callback to monitor incoming traffic.
        """
        # Guards against re-connecting an already connected client.
        if self._connected:
            return

        # Initializes the client
        self._client.connect(self._broker, self._port)

        # If the class is configured to connect to any topics, enables the connection callback and starts the monitoring
        # thread. Note, this expects the _on_message function to be configured to properly handle all supported
        # subscription topics.
        if len(self._subscribe_topics) != 0:
            # Adds the callback function and starts the monitoring loop.
            self._client.on_message = self._on_message
            self._client.loop_start()

        # Subscribes to necessary topics with qos of 0. Note, this assumes that the communication is happening over
        # a virtual TCP socket and, therefore, does nto need qos.
        for topic in self._subscribe_topics:
            self._client.subscribe(topic=topic, qos=0)

    def send_data(self, topic: str, payload: str | bytes | bytearray | float | None = None) -> None:
        """Publishes the input payload to the specified MQTT topic.

        Use this method to send data over top Unity via the appropriate topic. Note, this method does not verify the
        validity of the input topic or payload data. Ensure both are correct given your specific runtime configuration.

        Args:
            topic: The MQTT topic to publish the data to.
            payload: The data to be published. Keep this set to None to send an empty message, which is the most
                efficient form of binary boolean communication. Otherwise, use Gimbl or Ataraxis serialization
                protocols, depending on your specific use case.
        """
        self._client.publish(topic=topic, payload=payload, qos=0)

    @property
    def has_data(self) -> bool:
        """Returns True if the class stores data received from Unity inside the _output_queue."""
        if not self._output_queue.empty():
            return True
        return False

    def get_data(self) -> tuple[str, bytes | bytearray] | None:
        """Extracts and returns the first buffered message from the reward_queue.

        The received messages are saved as two-element tuples. The first element is a string that communicates the topic
        the message was sent to. The second element is the payload of the message, which is a bytes or bytearray
        object. If no buffered objects are stored in the queue (queue is empty), returns None.
        """
        if not self.has_data:
            return None

        data: tuple[str, bytes | bytearray] = self._output_queue.get_nowait()
        return data

    def disconnect(self) -> None:
        """Disconnects the client from the MQTT broker."""
        # Prevents running the rest of the code if the client was not connected.
        if not self._connected:
            return

        # Stops the listener thread if the client was subscribed to receive topic data.
        if len(self._subscribe_topics) != 0:
            self._client.loop_stop()

        # Disconnects from the client.
        self._client.disconnect()
