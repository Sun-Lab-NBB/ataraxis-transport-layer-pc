"""This module provides the SerialCommunication class and message structures used to bidirectionally communicate with
microcontroller systems running Ataraxis firmware.

The SerialCommunication class builds on top of the SerialTransportLayer class and encapsulates most of the parameters
and functions necessary to communicate with the controller running the default version of the microcontroller
Communication class.
"""

from src.ataraxis_transport_layer.transport_layer import SerialTransportLayer
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional
from ataraxis_base_utilities import console
from ataraxis_data_structures import NestedDictionary


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

    def get_prototype(self, code: np.uint8) -> NDArray[Any] | np.unsignedinteger[Any] | None:
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
        elif code == self.kTwoUnsignedBytes:
            return self._kTwoUnsignedBytesPrototype
        elif code == self.kThreeUnsignedBytes:
            return self._kThreeUnsignedBytesPrototype
        elif code == self.kFourUnsignedBytes:
            return self._kFourUnsignedBytesPrototype
        elif code == self.kOneUnsignedLong:
            return self._kOneUnsignedLongPrototype
        elif code == self.kOneUnsignedShort:
            return self._kOneUnsignedShortPrototype
        else:
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
    packed_data: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kRepeatedModuleCommand)

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
    packed_data: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kOneOffModuleCommand)

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
    packed_data: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kDequeueModuleCommand)

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
    packed_data: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kKernelCommand)

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
    packed_data: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    parameters_size: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kModuleParameters)

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
    packed_data: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    parameters_size: Optional[NDArray[np.uint8]] = field(init=False, default=None)
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kKernelParameters)

    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""

        # Packs the data into the numpy array. Since parameter count and type is known at initialization, this uses a
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


@dataclass()
class ModuleData:
    """Communicates the event state-code of the sender Module and includes an additional data object.

    Data object decoding is expected to be handled by the SerialCommunication class that receives the message.

    Attributes:
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
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    message: NDArray[np.uint8]
    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    event: np.uint8
    data_object: np.unsignedinteger[Any] | NDArray[Any]
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kModuleData)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleData object."""
        message = (
            f"ModuleData(protocol_code={self.protocol_code}, module_type={self.module_type}, "
            f"module_id={self.module_id}, command={self.command}, event={self.event}, "
            f"object_size={self.data_object.nbytes})."
        )
        return message


@dataclass()
class KernelData:
    """Communicates the event state-code of the Kernel and includes an additional data object.

    Data object decoding is expected to be handled by the SerialCommunication class that receives the message.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        command: The unique code of the command that was executed by the Kernel when it sent the data message.
        event: The unique byte-code of the event that prompted sending the data message. The event-code only needs to
            be unique with respect to the executed command code.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes class.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    message: NDArray[np.uint8]
    command: np.uint8
    event: np.uint8
    data_object: np.unsignedinteger[Any] | NDArray[Any]
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kKernelData)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelData object."""
        message = (
            f"KernelData(protocol_code={self.protocol_code}, command={self.command}, event={self.event}, "
            f"object_size={self.data_object.nbytes})."
        )
        return message


@dataclass()
class ModuleState:
    """Communicates the event state-code of the sender Module.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        module_type: The type-code of the module which sent the state message.
        module_id: The specific module ID within the broader module family specified by module_type.
        command: The unique code of the command that was executed by the module that sent the state message.
        event: The unique byte-code of the event that prompted sending the state message. The event-code only needs to
            be unique with respect to the module_type and command combination.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    message: NDArray[np.uint8]
    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    event: np.uint8
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kModuleState)

    def __repr__(self) -> str:
        """Returns a string representation of the ModuleState object."""
        message = (
            f"ModuleState(module_type={self.module_type}, module_id={self.module_id}, command={self.command}, "
            f"event={self.event})."
        )
        return message


@dataclass()
class KernelState:
    """Communicates the event state-code of the Kernel.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        command: The unique code of the command that was executed by the Kernel when it sent the state message.
        event: The unique byte-code of the event that prompted sending the state message. The event-code only needs to
            be unique with respect to the executed command code.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    message: NDArray[np.uint8]
    command: np.uint8
    event: np.uint8
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kKernelState)

    def __repr__(self) -> str:
        """Returns a string representation of the KernelState object."""
        message = f"KernelState(command={self.command}, event={self.event})."
        return message


@dataclass
class ReceptionCode:
    """Identifies the connected microcontroller by communicating its unique byte id-code.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        reception_code: The reception code originally sent as part of the outgoing Command or Parameters messages.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    message: NDArray[np.uint8]
    reception_code: np.uint8
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kIdentification)

    def __repr__(self) -> str:
        """Returns a string representation of the ReceptionCode object."""
        message = f"ReceptionCode(reception_code={self.reception_code})."
        return message


@dataclass()
class Identification:
    """Identifies the connected microcontroller by communicating its unique byte id-code.

    For the ID codes to be unique, they have to be manually assigned to the Kernel class of each concurrently
    used microcontroller.

    Attributes:
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
        controller_id: The unique ID of the microcontroller. This ID is hardcoded in the microcontroller firmware
            and helps track which AXMC firmware is running on the given controller.
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
    """

    message: NDArray[np.uint8]
    controller_id: np.uint8
    protocol_code: np.uint8 = field(init=False, default=SerialProtocols.kIdentification)

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

    Args:
        usb_port: The name of the USB port to connect to, e.g.: 'COM3' or '/dev/ttyUSB0'. You can use the
            list_available_ports() class method to get a list of discoverable serial port names.
        baudrate: The baudrate to be used to communicate with the Microcontroller. Should match the value used by
            the microcontroller for UART ports, ignored for USB ports. The appropriate baudrate for many UART-using
            microcontrollers is usually 115200.
        maximum_transmitted_payload_size: The maximum size of the payload that can be transmitted in a single data
            message. This is used to optimize memory usage and prevent overflowing the microcontroller's buffer.
            The default value is 254, which is a typical size for most microcontroller systems.

    Attributes:
        _transport_layer: A SerialTransportLayer instance that exposes the low-level methods that handle bidirectional
            communication with the microcontroller.
        _module_data: A DataMessage instance used to store incoming data payloads.
        identification_message: An IdentificationMessage instance used to store incoming controlled ID payloads.
        reception_message: An ReceptionMessage instance used to store incoming message reception code payloads.
    """

    def __init__(
        self,
        usb_port: str,
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
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

        # Pre-initializes structures used for processing received data. This optimzies runtime computations by avoiding
        # class
        self._module_data = ModuleData(
            message=np.empty(6, dtype=np.uint8),
            module_type=np.uint8(0),
            module_id=np.uint8(0),
            command=np.uint8(0),
            event=np.uint8(0),
            data_object=np.empty(1, dtype=np.uint8),
        )
        self._kernel_data = KernelData(
            message=np.empty(4, dtype=np.uint8),
            command=np.uint8(0),
            event=np.uint8(0),
            data_object=np.empty(1, dtype=np.uint8),
        )

        self._module_state = ModuleState(
            message=np.empty(5, dtype=np.uint8),
            module_type=np.uint8(0),
            module_id=np.uint8(0),
            command=np.uint8(0),
            event=np.uint8(0),
        )
        self._kernel_state = KernelState(
            message=np.empty(3, dtype=np.uint8),
            command=np.uint8(0),
            event=np.uint8(0),
        )

        self._identification = Identification(
            message=np.empty(2, dtype=np.uint8),
            controller_id=np.uint8(0),
        )
        self._reception = ReceptionCode(
            message=np.empty(2, dtype=np.uint8),
            reception_code=np.uint8(0),
        )

    @staticmethod
    def list_available_ports() -> tuple[dict[str, int | str], ...]:
        """Provides the information about each serial port addressable through the class (via pySerial library).

        This method is intended to be used for discovering and selecting the serial port names to use with this
        class.

        Returns:
            A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
            port.
        """
        # The method itself is defined in TransportLayer class, this wrapper just calls that method
        return SerialTransportLayer.list_available_ports()

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
            # Extracts the message payload, parses the header data and saves it into the DataMessage attribute.
            self._module_data.message, _ = self._transport_layer.read_data(message_prototype, start_index=0)
            self._module_data.parse_payload()  # Parses header data from message bytes
            return self._module_data
        elif protocol == SerialProtocols.RECEPTION.value:
            self.reception_message, _ = self._transport_layer.read_data(self._service_message_prototype, start_index=0)
            self.reception_message.parse_payload()  # Parses reception code value from the structure
            return self.reception_message
        elif protocol == SerialProtocols.IDENTIFICATION.value:
            self.identification_message, _ = self._transport_layer.read_data(
                self._service_message_prototype, start_index=0
            )
            self.identification_message.parse_payload()  # Parses controlled ID code value from the structure
            return self.identification_message
        else:
            message = (
                f"Unable to recognize the protocol code {protocol} of the received message.See the Protocols "
                f"enumeration for currently supported incoming message codes."
            )
            console.error(message, error=ValueError)

    def extract_data_object(
        self,
        prototype_object: np.unsignedinteger[Any] | np.signedinteger[Any] | np.floating[Any] | np.bool | NDArray[Any],
    ) -> np.unsignedinteger[Any] | np.signedinteger[Any] | np.floating[Any] | np.bool | NDArray[Any]:
        """Reconstructs the data object from the serialized data bytes using the provided prototype.

        This step completed data message reception by processing the additional message data. This has to be carried
        out separately, as data object structure is not known until the exact object prototype is determined based on
        the module-command-event ID information extracted from the message.

        Args:
            prototype_object: The prototype object that will be used to format the extracted data. The appropriate
                prototype for data extraction depends on the data format used by the sender module and has to be
                determined individually for each received data message. Currently, only numpy scalar or array prototypes
                are supported.

        Raises:
            ValueError: If the size of the provided prototype (in bytes) does not match the object size declared in the
            data message.
        """

        # If the provided prototype's byte-size does not match the object size declared in the received message,
        # raises an error.
        if self._module_data.object_size != prototype_object.nbytes:
            message = (
                "Unable to extract the requested data object from the received message payload. The size of the object "
                f"declared by the incoming data message {self._module_data.object_size} does not match the size of the "
                f"provided prototype {prototype_object.size} (in bytes). This may indicate that the data was "
                f"corrupted in transmission."
            )
            console.error(message, error=ValueError)

        # Uses the prototype to extract object data from the storage buffer, reconstruct the object and return it to
        # caller.
        data_object, _ = self._transport_layer.read_data(prototype_object, start_index=self.data_object_index)
        return data_object
