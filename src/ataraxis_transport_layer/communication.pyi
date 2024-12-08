<<<<<<< HEAD
<<<<<<< HEAD
from typing import Any
=======
from enum import IntEnum
from typing import Any, Callable
>>>>>>> 5b8062e (Added Communication module tests)
=======
from enum import IntEnum
from typing import Any, Callable
>>>>>>> origin/main
from dataclasses import dataclass
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from numpy.typing import NDArray
import paho.mqtt.client as mqtt
<<<<<<< HEAD
<<<<<<< HEAD
from ataraxis_data_structures import NestedDictionary
=======
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main

from .transport_layer import (
    SerialTransportLayer as SerialTransportLayer,
    list_available_ports as list_available_ports,
)

<<<<<<< HEAD
<<<<<<< HEAD
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

    kUndefined: np.uint8 = ...
    kRepeatedModuleCommand: np.uint8 = ...
    kOneOffModuleCommand: np.uint8 = ...
    kDequeueModuleCommand: np.uint8 = ...
    kKernelCommand: np.uint8 = ...
    kModuleParameters: np.uint8 = ...
    kKernelParameters: np.uint8 = ...
    kModuleData: np.uint8 = ...
    kKernelData: np.uint8 = ...
    kModuleState: np.uint8 = ...
    kKernelState: np.uint8 = ...
    kReceptionCode: np.uint8 = ...
    kIdentification: np.uint8 = ...
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

    kOneUnsignedByte: np.uint8 = ...
    _kOneUnsignedBytePrototype: np.uint8 = ...
    kTwoUnsignedBytes: np.uint8 = ...
    _kTwoUnsignedBytesPrototype: NDArray[Any] = ...
    kThreeUnsignedBytes: np.uint8 = ...
    _kThreeUnsignedBytesPrototype: NDArray[Any] = ...
    kFourUnsignedBytes: np.uint8 = ...
    _kFourUnsignedBytesPrototype: NDArray[Any] = ...
    kOneUnsignedLong: np.uint8 = ...
    _kOneUnsignedLongPrototype: np.uint32 = ...
    kOneUnsignedShort: np.uint8 = ...
    _kOneUnsignedShortPrototype: np.uint16 = ...
    def get_prototype(self, code: np.uint8) -> NDArray[np.uint8] | np.uint8 | np.uint16 | np.uint32 | None:
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

prototypes: Incomplete
protocols: Incomplete

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
=======
class SerialProtocols(IntEnum):
    """Stores the protocol codes used in data transmission between the PC and the microcontroller over the serial port.

    Each sent and received message starts with the specific protocol code from this enumeration that instructs the
    receiver on how to process the rest of the data payload. The codes available through this class have to match the
    contents of the kProtocols Enumeration available from the AtaraxisMicroController library
    (communication_assets namespace).

    Notes:
        The values available through this enumeration should be read through their 'as_uint8' property to enforce the
        type expected by other classes from ths library.
>>>>>>> 5b8062e (Added Communication module tests)
    """

    UNDEFINED: int
    REPEATED_MODULE_COMMAND: int
    ONE_OFF_MODULE_COMMAND: int
    DEQUEUE_MODULE_COMMAND: int
    KERNEL_COMMAND: int
    MODULE_PARAMETERS: int
    KERNEL_PARAMETERS: int
    MODULE_DATA: int
    KERNEL_DATA: int
    MODULE_STATE: int
    KERNEL_STATE: int
    RECEPTION_CODE: int
    IDENTIFICATION: int
    def as_uint8(self) -> np.uint8:
        """Convert the enum value to numpy.uint8 type.

        Returns:
            np.uint8: The enum value as a numpy unsigned 8-bit integer.
        """

_PROTOTYPE_FACTORIES: dict[int, Callable[[], NDArray[np.uint8] | np.uint8 | np.uint16 | np.uint32]]

class SerialPrototypes(IntEnum):
    """Stores the prototype codes used in data transmission between the PC and the microcontroller over the serial port.

    Prototype codes are used by Data messages (Kernel and Module) to inform the receiver about the structure (prototype)
    that can be used to deserialize the included data object. Transmitting these codes with the message ensures that
    the receiver has the necessary information to decode the data without doing any additional processing. In turn,
    this allows optimizing the reception procedure to efficiently decode the data objects.

    Notes:
        While the use of 8-bit (byte) value limits the number of mapped prototypes to 255 (256 if 0 is made a valid
        value), this number should be enough to support many unique runtime configurations.
    """

    ONE_UNSIGNED_BYTE: int
    TWO_UNSIGNED_BYTES: int
    THREE_UNSIGNED_BYTES: int
    FOUR_UNSIGNED_BYTES: int
    ONE_UNSIGNED_LONG: int
    ONE_UNSIGNED_SHORT: int
    def as_uint8(self) -> np.uint8:
        """Converts the enum value to numpy.uint8 type.

        Returns:
            The enum value as a numpy unsigned 8-bit integer.
        """
    def get_prototype(self) -> NDArray[np.uint8] | np.uint8 | np.uint16 | np.uint32:
        """Returns the prototype object associated with this prototype enum value.

        The prototype object returned by this method can be passed to the reading method of the SerialTransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this enum class.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type.
        """
    @classmethod
    def get_prototype_for_code(cls, code: np.uint8) -> NDArray[np.uint8] | np.uint8 | np.uint16 | np.uint32 | None:
        """Returns the prototype object associated with the input prototype code.

        The prototype object returned by this method can be passed to the reading method of the SerialTransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this enum.

        Args:
            code: The prototype byte-code to retrieve the prototype for.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type. If the input code is not one of
            the supported codes, returns None to indicate a matching error.
        """

@dataclass(frozen=True)
class RepeatedModuleCommand:
    """Instructs the addressed Module to repeatedly (recurrently) run the specified command."""

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
    cycle_delay: np.uint32 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""

@dataclass(frozen=True)
class OneOffModuleCommand:
<<<<<<< HEAD
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
=======
class SerialProtocols(IntEnum):
    """Stores the protocol codes used in data transmission between the PC and the microcontroller over the serial port.

    Each sent and received message starts with the specific protocol code from this enumeration that instructs the
    receiver on how to process the rest of the data payload. The codes available through this class have to match the
    contents of the kProtocols Enumeration available from the AtaraxisMicroController library
    (communication_assets namespace).

    Notes:
        The values available through this enumeration should be read through their 'as_uint8' property to enforce the
        type expected by other classes from ths library.
>>>>>>> origin/main
    """

    UNDEFINED: int
    REPEATED_MODULE_COMMAND: int
    ONE_OFF_MODULE_COMMAND: int
    DEQUEUE_MODULE_COMMAND: int
    KERNEL_COMMAND: int
    MODULE_PARAMETERS: int
    KERNEL_PARAMETERS: int
    MODULE_DATA: int
    KERNEL_DATA: int
    MODULE_STATE: int
    KERNEL_STATE: int
    RECEPTION_CODE: int
    IDENTIFICATION: int
    def as_uint8(self) -> np.uint8:
        """Convert the enum value to numpy.uint8 type.

        Returns:
            np.uint8: The enum value as a numpy unsigned 8-bit integer.
        """

_PROTOTYPE_FACTORIES: dict[int, Callable[[], NDArray[np.uint8] | np.uint8 | np.uint16 | np.uint32]]

class SerialPrototypes(IntEnum):
    """Stores the prototype codes used in data transmission between the PC and the microcontroller over the serial port.

    Prototype codes are used by Data messages (Kernel and Module) to inform the receiver about the structure (prototype)
    that can be used to deserialize the included data object. Transmitting these codes with the message ensures that
    the receiver has the necessary information to decode the data without doing any additional processing. In turn,
    this allows optimizing the reception procedure to efficiently decode the data objects.

    Notes:
        While the use of 8-bit (byte) value limits the number of mapped prototypes to 255 (256 if 0 is made a valid
        value), this number should be enough to support many unique runtime configurations.
    """

    ONE_UNSIGNED_BYTE: int
    TWO_UNSIGNED_BYTES: int
    THREE_UNSIGNED_BYTES: int
    FOUR_UNSIGNED_BYTES: int
    ONE_UNSIGNED_LONG: int
    ONE_UNSIGNED_SHORT: int
    def as_uint8(self) -> np.uint8:
        """Converts the enum value to numpy.uint8 type.

        Returns:
            The enum value as a numpy unsigned 8-bit integer.
        """
    def get_prototype(self) -> NDArray[np.uint8] | np.uint8 | np.uint16 | np.uint32:
        """Returns the prototype object associated with this prototype enum value.

        The prototype object returned by this method can be passed to the reading method of the SerialTransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this enum class.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type.
        """
    @classmethod
    def get_prototype_for_code(cls, code: np.uint8) -> NDArray[np.uint8] | np.uint8 | np.uint16 | np.uint32 | None:
        """Returns the prototype object associated with the input prototype code.

        The prototype object returned by this method can be passed to the reading method of the SerialTransportLayer
        class to deserialize the received data object. This should be automatically done by the SerialCommunication
        class that uses this enum.

        Args:
            code: The prototype byte-code to retrieve the prototype for.

        Returns:
            The prototype object that is either a numpy scalar or shallow array type. If the input code is not one of
            the supported codes, returns None to indicate a matching error.
        """

@dataclass(frozen=True)
class RepeatedModuleCommand:
    """Instructs the addressed Module to repeatedly (recurrently) run the specified command."""

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
<<<<<<< HEAD
=======
    cycle_delay: np.uint32 = ...
>>>>>>> origin/main
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
<<<<<<< HEAD
        """Returns a string representation of the OneOffModuleCommand object."""

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
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""

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
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelCommand object."""

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
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleParameters object."""

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
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelParameters object."""

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

=======
    """Instructs the addressed Module to run the specified command exactly once (non-recurrently)."""

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the OneOffModuleCommand object."""

@dataclass(frozen=True)
class DequeueModuleCommand:
    """Instructs the addressed Module to clear (empty) its command queue.

    Note, clearing the command queue does not terminate already executing commands, but it prevents recurrent commands
    from running again."""

    module_type: np.uint8
    module_id: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""

@dataclass(frozen=True)
class KernelCommand:
    """Instructs the Kernel to run the specified command exactly once.

    Currently, the Kernel only supports blocking one-off commands."""

    command: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelCommand object."""

@dataclass(frozen=True)
class ModuleParameters:
    """Instructs the addressed Module to overwrite its custom parameters object with the included object data."""

    module_type: np.uint8
    module_id: np.uint8
    parameter_data: tuple[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool, ...]
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleParameters object."""

@dataclass(frozen=True)
class KernelParameters:
    """Instructs the Kernel to update the microcontroller-wide parameters with the values included in the message.

    These parameters are shared by the Kernel and all custom Modules, and the exact parameter layout is hardcoded. This
    is in contrast to Module parameters, that differ between module types."""

    action_lock: np.bool
    ttl_lock: np.bool
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelParameters object."""

class ModuleData:
    """Communicates the event state-code of the sender Module and includes an additional data object.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        module_type: The type (family) code of the module that sent the message.
        module_id: The ID of the specific module within the broader module-family.
        command: The code of the command the module was executing when it sent the message.
        event: The code of the event that prompted sending the message.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes enumeration.
        _transport_layer: Stores the reference to the TransportLayer class.
    """

>>>>>>> 5b8062e (Added Communication module tests)
=======
        """Returns a string representation of the RepeatedModuleCommand object."""

@dataclass(frozen=True)
class OneOffModuleCommand:
    """Instructs the addressed Module to run the specified command exactly once (non-recurrently)."""

    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    return_code: np.uint8 = ...
    noblock: np.bool_ = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the OneOffModuleCommand object."""

@dataclass(frozen=True)
class DequeueModuleCommand:
    """Instructs the addressed Module to clear (empty) its command queue.

    Note, clearing the command queue does not terminate already executing commands, but it prevents recurrent commands
    from running again."""

    module_type: np.uint8
    module_id: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the RepeatedModuleCommand object."""

@dataclass(frozen=True)
class KernelCommand:
    """Instructs the Kernel to run the specified command exactly once.

    Currently, the Kernel only supports blocking one-off commands."""

    command: np.uint8
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelCommand object."""

@dataclass(frozen=True)
class ModuleParameters:
    """Instructs the addressed Module to overwrite its custom parameters object with the included object data."""

    module_type: np.uint8
    module_id: np.uint8
    parameter_data: tuple[np.signedinteger[Any] | np.unsignedinteger[Any] | np.floating[Any] | np.bool, ...]
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleParameters object."""

@dataclass(frozen=True)
class KernelParameters:
    """Instructs the Kernel to update the microcontroller-wide parameters with the values included in the message.

    These parameters are shared by the Kernel and all custom Modules, and the exact parameter layout is hardcoded. This
    is in contrast to Module parameters, that differ between module types."""

    action_lock: np.bool
    ttl_lock: np.bool
    return_code: np.uint8 = ...
    packed_data: NDArray[np.uint8] | None = ...
    parameters_size: NDArray[np.uint8] | None = ...
    protocol_code: np.uint8 = ...
    def __post_init__(self) -> None:
        """Packs the data into the numpy array to optimize future transmission speed."""
    def __repr__(self) -> str:
        """Returns a string representation of the KernelParameters object."""

class ModuleData:
    """Communicates the event state-code of the sender Module and includes an additional data object.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        module_type: The type (family) code of the module that sent the message.
        module_id: The ID of the specific module within the broader module-family.
        command: The code of the command the module was executing when it sent the message.
        event: The code of the event that prompted sending the message.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes enumeration.
        _transport_layer: Stores the reference to the TransportLayer class.
    """

>>>>>>> origin/main
    protocol_code: Incomplete
    message: Incomplete
    module_type: Incomplete
    module_id: Incomplete
    command: Incomplete
    event: Incomplete
    data_object: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: SerialTransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.
<<<<<<< HEAD
<<<<<<< HEAD

=======
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
        """
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleData object."""

class KernelData:
    """Communicates the event state-code of the Kernel and includes an additional data object.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> origin/main
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        command: The code of the command the Kernel was executing when it sent the message.
        event: The code of the event that prompted sending the message.
        data_object: The data object decoded from the received message. Note, data messages only support the objects
            whose prototypes are defined in the SerialPrototypes enumeration.
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
        _transport_layer: Stores the reference to the TransportLayer class.
    """

    protocol_code: Incomplete
    message: Incomplete
    command: Incomplete
    event: Incomplete
    data_object: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: SerialTransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        Raises:
            ValueError: If the prototype code transmitted with the message is not valid.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the KernelData object."""

class ModuleState:
    """Communicates the event state-code of the sender Module.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> origin/main
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        module_type: The type (family) code of the module that sent the message.
        module_id: The ID of the specific module within the broader module-family.
        command: The code of the command the module was executing when it sent the message.
        event: The code of the event that prompted sending the message.
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
    """

    protocol_code: Incomplete
    message: Incomplete
    module_type: Incomplete
    module_id: Incomplete
    command: Incomplete
    event: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: SerialTransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever ModuleData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.

        """
    def __repr__(self) -> str:
        """Returns a string representation of the ModuleState object."""

class KernelState:
    """Communicates the event state-code of the Kernel.

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
<<<<<<< HEAD
<<<<<<< HEAD
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
=======
=======
>>>>>>> origin/main
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
        command: The code of the command the Kernel was executing when it sent the message.
        event: The code of the event that prompted sending the message.
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
    """

    protocol_code: Incomplete
    message: Incomplete
    command: Incomplete
    event: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: SerialTransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the KernelState object."""

class ReceptionCode:
<<<<<<< HEAD
<<<<<<< HEAD
    """Identifies the connected microcontroller by communicating its unique byte id-code.
=======
    """Returns the reception_code originally received from the PC to indicate that the message with that code was
    received and parsed.
>>>>>>> 5b8062e (Added Communication module tests)
=======
    """Returns the reception_code originally received from the PC to indicate that the message with that code was
    received and parsed.
>>>>>>> origin/main

    This class initializes to nonsensical defaults and expects the SerialCommunication class that manages its lifetime
    to call update_message_data() method when necessary to parse valid incoming message data.

    Args:
        transport_layer: The reference to the TransportLayer class that is initialized and managed by the
            SerialCommunication class. This reference is used to read and parse the message data.

    Attributes:
<<<<<<< HEAD
<<<<<<< HEAD
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
=======
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
>>>>>>> 5b8062e (Added Communication module tests)
=======
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
>>>>>>> origin/main
        reception_code: The reception code originally sent as part of the outgoing Command or Parameters messages.
    """

    protocol_code: Incomplete
    message: Incomplete
    reception_code: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: SerialTransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the ReceptionCode object."""

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
<<<<<<< HEAD
<<<<<<< HEAD
        protocol_code: The unique code of the communication protocol used by this message structure. This is resolved
            automatically during class initialization.
        message: The original serialized message payload, from which the rest of the structure data was decoded.
            This is used to optimize data logging by saving the serialized data during active runtime and decoding it
            into a unified log format offline.
=======
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
>>>>>>> 5b8062e (Added Communication module tests)
=======
        protocol_code: Stores the protocol code used by this type of messages.
        message: Stores the serialized message payload.
>>>>>>> origin/main
        controller_id: The unique ID of the microcontroller. This ID is hardcoded in the microcontroller firmware
            and helps track which AXMC firmware is running on the given controller.

    """

    protocol_code: Incomplete
    message: Incomplete
    controller_id: Incomplete
    _transport_layer: Incomplete
    def __init__(self, transport_layer: SerialTransportLayer) -> None: ...
    def update_message_data(self) -> None:
        """Reads and parses the data stored in the reception buffer of the TransportLayer class, overwriting class
        attributes.

        This method should be called by the SerialCommunication class whenever KernelData message is received and needs
        to be parsed (as indicated by the incoming message protocol). This method will then access the reception buffer
        and attempt to parse the data.
        """
    def __repr__(self) -> str:
        """Returns a string representation of the Identification object."""

class SerialCommunication:
<<<<<<< HEAD
<<<<<<< HEAD
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
        _verbose: Stores the verbose flag.
=======
    """Wraps a SerialTransportLayer class instance and exposes methods that allow communicating with a microcontroller
    running AtaraxisMicroController firmware using the USB or UART protocol.

    This class is built on top of the SerialTransportLayer, designed to provide the microcontroller communication
    interface (API) for other Ataraxis libraries.

    Notes:
        This class is explicitly designed to use the same parameters as the Communication class used by the
        microcontroller. Do not modify this class unless you know what you are doing.

        Due to the use of many non-pickleable classes, this class cannot be piped to a remote process and has to be
        initialized by the remote process directly.

        This class is designed to integrate with DataLogger class available from the ataraxis_data_structures library.
        The DataLogger is used to write all incoming and outgoing messages to disk as serialized message payloads.

    Args:
        usb_port: The name of the USB port to use for communication to, e.g.: 'COM3' or '/dev/ttyUSB0'. This has to be
            the port to which the target microcontroller is connected. Use the list_available_ports() function available
            from this library to get the list of discoverable serial port names.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
            This queue is used to buffer and pipe data to be logged to the logger cores.
        source_id: The ID code to identify the source of the logged messages. This is used by the DataLogger to
            distinguish between log sources (classes that sent data to be logged) and, therefore, has to be unique for
            all Ataraxis classes that use DataLogger and are active at the same time.
        baudrate: The baudrate to use for the communication over the UART protocol. Should match the value used by
            the microcontrollers that only support UART protocol. This is ignored for microcontrollers that use the
            USB protocol.
        maximum_transmitted_payload_size: The maximum size of the payload that will be sent to the microcontroller in
            a single message. in bytes. This value has to match the expected maximum payload size of the target
            microcontroller.
        verbose: Determines whether to print sent and received data messages to console. This is used during debugging
            and should be disabled during production runtimes. The class itself does NOT enable the console, so the
            console has to be enabled manually for this flag to have an effect.
        test_mode: This parameter is only used during testing. When True, it initializes the underlying TransportLayer
            class in the test configuration. Make sure this is set to False during production runtime.

    Attributes:
=======
    """Wraps a SerialTransportLayer class instance and exposes methods that allow communicating with a microcontroller
    running AtaraxisMicroController firmware using the USB or UART protocol.

    This class is built on top of the SerialTransportLayer, designed to provide the microcontroller communication
    interface (API) for other Ataraxis libraries.

    Notes:
        This class is explicitly designed to use the same parameters as the Communication class used by the
        microcontroller. Do not modify this class unless you know what you are doing.

        Due to the use of many non-pickleable classes, this class cannot be piped to a remote process and has to be
        initialized by the remote process directly.

        This class is designed to integrate with DataLogger class available from the ataraxis_data_structures library.
        The DataLogger is used to write all incoming and outgoing messages to disk as serialized message payloads.

    Args:
        usb_port: The name of the USB port to use for communication to, e.g.: 'COM3' or '/dev/ttyUSB0'. This has to be
            the port to which the target microcontroller is connected. Use the list_available_ports() function available
            from this library to get the list of discoverable serial port names.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
            This queue is used to buffer and pipe data to be logged to the logger cores.
        source_id: The ID code to identify the source of the logged messages. This is used by the DataLogger to
            distinguish between log sources (classes that sent data to be logged) and, therefore, has to be unique for
            all Ataraxis classes that use DataLogger and are active at the same time.
        baudrate: The baudrate to use for the communication over the UART protocol. Should match the value used by
            the microcontrollers that only support UART protocol. This is ignored for microcontrollers that use the
            USB protocol.
        maximum_transmitted_payload_size: The maximum size of the payload that will be sent to the microcontroller in
            a single message. in bytes. This value has to match the expected maximum payload size of the target
            microcontroller.
        verbose: Determines whether to print sent and received data messages to console. This is used during debugging
            and should be disabled during production runtimes. The class itself does NOT enable the console, so the
            console has to be enabled manually for this flag to have an effect.
        test_mode: This parameter is only used during testing. When True, it initializes the underlying TransportLayer
            class in the test configuration. Make sure this is set to False during production runtime.

    Attributes:
>>>>>>> origin/main
        _transport_layer: The TransportLayer instance that handles the communication.
        _module_data: Received ModuleData messages are unpacked into this structure.
        _kernel_data: Received KernelData messages are unpacked into this structure.
        _module_state: Received ModuleState messages are unpacked into this structure.
        _kernel_state: Received KernelState messages are unpacked into this structure.
        _identification: Received Identification messages are unpacked into this structure.
        _reception_code: Received ReceptionCode messages are unpacked into this structure.
        _timestamp_timer: The PrecisionTimer instance used to stamp incoming and outgoing data as it is logged.
        _source_id: Stores the unique integer-code that identifies the class instance in data logs.
        _logger_queue: Stores the multiprocessing Queue that buffers and pipes the data to the Logger process(es).
        _verbose: Stores the verbose flag.
        _usb_port: Stores the ID of the USB port used for communication.
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
    """

    _transport_layer: Incomplete
    _module_data: Incomplete
    _kernel_data: Incomplete
    _module_state: Incomplete
    _kernel_state: Incomplete
    _identification: Incomplete
    _reception_code: Incomplete
    _timestamp_timer: Incomplete
    _source_id: Incomplete
    _logger_queue: Incomplete
    _verbose: Incomplete
<<<<<<< HEAD
<<<<<<< HEAD
=======
    _usb_port: Incomplete
>>>>>>> 5b8062e (Added Communication module tests)
=======
    _usb_port: Incomplete
>>>>>>> origin/main
    def __init__(
        self,
        usb_port: str,
        logger_queue: MPQueue,
<<<<<<< HEAD
<<<<<<< HEAD
        source_id: int,
=======
        source_id: np.uint8,
>>>>>>> 5b8062e (Added Communication module tests)
=======
        source_id: np.uint8,
>>>>>>> origin/main
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
        *,
        verbose: bool = False,
<<<<<<< HEAD
<<<<<<< HEAD
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the SerialCommunication object."""
    @staticmethod
    def list_available_ports() -> tuple[dict[str, int | str], ...]:
        """Provides the information about each serial port addressable through the class (via pySerial library).

        This method is intended to be used for discovering and selecting the serial port names to use with this
        class.

        Returns:
            A tuple of dictionaries with each dictionary storing ID and descriptive information about each discovered
            port.
        """
=======
=======
>>>>>>> origin/main
        test_mode: bool = False,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the SerialCommunication object."""
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
    def send_message(
        self,
        message: RepeatedModuleCommand
        | OneOffModuleCommand
        | DequeueModuleCommand
        | KernelCommand
        | KernelParameters
        | ModuleParameters,
    ) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
        """Packages the input command or parameters message and sends it to the connected microcontroller.

        This method transmits any outgoing message to the microcontroller. To do so, it relies on every valid message
        structure exposing a packed_data attribute, that contains the serialized payload data to be sent.
        Overall, this method is a wrapper around the SerialTransportLayer's write_data() and send_data() methods.
=======
        """Serializes the input command or parameters message and sends it to the connected microcontroller.

        This method relies on every valid outgoing message structure exposing a packed_data attribute, that contains
        the serialized payload data to be sent. Functionally, this method is a wrapper around the
        SerialTransportLayer's write_data() and send_data() methods.
>>>>>>> 5b8062e (Added Communication module tests)
=======
        """Serializes the input command or parameters message and sends it to the connected microcontroller.

        This method relies on every valid outgoing message structure exposing a packed_data attribute, that contains
        the serialized payload data to be sent. Functionally, this method is a wrapper around the
        SerialTransportLayer's write_data() and send_data() methods.
>>>>>>> origin/main

        Args:
            message: The command or parameters message to send to the microcontroller.
        """
    def receive_message(
        self,
    ) -> ModuleData | ModuleState | KernelData | KernelState | Identification | ReceptionCode | None:
<<<<<<< HEAD
<<<<<<< HEAD
        """Receives the incoming message from the connected microcontroller and parses it into the matching class
        message attribute.

        This method receives all valid incoming message structures. To do so, it uses the protocol code, assumed to be
        stored in the first variable of each payload, to determine how to parse the data.
=======
        """Receives the incoming message from the connected microcontroller and parses into the appropriate structure.

        This method uses the protocol code, assumed to be stored in the first variable of each received payload, to
        determine how to parse the data. It then parses into a precreated message structure stored in class attributes.
>>>>>>> 5b8062e (Added Communication module tests)
=======
        """Receives the incoming message from the connected microcontroller and parses into the appropriate structure.

        This method uses the protocol code, assumed to be stored in the first variable of each received payload, to
        determine how to parse the data. It then parses into a precreated message structure stored in class attributes.
>>>>>>> origin/main

        Notes:
            To optimize overall runtime speed, this class creates message structures for all supported messages at
            initialization and overwrites the appropriate message attribute with the data extracted from each received
            message payload. This method than returns the reference to the overwritten class attribute. Therefore,
            it is advised to copy or finish working with the structure returned by this method before receiving another
            message. Otherwise, it is possible that the received message will be used to overwrite the data of the
            previously referenced structure, leading to the loss of unprocessed / unsaved data.

        Returns:
            A reference the parsed message structure instance stored in class attributes, or None, if no message was
<<<<<<< HEAD
<<<<<<< HEAD
            received.
=======
            received. Note, None return does not indicate an error, but rather indicates that the microcontroller did
            not send any data.
>>>>>>> 5b8062e (Added Communication module tests)
=======
            received. Note, None return does not indicate an error, but rather indicates that the microcontroller did
            not send any data.
>>>>>>> origin/main

        Raises:
            ValueError: If the received message uses an invalid (unrecognized) message protocol code.

        """
    def _log_data(self, timestamp: int, data: NDArray[np.uint8], *, output: bool = False) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
        """Bundles the input data with ID variables and sends it to be saved to disk by the DataLogger class.

        Args:
            timestamp: The value of the timestamp timer 'elapsed' property for the logged data.
            data: The byte-serialized message payload that was sent or received.
            output: Determines whether the logged data was sent or received. This is only used if the class is
                initialized in verbose mode.
        """

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

    _broker: Incomplete
    _port: Incomplete
    _connected: bool
    _subscribe_topics: Incomplete
=======
        """Packages and sends the input data to teh DataLogger instance that writes it to disk.

        Args:
            timestamp: The value of the timestamp timer 'elapsed' property that communicates the number of elapsed
                microseconds relative to the 'onset' timestamp.
            data: The byte-serialized message payload that was sent or received.
            output: Determines whether the logged data was sent or received. This is only used if the class is
                initialized in verbose mode to format messages displayed via the terminal.
        """

class UnityCommunication:
=======
        """Packages and sends the input data to teh DataLogger instance that writes it to disk.

        Args:
            timestamp: The value of the timestamp timer 'elapsed' property that communicates the number of elapsed
                microseconds relative to the 'onset' timestamp.
            data: The byte-serialized message payload that was sent or received.
            output: Determines whether the logged data was sent or received. This is only used if the class is
                initialized in verbose mode to format messages displayed via the terminal.
        """

class UnityCommunication:
>>>>>>> origin/main
    """Wraps an MQTT client and exposes methods for communicating with Unity game engine running one of the
    Ataraxis-compatible tasks.

    This class leverages MQTT protocol on Python side and the Gimbl library (that also wraps an MQTT client) on the
    Unity side to establish bidirectional communication between Python and Virtual Reality (VR) game world. Primarily,
    the class is intended to be used together with SerialCommunication class to transfer data between microcontrollers
    and Unity. Usually, both communication classes will be managed by the same process (core) that handles the necessary
    transformations to bridge MQTT and Serial communication protocols used by this library.

    Notes:
        In the future, this class may be phased out in favor of a unified communication protocol that would use
        Zero-MQ binding instead of MQTT to transmit byte-serialized payloads. For now, it instead reuses Gimbl
        architecture with slight optimizations as a temporary, proof-of-concept solution. Regardless of underlying
        protocol, the class functionality and API are likely to remain stable.

        MQTT protocol requires a broker that facilitates communication, which this class does NOT provide. Make sure
        your infrastructure includes a working MQTT broker before using this class. See https://mqtt.org/ for more
        details.

    Args:
        ip: The IP address of the MQTT broker that facilitates the communication.
        port: The socket port used by the MQTT broker that facilitates the communication.
        monitored_topics: The list of MQTT topics which the class instance should subscribe to and monitor for incoming
            messages.

    Attributes:
        _ip: Stores the IP address of the MQTT broker.
        _port: Stores the port used by the broker's TCP socket.
        _connected: Tracks whether the class instance is currently connected to the MQTT broker.
        _monitored_topics: Stores the topics the class should monitor for incoming messages sent from Unity.
        _output_queue: A multithreading queue used to buffer incoming messages received from Unity before their data is
            requested via class methods.
        _client: Stores the initialized mqtt client instance that carries out the communication.

    Raises:
        RuntimeError: If the MQTT broker cannot be connected to using the provided IP and Port.
    """

    _ip: Incomplete
    _port: Incomplete
    _connected: bool
    _monitored_topics: Incomplete
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
    _output_queue: Incomplete
    _client: Incomplete
    def __init__(
        self, ip: str = "127.0.0.1", port: int = 1883, monitored_topics: None | tuple[str, ...] = None
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the UnityCommunication object."""
    def __del__(self) -> None:
        """Ensures proper resource release when the class instance is garbage-collected."""
    def _on_message(self, _client: mqtt.Client, _userdata: Any, message: mqtt.MQTTMessage) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
        """The callback function used to receive data from MQTT.
=======
        """The callback function used to receive data from MQTT broker.
>>>>>>> 5b8062e (Added Communication module tests)
=======
        """The callback function used to receive data from MQTT broker.
>>>>>>> origin/main

        When passed to the client, this function will be called each time a new message is received. This function
        will then record the message topic and payload and put them into the output_queue for the data to be consumed
        by external processes.

        Args:
            _client: The MQTT client that received the message. Currently not used.
            _userdata: Custom user-defined data. Currently not used.
            message: The received MQTT message.
        """
    def connect(self) -> None:
<<<<<<< HEAD
<<<<<<< HEAD
        """Connects to the requested MQTT channels and sets up the necessary callback routines.

        This method has to be called to initialize communication. If this method is called, the disconnect() method
        should be called before garbage-collecting the class to ensure proper resource release.
=======
=======
>>>>>>> origin/main
        """Connects to the MQTT broker and subscribes to the requested input topics.

        This method has to be called to initialize communication, both for incoming and outgoing messages. Any message
        sent to the MQTT broker from unity before this method is called may not reach this class.
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main

        Notes:
            If this class instance subscribes (listens) to any topics, it will start a perpetually active thread
            with a listener callback to monitor incoming traffic.
        """
    def send_data(self, topic: str, payload: str | bytes | bytearray | float | None = None) -> None:
        """Publishes the input payload to the specified MQTT topic.

<<<<<<< HEAD
<<<<<<< HEAD
        Use this method to send data over top Unity via the appropriate topic. Note, this method does not verify the
        validity of the input topic or payload data. Ensure both are correct given your specific runtime configuration.

        Args:
            topic: The MQTT topic to publish the data to.
            payload: The data to be published. Keep this set to None to send an empty message, which is the most
                efficient form of binary boolean communication. Otherwise, use Gimbl or Ataraxis serialization
                protocols, depending on your specific use case.
        """
    @property
    def has_data(self) -> bool:
        """Returns True if the class stores data received from Unity inside the _output_queue."""
    def get_data(self) -> tuple[str, bytes | bytearray] | None:
        """Extracts and returns the first buffered message from the reward_queue.

        The received messages are saved as two-element tuples. The first element is a string that communicates the topic
        the message was sent to. The second element is the payload of the message, which is a bytes or bytearray
        object. If no buffered objects are stored in the queue (queue is empty), returns None.
=======
=======
>>>>>>> origin/main
        This method should be used for sending data to Unity via one of the Gimbl-defined input topics. This method
        does not verify the validity of the input topic or payload data. Ensure both are correct given the specific
        configuration of Unity scripts and the version of the Gimbl library used to bind MQTT on Unity's side.

        Args:
            topic: The MQTT topic to publish the data to.
            payload: The data to be published. When set to None, an empty message will be sent, which is often used as
                a boolean trigger.
        """
    @property
    def has_data(self) -> bool:
        """Returns True if the instance received messages from Unity and can output received data via the get_dataq()
        method."""
    def get_data(self) -> tuple[str, bytes | bytearray] | None:
        """Extracts and returns the first available message stored inside the instance buffer queue.

        Returns:
            A two-element tuple. The first element is a string that communicates the MQTT topic of the received message.
            The second element is the payload of the message, which is a bytes or bytearray object. If no buffered
            objects are stored in the queue (queue is empty), returns None.
<<<<<<< HEAD
>>>>>>> 5b8062e (Added Communication module tests)
=======
>>>>>>> origin/main
        """
    def disconnect(self) -> None:
        """Disconnects the client from the MQTT broker."""
