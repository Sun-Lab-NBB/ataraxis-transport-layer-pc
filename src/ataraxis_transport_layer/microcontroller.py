from abc import abstractmethod
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from multiprocessing.managers import SyncManager
import numpy as np
from ataraxis_data_structures import NestedDictionary, SharedMemoryArray
from ataraxis_base_utilities import console
from .communication import ModuleData, ModuleState, KernelCommand, SerialCommunication, _prototypes


class ModuleInterface:
    """The base class from which all custom ModuleInterface classes should inherit.

    Interface classes encapsulates module-specific parameters and data handling methods which are used by the
    MicrocontrollerInterface class to communicate with individual hardware modules. Overall, this arrangement is similar
    to how custom modules inherit from the (base) Module class in the AtaraxisMicroController library.

    Interface classes loosely follow the structure of the AtaraxisMicroController (AXMC) library and allow the PC to
    receive and process data from the microcontrollers. Due to a high degree of custom module variability, it is
    currently not possible to provide a 'one-fits-all' Module interface that is also highly efficient for real time
    communication. Therefore, similar to AXMC library, the interface for each custom module has to be implemented
    separately on a need-base method. The (base) class exposes the static API that MicroControllerInterface class can
    use to integrate each custom interface implementation with the general communication runtime cycle.

    To make this integration possible, this class declares a number of abstract (pure virtual) methods that developers
    have to implement for their interfaces. Follow the implementation guidelines in the docstrings and check the
    default modules included with the library distribution for guidance.

    Notes:
        When inheriting from this class, remember to call the parent's init method in the child class init method by
        using 'super().__init__()'! if this is not done, the MicrocontrollerInterface class will likely not be able to
        properly interact with your ModuleInterface!

    Args:
        type_name: The name of the type (family) of Modules managed by this interface, 'e.g.: Rotary_Encoder'.
        module_type: The byte id-code of the type (family) of Modules managed by this interface. This has to match the
            code used by the module implementation in AXMC. Note, valid byte-codes range from 1 to 255.
        module_id: The instance byte-code ID of the module. This is used to identify unique instances of the same
            module type, such as different rotary encoders if more than one is used concurrently. Note, valid
            byte-codes range from 1 to 255.
        module_notes: Additional notes or description of the module. This can be used to provide further information
            about the interface module, such as the composition of its hardware or the location within broader
            experimental system. These notes will be instance-specific (unique given the module_type x module_id
            combination)!

    Attributes:
        _module_type: Store the type (family) of the interfaced module.
        _module_id: Stores specific id of the interfaced module within the broader type (family).
        _type_name: Stores a string-name of the module_type code. This is used to make the controller identifiable to
            humans, the code will only use the module_type code during runtime.
        _module_notes: Stores additional notes about the module.
        _custom_codes_map: A NestedDictionary that maps all custom status codes, command codes and data object layouts
            to meaningful names and additional descriptions. This dictionary is merged into the main map dictionary that
            aggregates the data for all microcontrollers used during runtime.
        _status_section: The dictionary path for the module-type-specific custom status codes section. This path is used
            when filling the custom_codes_map dictionary of the Module.
        _command_section: Same as _status_section, but stores command code mappings.

    """

    def __init__(self, type_name: str, module_type: np.uint8, module_id: np.uint8, module_notes: str | None = None):
        # Packages module type and id arguments into class attributes. They will also be added to the map dictionary
        # below, but keeping them as fields is helpful or faster value access (streamlines Controller-Module
        # interactions during communication cycling).
        self._module_type: np.uint8 = module_type
        self._module_id: np.uint8 = module_id
        self._type_name: str = type_name
        self._module_notes: str = '' if module_notes is None else module_notes

        # Precreates the custom_codes_map dictionary. This dictionary is filled by 'write' methods inherited from this
        # class. This class is used to process logged data after data acquisition runtime ends.
        self._custom_codes_map = NestedDictionary()

        # Adds and seeds the status_codes section to reserve code 0.
        self._status_section: str = f"{type_name}_module.status_codes"
        section = f"{self._status_section}.kUndefined"
        description = "This value is currently not used, but it statically reserves 0 as a non-valid status code."
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.code", value=0)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.description", value=description)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.error", value=False)

        # Adds and seeds the commands section to reserve code 0.
        self._command_section: str = f"{type_name}_module.commands"
        section = f"{self._command_section}.kUndefined"
        description = "This value is currently not used, but it statically reserves 0 as a non-valid command code."
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.code", value=0)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.description", value=description)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.addressable", value=False)

        self._data_object_section: str = f"{type_name}_module.data_objects"

    def write_status_code(self, code_name: str, code: int, description: str, error: bool):
        """Writes the provided status code information to the class code_map dictionary.

        This method allows interactively filling the code_map dictionary of the class. When MicroControllerInterface
        class initializes, it builds a dictionary that maps byte-codes used during communication to human-friendly
        names and additional descriptive data. This step is crucial, as all incoming and outgoing data is logged as
        serialized byte arrays for maximum throughput and parsed into more human-friendly formats offline. If the map
        dictionary is not created properly, it may be challenging or even impossible to parse the logged data.

        This method verifies that the input information is valid and, if so, creates and writes it as a new entry to the
        appropriate section of the local code_map dictionary. When this class is passed to the MicroControllerInterface
        class, it will extract and fuse the module-specific dictionary into the main runtime dictionary.

        Args:
            code_name: The meaningful name for the status represented by the code. It is advised to use the same names
                as used in the Microcontroller library code, e.g.: 'kValveOpen'.
            code: A value from 51 to 255 that represents the status in serial communication.
            description: The string that stores the description of the status. Use this field to provide information
                that may be relevant for future processing of the deserialized status data.
            error: A boolean flag that determines whether this is an error or a non-error code. This is used to optimize
                certain runtime aspects, such as 'online' error handling.
        """
        # If the input status code is not a valid byte-value, raises a ValueError
        if 255 < code < 51:
            message = (
                f"Unsupported byte-code value {code} encountered when adding a new {code_name} status_code entry to "
                f"the code map dictionary of {self._type_name} module. Valid custom status codes range from 51 to 255. "
                f"Codes 0 through 50 are reserved for system use."
            )
            console.error(message=message, error=ValueError)

        # Otherwise, writes the input
        section = f"{self._status_section}.{code_name}"
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.code", value=code)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.description", value=description)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.error", value=error)

    def write_command_code(self, code_name: str, code: int, description: str, addressable: bool):
        """Writes the provided command code information to the class code_map dictionary.

        Overall, this method serves a similar purpose as write_status_code() method, but is designed to create and
        write command code data as a new entry to the appropriate section of the local code_map dictionary. Command
        codes are used both when the controller sends data to the PC and when the PC requests the controller to do
        something (execute a command).

        Args:
            code_name: The meaningful name for the command represented by the code. It is advised to use the same names
                as used in the Microcontroller library code, e.g.: 'kSendPulse'.
            code: A value from 1 to 255 that represents the command in serial communication.
            description: The string that stores the description of the command. Use this field to provide information
                that may be relevant for future processing of the deserialized command data.
            addressable: A boolean flag that determines whether this command can be addressed (executed) from the PC
                or not. Notably, some commands can only be executed by the microcontroller itself or as part of larger
                command. This is most commonly seen for the AXMC Kernel class, where certain runtime-critical
                commands are not addressable by design.
        """
        # If the input code is outside the 1 to 255 range, raises a ValueError
        if 255 < code < 1:
            message = (
                f"Unsupported byte-code value {code} encountered when adding a new {code_name} command entry to "
                f"the class map dictionary of {self._type_name} module. Valid custom command codes range from 1 to "
                f"255. Code 0 is reserved for system use."
            )
            console.error(message=message, error=ValueError)

        # Otherwise, writes the input
        section = f"{self._status_section}.{code_name}"
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.code", value=code)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.description", value=description)
        self._custom_codes_map.write_nested_value(variable_path=f"{section}.addressable", value=addressable)

    def write_data_map(self, command_code: int, event_code: int, prototype_code: int,
                       prototype_field_names: tuple[int, ...], prototype_descriptions: tuple[str, ...]):

        # Extracts the matching prototype object
        prototype = _prototypes.get_prototype(code=prototype_code)

        # If extraction method returns None, then the input prototype_code is not supported.
        if prototype is None:
            message = (
                f"Invalid message_prototype {prototype_code} code encountered when adding a new data entry for "
                f"event {event_code} and command {command_code} combination to the code map dictionary of the "
                f"{self._type_name} module. Use one of the prototype codes available through the SerialPrototypes "
                f"dataclass."
            )
            console.error(message=message, error=ValueError)

        # Verifies that the length of prototype field names and descriptions tuples matches each other and the size of
        # the prototype. Since all prototypes are numpy arrays or scalars, their element-size can always be inferred
        # using the 'size' property.
        if len(prototype_field_names) != len(prototype_descriptions) != prototype.size:
            message = (
                f"The length of the prototype_field_names argument ({len(prototype_field_names)}) has to match the "
                f"length of the prototype_descriptions argument ({len(prototype_descriptions)}) and the length of the "
                f"prototype object ({prototype.size}) when adding a new data entry for event {event_code} and "
                f"command {command_code} combination to the code map dictionary of the {self._type_name} module."
            )
            console.error(message=message, error=ValueError)

        # Otherwise, adds additional descriptions for the message data object to the appropriate dictionary section.

        section = f"{self._data_object_section}.{command_code}_{event_code}_{prototype_code}"
        for name, description in zip(prototype_field_names, prototype_descriptions):
            self._custom_codes_map.write_nested_value(variable_path=f"{section}.{name}.description", value=description)

    @abstractmethod
    def process_data(self, message: ModuleData | ModuleState):
        raise NotImplementedError("process_data must be implemented by subclass")


class EncoderModule(ModuleInterface):
    def __init__(self, module_type: np.uint8, module_id: np.uint8):
        # Call parent's __init__ first
        super().__init__(type_name="Encoder", module_type=module_type, module_id=module_id)

        # Yes.

    def process_data(self, message: ModuleData | ModuleState):
        pass


class MicroControllerInterface:

    def __init__(
            self,
            name: str,
            usb_port: str,
            baudrate: int,
            maximum_transmitted_payload_size: int,
            id_code: int,
            modules: tuple[ModuleInterface, ...],
    ):
        self._name: str = name

        self._modules = modules

        # Sets up the multiprocessing Queue, which is used to buffer and pipe commands and parameters to be sent to the
        # microcontroller to the communication runtime method running on the isolated core (in a daemon Process).
        self._mp_manager: SyncManager = multiprocessing.Manager()
        self._transmission_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        # Also creates a Queue that is used to transfer the data to be logged to the Logger class. All logged data
        # will be queued in the form of byte numpy arrays.
        self._logger_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        # Instantiates an array that is used to terminate and adjust the communication runtime method. Like the Queue
        # object from above, this is a shared object that allows sharing data between isolated Processes (cores). Unlike
        # Queue, this object is a shared-buffer numpy array, which makes it uniquely adapted for communicating
        # runtime state-flags.
        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator_array",  # Uses class name to ensure the array buffer name is unique
            prototype=np.array([0, 0], dtype=np.uint8),
        )  # Instantiation automatically connects the main process to the array.

        # Sets up the communication process. This process continuously cycles through the communication loop until
        # terminated, enabling bidirectional communication with the controller.
        self._communication_loop: Process = Process(
            target=self.runtime_cycle,
            args=(self.transmission_queue, self._terminator_array, id_code, usb_port, baudrate,
                  maximum_transmitted_payload_size),
            daemon=True,
        )

        # Pre-packages Kernel commands to improve their runtime speed
        self._identify_command = KernelCommand(
            command=np.uint8(3),
            return_code=np.uint8(0),
        )

        self._reset_command = KernelCommand(
            command=self._identify_command_code,
            return_code=np.uint8(0),
        )

        self._communication.send_message(identify_command)

    @staticmethod
    def build_core_code_map() -> NestedDictionary:
        # Pre-initializes with a seed dictionary that includes the purpose (description) of the dictionary file
        message = (
            "This dictionary maps byte-values used by the Core classes that manage microcontroller runtime to "
            "meaningful names and provides a human-friendly description for each byte-code. This information is "
            "used by parser classes when decoding logged communication data, which is stored as serialized byte "
            "strings. Without a correct code-map, it will be impossible to accurately decode logged data! Note, this "
            "map only tracks the status codes of the Core classes, custom user-defined module assets are mapped by"
            "a different dictionary (custom_assets_code_map)."
        )
        code_dictionary = NestedDictionary(seed_dictionary={"description": message})

        # Kernel: module type
        # Since this is a very small section, does not wrap the code into a function.
        code_dictionary.write_nested_value(variable_path="kernel.module_type.code", value=1)
        message = "The byte-code that identifies messages sent by or to the Kernel module of the MicroController."
        code_dictionary.write_nested_value(variable_path="kernel.module_type.description", value=message)

        # Kernel: status codes
        # Uses a function for better code readability. This is done for most other sections.
        code_dictionary = MicroController._write_kernel_status_codes(code_dictionary)

        # Kernel: command codes
        code_dictionary = MicroController._write_kernel_command_codes(code_dictionary)

        # Module: core status codes.
        # Note, primarily, modules use custom status and command codes for each module family. These are available from
        # custom_codes_map dictionary. This section specifically tracks the 'core' codes inherited from the base Module
        # class.
        code_dictionary = MicroController._write_base_module_status_codes(code_dictionary)

        # Communication: status codes
        code_dictionary = MicroController._write_communication_status_codes(code_dictionary)

        # TransportLayer: status codes
        # This and the following sections track codes from classes wrapped by the Communication class. Due to the
        # importance of the communication library, we track all status codes that are (theoretically) relevant for
        # communication.
        code_dictionary = MicroController._write_transport_layer_status_codes(code_dictionary)

        # COBS: (Consistent Over Byte Stuffing) status codes
        code_dictionary = MicroController._write_cobs_status_codes(code_dictionary)

        # CRC: (Cyclic Redundancy Check) status codes
        code_dictionary = MicroController._write_crc_status_codes(code_dictionary)

        return code_dictionary

    @staticmethod
    def _write_kernel_status_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the kernel.status_codes section of the core_codes_map dictionary with data.

        The Kernel class manages the microcontroller runtime and encapsulates access to custom Module classes that
        directly control the connected hardware. Since Kernel acts as the mediator between the Modules and the PC,
        its status codes are used to convey the current runtime state and any broad runtime-related errors. This is
        in contrast to Modules, whose' status codes mostly reflect the current state of the hardware managed by the
        given Module.

        Make sure this method matches the actual state of the Kernel class from the AtaraxisMicroController library!

        Args:
            code_dictionary: The dictionary to be filled with kernel status codes.

        Returns:
            The updated dictionary with kernel status codes information filled.
        """
        section = "kernel.status_codes.kStandBy"
        description = "This value is currently not used, but it statically reserves 0 as a non-valid status code."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=0)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kSetupComplete"
        description = (
            "The microcontroller hardware (e.g.: pin modes) and software (e.g.: custom parameter structures) was "
            "successfully (re)set to hardcoded defaults."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=1)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleSetupError"
        description = (
            "The microcontroller was not able to (re)set its hardware and software due to one of the managed custom "
            "modules failing its' setup method runtime."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=2)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kDataReceptionError"
        description = (
            "The Kernel failed to parse the data sent from the PC. This can be due to a number of errors, including "
            "corruption of data in transmission and unsupported incoming message format."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=3)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kDataSendingError"
        description = (
            "The Kernel failed to send a DataMessage to the PC due to an underlying Communication or TransportLayer "
            "class failure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=4)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kStateSendingError"
        description = (
            "The Kernel failed to send a StateMessage to the PC due to an underlying Communication or "
            "TransportLayer class failure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=5)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kServiceSendingError"
        description = (
            "The Kernel failed to send a Service message to the PC due to an underlying Communication or "
            "TransportLayer class failure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=6)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kInvalidMessageProtocol"
        description = (
            "The Kernel has received a message from the PC that does not use a valid (supported) message protocol. "
            "The message protocol is communicated by the first variable of each message payload and determines how to "
            "parse the rest of the payload. This error typically indicates a mismatch between the PC and "
            "Microcontroller codebase versions or data corruption errors."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=7)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kKernelParametersSet"
        description = (
            "New parameter-values addressed to the Kernel (controller-wide DynamicRuntimeParameters) were received and "
            "applied successfully."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=8)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleParametersSet"
        description = (
            "New parameter-values addressed to the custom (user-defined) Module class instance were received and "
            "applied successfully."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=9)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleParametersError"
        description = (
            "Failed to apply the custom Module-addressed parameter-values. This may be due to mismatching data format "
            "that interferes with parameter data extraction, but can also be due to a different, module-class-specific "
            "error."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=10)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kCommandNotRecognized"
        description = (
            "The Kernel has received an unknown command code from the PC. Usually, this indicates data corruption or a "
            "mismatch between the PC and Microcontroller codebase versions."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=11)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kTargetModuleNotFound"
        description = (
            "Unable to find the Module addressed by a Command or Parameters message sent from the PC. The module_type "
            "and module_id fields of the message did not match any of the custom Modules. Usually, this indicates a "
            "malformed message (user-error)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=12)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @staticmethod
    def _write_kernel_command_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the kernel.commands section of the core_codes_map dictionary with data.

        The Kernel class manages the microcontroller runtime and encapsulates access to custom Module classes that
        directly control the connected hardware. The Kernel broadly has 2 types of commands: addressable and
        non-addressable. Addressable commands are triggered by sending a CommandMessage from the PC, non-addressable
        commands are hardcoded to execute during appropriate phases of microcontroller firmware runtime.

        Make sure this method matches the actual state of the Kernel class from the AtaraxisMicroController library!

        Args:
            code_dictionary: The dictionary to be filled with kernel command codes.

        Returns:
            The updated dictionary with kernel command codes information filled.
        """
        section = "kernel.commands.kStandby"
        description = "Standby code used during class initialization."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=0)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = "kernel.commands.kReceiveData"
        description = (
            "Attempts to receive and parse the command and parameters data sent from the PC. This command is "
            "automatically triggered at the beginning of each controller runtime cycle. Note, this command "
            "is always triggered before running any queued or newly received module commands."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=1)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = "kernel.commands.kResetController"
        description = (
            "(Re)sets the hardware and software parameters of the Kernel and all managed modules. This command code is "
            "used both during the initial setup of the controller and when the Kernel is instructed to reset the "
            "controller. Note, if the Setup runtime fails for any reason, the controller deadlocks in a mode that "
            "flashes the LED indicator. The controller firmware has to be reset to escape that mode (this is an "
            "intentional safety-promoting design choice)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=2)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = "kernel.commands.kIdentifyController"
        description = (
            "Transmits the unique ID of the controller that was hardcoded in the microcode firmware version running on "
            "the microcontroller. This command is used to verify the identity of the connected controller from the PC."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=3)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=True)

        return code_dictionary

    @staticmethod
    def _write_base_module_status_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the module.status_codes section of the core_codes_map dictionary with data.

        Module classes directly control the hardware connected to the microcontroller. Their status and command codes
        are primarily mapped by custom dictionary writer functions expected to be available through (python) Module
        class instances. However, since all custom Modules inherit from the same (base) Module class, some status codes
        used by custom modules come from the shared parent class. These shared status codes are added by this method, as
        they are the same across all modules.

        Make sure this method matches the actual state of the (base) Module class from the AtaraxisMicroController
        library!

        Args:
            code_dictionary: The dictionary to be filled with module status codes.

        Returns:
            The updated dictionary with module status codes information filled.
        """
        section = "module.status_codes.kStandBy"
        description = "The value used to initialize the class status tracker."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=0)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kDataSendingError"
        description = (
            "The Module failed to send a DataMessage to the PC. Usually, this indicates that the chosen data payload "
            "format is not valid."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=1)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "module.status_codes.kStateSendingError"
        description = (
            "The Module failed to send a StateMessage to the PC. State messages work similar to Data messages, but "
            "they are used in cases where data objects do not need to be included with event-codes. State messages "
            "allow optimizing data transmission by avoiding costly data-object-related logic and buffering."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=2)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "module.status_codes.kCommandCompleted"
        description = (
            "Indicates that the active command of the module has been completed. This status is reported whenever a "
            "command is replaced by a new command or is terminated with no further queued or recurring commands."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=3)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kCommandNotRecognized"
        description = (
            "This error-code indicates that a queued command was not recognized by the RunActiveCommand() method "
            "of the target module and, consequently, was not executed."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=4)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @staticmethod
    def _write_cobs_status_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the cobs.status_codes section of the core_codes_map dictionary with data.

        COBS (Consistent Overhead Byte Stuffing) is used during data transmission and reception to encode payloads into
        packet structures. This functionality is realized by the TransportLayer class via COBSProcessor helper. In turn,
        TransportLayer is wrapped by the Communication microcontroller class, which is used by Kernel and Module(s) to
        send and receive data.

        Make sure this method matches the actual state of the COBSProcessor class from the AtaraxisMicroController
        library!

        Args:
            code_dictionary: The dictionary to be filled with COBS status codes.

        Returns:
            The updated dictionary with COBS status codes information filled.
        """
        section = "cobs.status_codes.kStandby"
        description = "The value used to initialize the class status tracker."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=11)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "cobs.status_codes.kEncoderTooSmallPayloadSize"
        description = (
            "Failed to encode payload because payload size is too small. Valid payloads have to include at least 1 "
            "data byte."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=12)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kEncoderTooLargePayloadSize"
        description = (
            "Failed to encode payload because payload size is too large. Valid payloads can be at most 254 bytes in "
            "length to comply with COBS protocol limitations."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=13)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kEncoderPacketLargerThanBuffer"
        description = (
            "Failed to pack the encoded payload packet into the storage buffer, as the buffer does not have enough "
            "space to accommodate the encoded payload."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=14)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPayloadAlreadyEncoded"
        description = (
            "Failed to encode the payload, as it appears to be already encoded. This is inferred from the overhead "
            "byte placeholder in the buffer array being set to a non-0 value."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=15)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPayloadEncoded"
        description = "Payload was successfully encoded into a transmittable packet using COBS protocol."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=16)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "cobs.status_codes.kDecoderTooSmallPacketSize"
        description = (
            "Failed to decode the payload out of a COBS-encoded packet, because packet size is too small. The valid "
            "minimal packet size is 3 bytes (Overhead, 1 data byte, delimiter byte)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=17)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kDecoderTooLargePacketSize"
        description = (
            "Failed to decode the payload out of a COBS-encoded packet, because packet size is too large. The maximum "
            "supported packet size is 256 bytes (Overhead, 254 payload bytes, delimiter byte). This limitation is due "
            "to the COBS protocol's limitation on the maximum encoded payload size."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=18)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kDecoderPacketLargerThanBuffer"
        description = (
            "Failed to decode the payload out of a COBS-encoded packet, because the decoded payload is larger than the "
            "the available buffer space. This error has to do with how the microcontrollers handle data "
            "processing, which relies on using a shared static buffer for all operations. While unlikely, there can be "
            "a case where the buffer size is not allocated properly, leading to the microcontroller running out of "
            "space when decoding a large COBS-encoded packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=19)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kDecoderUnableToFindDelimiter"
        description = (
            "Failed to find the delimiter at the end of the packet. All valid COBS-encoded packets have to end with an "
            "unencoded delimiter value. If this expectation is violated, this likely indicates that the data was "
            "corrupted during transmission (and the CRC check failed to detect that)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=20)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kDecoderDelimiterFoundTooEarly"
        description = (
            "Found the unencoded delimiter value before reaching the end of the packet. Valid COBS-encoded packets "
            "only have a single unencoded delimiter value at the end of the packet. If this expectation is violated, "
            "this likely indicates that the data was corrupted during transmission or the two communicating systems "
            "are using different delimiter byte values."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=21)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPacketAlreadyDecoded"
        description = (
            "Failed to decode the packet, as it appears to be already decoded. This is inferred from the overhead "
            "byte placeholder in the buffer array being set to a 0 value. An overhead byte for the valid packet has to "
            "be a value between 1 and 255."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=22)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPayloadDecoded"
        description = "Payload was successfully decoded from the received COBS-encoded packet."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=23)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        return code_dictionary

    @staticmethod
    def _write_crc_status_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the crc.status_codes section of the core_codes_map dictionary with data.

        CRC (Cyclic Redundancy Check) is used during data transmission and reception to verify the integrity of the
        transmitted data. Specifically, it involves dividing the COBS-encoded data by a polynomial and using the
        remained of the division as the checksum or the transmitted data. This functionality is realized by the
        TransportLayer class via CRCProcessor helper. In turn, TransportLayer is wrapped by the Communication
        microcontroller class, which is used by Kernel and Module(s) to send and receive data.

        Make sure this method matches the actual state of the CRCProcessor class from the AtaraxisMicroController
        library!

        Args:
            code_dictionary: The dictionary to be filled with CRC status codes.

        Returns:
            The updated dictionary with CRC status codes information filled.
        """
        section = "crc.status_codes.kStandby"
        description = "The value used to initialize the class status tracker."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=51)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "crc.status_codes.kCalculateCRCChecksumBufferTooSmall"
        description = (
            "Failed to calculate the CRC checksum, because the size of the buffer that holds the packet is  too small. "
            "Specifically, the buffer used for CRC calculation has to be at least 3 bytes in size, consistent with the "
            "valid minimum size of the COBS-encoded packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=52)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "crc.status_codes.kCRCChecksumCalculated"
        description = "CRC checksum for the COBS-encoded packet was successfully calculated."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=53)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "crc.status_codes.kAddCRCChecksumBufferTooSmall"
        description = (
            "Unable to append the calculated CRC checksum to the packet-containing buffer. Since the microcontroller "
            "uses the same static buffer for all data transmission operations, it is possible that the buffer was not "
            "allocated properly, leading to the microcontroller running out of space when appending the checksum."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=54)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "crc.status_codes.kCRCChecksumAddedToBuffer"
        description = "Calculated CRC checksum was successfully added to the packet buffer."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=55)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "crc.status_codes.kReadCRCChecksumBufferTooSmall"
        description = (
            "Unable to read the CRC checksum transmitted with the packet from the shared buffer. Usually, this "
            "indicates that the PC and the microcontroller use different CRC sizes. Specifically, if a PC uses a "
            "32-bit CRC, while the microcontroller uses a 16-bit CRC, this error could occur due to static "
            "microcontroller buffer size allocation."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=56)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "crc.status_codes.kCRCChecksumReadFromBuffer"
        description = "CRC checksum transmitted with the packet was successfully read from the shared buffer."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=57)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        return code_dictionary

    @staticmethod
    def _write_transport_layer_status_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the transport_layer.status_codes section of the core_codes_map dictionary with data.

        Transport Layer class carries out the necessary low-level transformations to send and receive data over the
        serial interface. This class is wrapped and used by the Communication class to carry out the PC-microcontroller
        communication.

        Make sure this method matches the actual state of the TransportLayer class from the AtaraxisMicroController
        library!

        Args:
            code_dictionary: The dictionary to be filled with Transport Layer status codes.

        Returns:
            The updated dictionary with Transport Layer status codes information filled.
        """
        section = "transport_layer.status_codes.kStandby"
        description = "The value used to initialize the class status tracker."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=101)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketConstructed"
        description = "The serialized data packet to be sent to the PC was successfully constructed."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=102)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketSent"
        description = "The serialized data packet was successfully sent to the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=103)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketStartByteFound"
        description = (
            "Found the start byte of the incoming packet when parsing received serialized data. This "
            "indicates that the processed serial stream contains a valid data packet to be parsed."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=104)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketStartByteNotFound"
        description = (
            "Unable to find the start byte of the incoming packet in the incoming serial data stream. Since serial "
            "communication interface can 'receive' noise-bytes, packet reception only starts when start byte value is "
            "found. If this value is not found, this indicates that either no data was received, or that a "
            "communication error has occurred."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=105)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPayloadSizeByteFound"
        description = (
            "Found the payload size byte of the incoming packet when parsing received serialized data. This byte "
            "is used to determine the size of the incoming packet, which is needed dot correctly parse the packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=106)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPayloadSizeByteNotFound"
        description = (
            "Unable to find the payload size byte of the incoming packet in the incoming serial data stream. Since "
            "this information is needed to correctly parse the packet (it is used to verify packet integrity), without "
            "this information, the packet parsing cannot be completed."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=107)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kInvalidPayloadSize"
        description = (
            "The found payload size is not valid. Specifically, valid payloads can have a size between 1 and 254 (the "
            "upper limit is due to COBS specifications). Encountering a payload size value of 255 or 0 would, "
            "therefore, trigger this error."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=108)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPacketTimeoutError"
        description = (
            "Unable to parse the incoming packet, as packet reception has stalled. If reception starts before all "
            "bytes of the packet are received by the microcontroller, the parser will wait a reasonable amount of "
            "time to receive the missing bytes. If these bytes do not arrive in time, this error is triggered. This "
            "error specifically applies to parsing the payload of the packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=109)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kNoBytesToParseFromBuffer"
        description = (
            "The microcontroller did not receive any bytes to parse the packet from or the received bytes did not "
            "contain packet data (were noise-generated). This is a non-error status used to communicate that there "
            "was no packet data to process."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=110)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketParsed"
        description = "Packet was successfully parsed from the received serial bytes stream."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=111)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kCRCCheckFailed"
        description = (
            "The parsed packet has failed the CRC check. This indicates that packet's data was corrupted in "
            "transmission. Alternatively, this can suggest that the PC and microcontroller use non-matching CRC "
            "parameters."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=112)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPacketValidated"
        description = (
            "The parsed packet's integrity was validated by passing a CRC check. The packet was not corrupted during "
            "transmission."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=113)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketReceived"
        description = (
            "The packet sent from the PC was successfully received, parsed and validated and is ready for payload "
            "decoding."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=114)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kWriteObjectBufferError"
        description = (
            "Unable to write the provided object to the message payload buffer. The TransportLayer serializes the "
            "objects (data) to be sent to the PC into a shared bytes buffer. If the provided object is too large to "
            "fit into the available buffer space, this error is triggered."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=115)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kObjectWrittenToBuffer"
        description = (
            "The object (data) to be sent to the PC has been successfully serialized (written) into the message "
            "payload buffer."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=116)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kReadObjectBufferError"
        description = (
            "Unable to read the provided object's data from the received message payload buffer. The received data has "
            "to be deserialized (converted from bytes to the original format) by the TransportLayer class using "
            "provided 'prototypes' or 'containers' to infer the data format. If the container requests more data than "
            "available from the parsed message buffer, this error is triggered."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=117)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kObjectReadFromBuffer"
        description = (
            "The object (data) received from the PC has been successfully deserialized (read) from the parsed message "
            "payload buffer."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=118)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kDelimiterNotFoundError"
        description = (
            "Unable to find the unencoded delimiter byte value at the end of the incoming packet. Since TransportLayer "
            "class carries out basic data validation as it parses the packet from the serial byte stream, it checks "
            "that the packet endswith a delimiter. If this expectation is violated, this error is triggered to "
            "indicate potential data corruption."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=119)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kDelimiterFoundTooEarlyError"
        description = (
            "The delimiter byte value that is expected to be found at the end of the incoming packet is found before "
            "reaching teh end of the packet. This indicates data corruption."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=120)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPostambleTimeoutError"
        description = (
            "Unable to parse the incoming packet, as packet reception has stalled. If reception starts before all "
            "bytes of the packet are received by the microcontroller, the parser will wait a reasonable amount of "
            "time to receive the missing bytes. If these bytes do not arrive in time, this error is triggered. This "
            "error specifically applies to parsing the CRC postamble of the packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=121)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @staticmethod
    def _write_communication_status_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the communication.status_codes section of the core_codes_map dictionary with data.

        Communication class wraps TransportLayer and provides a high-level API interface for communicating between
        PC and microcontrollers running Ataraxis software. Together with Kernel and (base) Module, the Communication
        class forms the 'core' triad of classes that jointly manage the runtime of every Ataraxis-compatible
        microcontroller.

        Make sure this method matches the actual state of the Communication class from the AtaraxisMicroController
        library!

        Args:
            code_dictionary: The dictionary to be filled with Communication status codes.

        Returns:
            The updated dictionary with Communication status codes information filled.
        """
        section = "communication.status_codes.kStandby"
        description = "Standby placeholder used to initialize the Communication class status tracker."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=151)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kReceptionError"
        description = "Communication class ran into an error when attempting to receive a message from the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=152)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kParsingError"
        description = "Communication class ran into an error when parsing (decoding) a message received from the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=153)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kPackingError"
        description = (
            "Communication class ran into an error when writing (serializing) the message data into the transmission "
            "payload buffer."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=154)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kTransmissionError"
        description = "Communication class ran into an error when transmitting (sending) a message to the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=155)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kMessageSent"
        description = "Communication class successfully transmitted a message to the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=156)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kMessageReceived"
        description = "Communication class successfully received a message from the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=157)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kInvalidProtocol"
        description = (
            "The received or transmitted protocol code is not valid for that type of operation. This error is raised "
            "whenever the encountered message protocol code is not one of the expected codes for the given operation. "
            "With the way the microcontroller library is written, this applies in two cases: when sending Service "
            "messages and when receiving the data from the PC. Currently, the controller only expects Command and "
            "Parameters messages to be received from the PC and only expects to send ReceptionCode and Identification "
            "service messages. Sending State and Data messages is also possible, but those structures are hardcoded to "
            "always be correct on the microcontroller's side."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=158)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kNoBytesToReceive"
        description = (
            "Communication class did not receive enough bytes to process the message. This is not an error, most "
            "higher-end microcontrollers will spend a sizeable chunk of their runtime with no communication data to "
            "process."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=159)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kParameterMismatch"
        description = (
            "The number of extracted parameter bytes does not match the size of the input structure. Currently, this "
            "only applies to Module-addressed parameter structures. Since the exact size and format of the "
            "Module-addressed parameters structure is not known at compile time, the class receives such messages in "
            "two steps. First, it uses the message header to identify the target module and then instructs the module "
            "to parse the parameters object that follows the message header. If the number of bytes necessary to fill "
            "the module's parameter object with data does not exactly match the size of the data contained in the "
            "message parameters payload section, this error is raised. Seeing this error suggests that the parameters "
            "sent in the message were not intended for the allegedly targeted module (due to inferred parameter "
            "structure mismatch)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=160)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kParametersExtracted"
        description = (
            "Module parameter data has been successfully extracted and written into the module's parameter structure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=161)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kExtractionForbidden"
        description = (
            "Unable to extract Module-addressed parameters, as the class currently holds a different message in "
            "its reception buffer. Currently, only Module-addressed parameters need to be extracted by a separate "
            "method call. Calling the method for Kernel-addressed parameters (or any other message) will produce this "
            "error. Since TransportLayer only holds one message in its reception buffer at a time, the module "
            "parameters have to be extracted by the addressed module before the Communication is instructed to receive "
            "a new message. Otherwise, the unprocessed parameter data will be lost."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=162)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @staticmethod
    def runtime_cycle(command_queue: MPQueue, terminator_array: SharedMemoryArray, controller_id: int,
                      usb_port: name, baudrate: int, maximum_transmitted_payload_size: int) -> None:
        # Initializes the communication class. It is critical that this is done inside the method running in an
        # isolated process.
        communication = SerialCommunication(usb_port=usb_port, baudrate=baudrate,
                                            maximum_transmitted_payload_size=maximum_transmitted_payload_size)
        terminator_array.connect()

        while True:
            out_data = command_queue.get()
            communication.send_message(out_data)

            if terminator_array.read_data(index=0, convert_output=True):
                break  # Terminates the loop

    def identify_controller(self):
        self.transmission_queue.put(self._identify_command)
