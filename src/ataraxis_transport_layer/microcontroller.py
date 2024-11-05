from abc import abstractmethod
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
)
from queue import Empty
from multiprocessing.managers import SyncManager
import numpy as np
from ataraxis_data_structures import NestedDictionary, SharedMemoryArray
from ataraxis_base_utilities import console
from ataraxis_time.precision_timer.timer_class import PrecisionTimer
from numba import uint8

from .communication import (
    Identification,
    ReceptionCode,
    RepeatedModuleCommand,
    OneOffModuleCommand,
    DequeueModuleCommand,
    KernelCommand,
    KernelData,
    KernelState,
    ModuleData,
    ModuleState,
    KernelParameters,
    ModuleParameters,
    SerialCommunication,
    prototypes,
    protocols,
)


class ModuleInterface:
    """The base class from which all custom ModuleInterface classes should inherit.

    Interface classes encapsulate module-specific parameters and data handling methods which are used by the
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
        type_name: The name of the Module type (family) managed by this interface, 'e.g.: Rotary_Encoder'.
        module_type: The byte id-code of the type (family) of Modules managed by this interface. This has to match the
            code used by the module implementation in AXMC. Note, valid byte-codes range from 1 to 255.
        module_id: The instance byte-code ID of the module. This is used to identify unique instances of the same
            module type, such as different rotary encoders if more than one is used concurrently. Note, valid
            byte-codes range from 1 to 255.
        module_notes: Additional notes or description of the module. This can be used to provide further information
            about the interface module, such as the composition of its hardware or the location within the broader
            experimental system. These notes will be instance-specific (unique given the module_type x module_id
            combination)!
        process_data: A boolean flag that determines whether this module has additional logic to process
            incoming data other than logging it (which is done for all received and sent data automatically). Use this
            flag to optimize runtime performance by disabling unnecessary checks and runtimes for modules that do not
            contain custom data processing logic. Note, regardless of this flag's value, you still need to implement the
            process_data() abstract method, but it may not be called at all during runtime.

    Attributes:
        _module_type: Store the type (family) of the interfaced module.
        _module_id: Stores specific id of the interfaced module within the broader type (family).
        _type_name: Stores a string-name of the module_type code. This is used to make the controller identifiable to
            humans, the code will only use the module_type code during runtime.
        _module_notes: Stores additional notes about the module.
    """

    def __init__(
        self,
        type_name: str,
        module_type: np.uint8,
        module_id: np.uint8,
        module_notes: str | None = None,
        *,
        process_data: bool = False,
    ) -> None:
        # Transfers arguments to class attributes.
        self._module_type: np.uint8 = module_type
        self._module_id: np.uint8 = module_id
        self._type_name: str = type_name
        self._module_notes: str = "" if module_notes is None else module_notes
        self._process_data: bool = process_data

    @abstractmethod
    def process_data(self, message: ModuleData | ModuleState):
        """Contains the additional processing logic for incoming State and Data messages.

        This method allows providing custom data-handling logic for some or all State and Data messages sent by the
        Module to the PC. Note, this method should NOT contain data logging, as all incoming and outgoing messages are
        automatically logged. Instead, it should contain specific data-driven actions, e.g.: Sending a command to Unity
        if the Module transmits a certain state code.

        Notes:
            This method should contain a sequence of if-else statements that filter and execute the necessary logic
            depending on the command and codes of the message and, for data messages, specific data object values.
            See one of the default module interface implementations for details on how to write this method.
        """
        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"process_data method for {self._type_name} Module must be implemented when subclassing the base "
            f"ModuleInterface."
        )

    @abstractmethod
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        """Writes custom module status, command, and object data information to the provided code_map dictionary.

        This method is called by the MicroControllerInterface that manages the Module to fill the shared code_map
        dictionary with module-specific data. This maps number-codes used during serialized communication to represent
        commands, events, and additional data objects to human-readable names and descriptions. In turn, this
        information is used to transform logged data, which is stored as serialized byte-strings, into a format more
        suitable for data analysis and long-term storage.

        Notes:
            See MicroControllerInterface class for examples on how to write this method (and fill the code_map
            dictionary). Note, if this method is not implemented properly, it may be challenging to decode the logged
            data in the future.

            This method should fill all relevant module-type sections: commands, status_codes, and object_data.
            This method will only be called once for each unique module_type.
        """
        raise NotImplementedError(
            f"write_code_map method for {self._type_name} Module must be implemented when subclassing the base "
            f"ModuleInterface."
        )

    @property
    def module_type(self) -> np.uint8:
        """Returns the module's type (family) byte-code."""
        return self._module_type

    @property
    def type_name(self) -> str:
        """Returns the module's type (family) human-readable name."""
        return self._type_name

    @property
    def module_id(self) -> np.uint8:
        """Returns the module's ID byte-code (instance-specific identifier code)."""
        return self._module_id

    @property
    def module_notes(self) -> str:
        """Returns additional notes for the specific module instance (unique for each module_id and module_type
        combination)."""
        return self._module_notes


class MicroControllerInterface:

    def __init__(
        self,
        name: str,
        controller_id: int,
        controller_description: str,
        usb_port: str,
        baudrate: int,
        maximum_transmitted_payload_size: int,
        logger_queue: MPQueue,
        logger_source_id: int,
        modules: tuple[ModuleInterface, ...],
    ):
        # Saves controller id information to class attributes
        self._name: str = name
        self._controller_id: int = controller_id

        # Also saves Module tuple into class attribute
        self._modules = modules

        # Also stores the information to connect to the controller via the USB / UART interface
        self._usb_port: str = usb_port
        self._baudrate: int = baudrate
        self._maximum_transmitted_payload_size: int = maximum_transmitted_payload_size

        # Also saves the Logger queue reference and the static source ID code assigned to the microcontroller at
        # initialization
        self._logger_queue: MPQueue = logger_queue
        self._logger_source_id: int = logger_source_id

        # Sets up the multiprocessing Queue, which is used to buffer and pipe commands and parameters to be sent to the
        # microcontroller to the communication runtime method running on the isolated core (in a daemon Process).
        self._mp_manager: SyncManager = multiprocessing.Manager()
        self._transmission_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        # Initializes the PrecisionTimer class which is used to time-stamp logged data.
        self._timer = PrecisionTimer(precision="us")

        # Instantiates the shared array used to bidirectionally communicate with the runtime of the daemon process.
        # This array acts in parallel with the command / parameters Queue and offers a better way of communicating the
        # runtime data between the main process that controls the experiment and the daemon process running the
        # microcontroller interface.
        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator_array",  # Uses class name to ensure the array buffer name is unique
            prototype=np.zeros(shape=1, dtype=np.uint8),
        )  # Instantiation automatically connects the main process to the array.

        # Pre-packages Kernel commands into attributes. Since Kernel commands are known and fixed at compilation,
        # they only need to be defined once.
        self._reset_command = KernelCommand(
            command=np.uint8(2),
            return_code=np.uint8(0),
        )
        self._identify_command = KernelCommand(
            command=np.uint8(3),
            return_code=np.uint8(0),
        )

        # Sets up the communication process. This process continuously cycles through the communication loop until
        # terminated, enabling bidirectional communication with the controller.
        self._communication_process: Process = Process(
            target=self.runtime_cycle,
            args=(self,),
            daemon=True,
        )

    def build_core_code_map(self) -> NestedDictionary:
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

        # Kernel: status codes
        code_dictionary = self._write_kernel_status_codes(code_dictionary)

        # Kernel: command codes
        code_dictionary = self._write_kernel_command_codes(code_dictionary)

        # Module: core status codes.
        # Note, primarily, modules use custom status and command codes for each module family. These are available from
        # custom_codes_map dictionary. This section specifically tracks the 'core' codes inherited from the base Module
        # class.
        code_dictionary = self._write_base_module_status_codes(code_dictionary)

        # Communication: status codes
        code_dictionary = self._write_communication_status_codes(code_dictionary)

        # TransportLayer: status codes
        # This and the following sections track codes from classes wrapped by the Communication class. Due to the
        # importance of the communication library, we track all status codes that are (theoretically) relevant for
        # communication.
        code_dictionary = self._write_transport_layer_status_codes(code_dictionary)

        # COBS: (Consistent Over Byte Stuffing) status codes
        code_dictionary = self._write_cobs_status_codes(code_dictionary)

        # CRC: (Cyclic Redundancy Check) status codes
        code_dictionary = self._write_crc_status_codes(code_dictionary)

        # Generates and appends custom module information to the dictionary
        added_modules = set()  # This is used to ensure custom information is added once per type
        for module in self._modules:
            if module.module_type in added_modules:
                code_dictionary.write_nested_value(
                    variable_path=f"{module.type_name}_module.id.{module.module_id}", value=module.module_notes
                )
                continue

            added_modules.add(module.module_type)

            code_dictionary.write_nested_value(variable_path=f"{module.type_name}_module", value=module.code_map)

            code_dictionary.write_nested_value(
                variable_path=f"{module.type_name}_module.code.id", value=module.module_type
            )
            code_dictionary.write_nested_value(
                variable_path=f"{module.type_name}_module.id.{module.module_id}", value=module.module_notes
            )

        return code_dictionary

    @classmethod
    def _write_kernel_status_codes(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(0))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kSetupComplete"
        description = (
            "The microcontroller hardware (e.g.: pin modes) and software (e.g.: custom parameter structures) was "
            "successfully (re)set to hardcoded defaults."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleSetupError"
        description = (
            "The microcontroller was not able to (re)set its hardware and software due to one of the managed custom "
            "modules failing its' setup method runtime."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kDataReceptionError"
        description = (
            "The Kernel failed to parse the data sent from the PC. This can be due to a number of errors, including "
            "corruption of data in transmission and unsupported incoming message format."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kDataSendingError"
        description = (
            "The Kernel failed to send a DataMessage to the PC due to an underlying Communication or TransportLayer "
            "class failure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(4))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kStateSendingError"
        description = (
            "The Kernel failed to send a StateMessage to the PC due to an underlying Communication or "
            "TransportLayer class failure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(5))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kServiceSendingError"
        description = (
            "The Kernel failed to send a Service message to the PC due to an underlying Communication or "
            "TransportLayer class failure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(6))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kInvalidMessageProtocol"
        description = (
            "The Kernel has received a message from the PC that does not use a valid (supported) message protocol. "
            "The message protocol is communicated by the first variable of each message payload and determines how to "
            "parse the rest of the payload. This error typically indicates a mismatch between the PC and "
            "Microcontroller codebase versions or data corruption errors."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(7))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kKernelParametersSet"
        description = (
            "New parameter-values addressed to the Kernel (controller-wide DynamicRuntimeParameters) were received and "
            "applied successfully."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(8))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleParametersSet"
        description = (
            "New parameter-values addressed to the custom (user-defined) Module class instance were received and "
            "applied successfully."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(9))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleParametersError"
        description = (
            "Failed to apply the custom Module-addressed parameter-values. This may be due to mismatching data format "
            "that interferes with parameter data extraction, but can also be due to a different, module-class-specific "
            "error."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(10))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kCommandNotRecognized"
        description = (
            "The Kernel has received an unknown command code from the PC. Usually, this indicates data corruption or a "
            "mismatch between the PC and Microcontroller codebase versions."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(11))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kTargetModuleNotFound"
        description = (
            "Unable to find the Module addressed by a Command or Parameters message sent from the PC. The module_type "
            "and module_id fields of the message did not match any of the custom Modules. Usually, this indicates a "
            "malformed message (user-error)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(12))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @classmethod
    def _write_kernel_command_codes(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(0))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = "kernel.commands.kReceiveData"
        description = (
            "Attempts to receive and parse the command and parameters data sent from the PC. This command is "
            "automatically triggered at the beginning of each controller runtime cycle. Note, this command "
            "is always triggered before running any queued or newly received module commands."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = "kernel.commands.kIdentifyController"
        description = (
            "Transmits the unique ID of the controller that was hardcoded in the microcode firmware version running on "
            "the microcontroller. This command is used to verify the identity of the connected controller from the PC."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=True)

        return code_dictionary

    @classmethod
    def _write_kernel_object_data(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the kernel.object_data section of the core_codes_map dictionary with data.

        This section is used to provide additional information about the values of the data objects used by KernelData
        messages. These objects usually have a different interpretation depending on the event-code of the message they
        are sent with.

        Args:
            code_dictionary: The dictionary to be filled with kernel object data.

        Returns:
            The updated dictionary with kernel object data information filled.
        """

        section = "kernel.data_objects.kModuleSetupErrorObject"
        description_1 = "The type-code of the module that failed its setup sequence."
        description_2 = "The id-code of the module that failed its setup sequence."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=uint8(2))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.names", value=("module_type", "module_id"))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "kernel.data_objects.kInvalidMessageProtocolObject"
        description_1 = "The invalid protocol byte-code value that was received by the Kernel."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=uint8(7))
        code_dictionary.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedByte)
        code_dictionary.write_nested_value(variable_path=f"{section}.names", value=("protocol_code",))
        code_dictionary.write_nested_value(variable_path=f"{section}.descriptions", value=(description_1,))

        section = "kernel.data_objects.kModuleParametersErrorObject"
        description_1 = (
            "The type-code of the module that failed to extract and apply its parameter data from received message."
        )
        description_2 = (
            "The id-code of the module that failed to extract and apply its parameter data from received message."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=uint8(10))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.names", value=("module_type", "module_id"))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "kernel.data_objects.kTargetModuleNotFoundObject"
        description_1 = (
            "The type-code of the addressed module transmitted by the message whose addressee was not found."
        )
        description_2 = "The id-code of the addressed module transmitted by the message whose addressee was not found."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=uint8(12))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.names", value=("target_type", "target_id"))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        return code_dictionary

    @classmethod
    def _write_base_module_status_codes(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(0))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kDataSendingError"
        description = (
            "The Module failed to send a DataMessage to the PC. Usually, this indicates that the chosen data payload "
            "format is not valid."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "module.status_codes.kStateSendingError"
        description = (
            "The Module failed to send a StateMessage to the PC. State messages work similar to Data messages, but "
            "they are used in cases where data objects do not need to be included with event-codes. State messages "
            "allow optimizing data transmission by avoiding costly data-object-related logic and buffering."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "module.status_codes.kCommandCompleted"
        description = (
            "Indicates that the active command of the module has been completed. This status is reported whenever a "
            "command is replaced by a new command or is terminated with no further queued or recurring commands."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kCommandNotRecognized"
        description = (
            "This error-code indicates that a queued command was not recognized by the RunActiveCommand() method "
            "of the target module and, consequently, was not executed."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(4))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @classmethod
    def _write_cobs_status_codes(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(11))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "cobs.status_codes.kEncoderTooSmallPayloadSize"
        description = (
            "Failed to encode payload because payload size is too small. Valid payloads have to include at least 1 "
            "data byte."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(12))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kEncoderTooLargePayloadSize"
        description = (
            "Failed to encode payload because payload size is too large. Valid payloads can be at most 254 bytes in "
            "length to comply with COBS protocol limitations."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(13))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kEncoderPacketLargerThanBuffer"
        description = (
            "Failed to pack the encoded payload packet into the storage buffer, as the buffer does not have enough "
            "space to accommodate the encoded payload."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(14))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPayloadAlreadyEncoded"
        description = (
            "Failed to encode the payload, as it appears to be already encoded. This is inferred from the overhead "
            "byte placeholder in the buffer array being set to a non-0 value."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(15))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPayloadEncoded"
        description = "Payload was successfully encoded into a transmittable packet using COBS protocol."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(16))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "cobs.status_codes.kDecoderTooSmallPacketSize"
        description = (
            "Failed to decode the payload out of a COBS-encoded packet, because packet size is too small. The valid "
            "minimal packet size is 3 bytes (Overhead, 1 data byte, delimiter byte)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(17))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kDecoderTooLargePacketSize"
        description = (
            "Failed to decode the payload out of a COBS-encoded packet, because packet size is too large. The maximum "
            "supported packet size is 256 bytes (Overhead, 254 payload bytes, delimiter byte). This limitation is due "
            "to the COBS protocol's limitation on the maximum encoded payload size."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(18))
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(19))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kDecoderUnableToFindDelimiter"
        description = (
            "Failed to find the delimiter at the end of the packet. All valid COBS-encoded packets have to end with an "
            "unencoded delimiter value. If this expectation is violated, this likely indicates that the data was "
            "corrupted during transmission (and the CRC check failed to detect that)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(20))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kDecoderDelimiterFoundTooEarly"
        description = (
            "Found the unencoded delimiter value before reaching the end of the packet. Valid COBS-encoded packets "
            "only have a single unencoded delimiter value at the end of the packet. If this expectation is violated, "
            "this likely indicates that the data was corrupted during transmission or the two communicating systems "
            "are using different delimiter byte values."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(21))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPacketAlreadyDecoded"
        description = (
            "Failed to decode the packet, as it appears to be already decoded. This is inferred from the overhead "
            "byte placeholder in the buffer array being set to a 0 value. An overhead byte for the valid packet has to "
            "be a value between 1 and 255."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(22))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "cobs.status_codes.kPayloadDecoded"
        description = "Payload was successfully decoded from the received COBS-encoded packet."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(23))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        return code_dictionary

    @classmethod
    def _write_crc_status_codes(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "crc.status_codes.kCalculateCRCChecksumBufferTooSmall"
        description = (
            "Failed to calculate the CRC checksum, because the size of the buffer that holds the packet is  too small. "
            "Specifically, the buffer used for CRC calculation has to be at least 3 bytes in size, consistent with the "
            "valid minimum size of the COBS-encoded packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(52))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "crc.status_codes.kCRCChecksumCalculated"
        description = "CRC checksum for the COBS-encoded packet was successfully calculated."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(53))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "crc.status_codes.kAddCRCChecksumBufferTooSmall"
        description = (
            "Unable to append the calculated CRC checksum to the packet-containing buffer. Since the microcontroller "
            "uses the same static buffer for all data transmission operations, it is possible that the buffer was not "
            "allocated properly, leading to the microcontroller running out of space when appending the checksum."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(54))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "crc.status_codes.kCRCChecksumAddedToBuffer"
        description = "Calculated CRC checksum was successfully added to the packet buffer."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(55))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "crc.status_codes.kReadCRCChecksumBufferTooSmall"
        description = (
            "Unable to read the CRC checksum transmitted with the packet from the shared buffer. Usually, this "
            "indicates that the PC and the microcontroller use different CRC sizes. Specifically, if a PC uses a "
            "32-bit CRC, while the microcontroller uses a 16-bit CRC, this error could occur due to static "
            "microcontroller buffer size allocation."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(56))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "crc.status_codes.kCRCChecksumReadFromBuffer"
        description = "CRC checksum transmitted with the packet was successfully read from the shared buffer."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(57))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        return code_dictionary

    @classmethod
    def _write_transport_layer_status_codes(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(101))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketConstructed"
        description = "The serialized data packet to be sent to the PC was successfully constructed."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(102))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketSent"
        description = "The serialized data packet was successfully sent to the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(103))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketStartByteFound"
        description = (
            "Found the start byte of the incoming packet when parsing received serialized data. This "
            "indicates that the processed serial stream contains a valid data packet to be parsed."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(104))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketStartByteNotFound"
        description = (
            "Unable to find the start byte of the incoming packet in the incoming serial data stream. Since serial "
            "communication interface can 'receive' noise-bytes, packet reception only starts when start byte value is "
            "found. If this value is not found, this indicates that either no data was received, or that a "
            "communication error has occurred."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(105))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPayloadSizeByteFound"
        description = (
            "Found the payload size byte of the incoming packet when parsing received serialized data. This byte "
            "is used to determine the size of the incoming packet, which is needed dot correctly parse the packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(106))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPayloadSizeByteNotFound"
        description = (
            "Unable to find the payload size byte of the incoming packet in the incoming serial data stream. Since "
            "this information is needed to correctly parse the packet (it is used to verify packet integrity), without "
            "this information, the packet parsing cannot be completed."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(107))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kInvalidPayloadSize"
        description = (
            "The found payload size is not valid. Specifically, valid payloads can have a size between 1 and 254 (the "
            "upper limit is due to COBS specifications). Encountering a payload size value of 255 or 0 would, "
            "therefore, trigger this error."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(108))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPacketTimeoutError"
        description = (
            "Unable to parse the incoming packet, as packet reception has stalled. If reception starts before all "
            "bytes of the packet are received by the microcontroller, the parser will wait a reasonable amount of "
            "time to receive the missing bytes. If these bytes do not arrive in time, this error is triggered. This "
            "error specifically applies to parsing the payload of the packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(109))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kNoBytesToParseFromBuffer"
        description = (
            "The microcontroller did not receive any bytes to parse the packet from or the received bytes did not "
            "contain packet data (were noise-generated). This is a non-error status used to communicate that there "
            "was no packet data to process."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(110))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketParsed"
        description = "Packet was successfully parsed from the received serial bytes stream."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(111))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kCRCCheckFailed"
        description = (
            "The parsed packet has failed the CRC check. This indicates that packet's data was corrupted in "
            "transmission. Alternatively, this can suggest that the PC and microcontroller use non-matching CRC "
            "parameters."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(112))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPacketValidated"
        description = (
            "The parsed packet's integrity was validated by passing a CRC check. The packet was not corrupted during "
            "transmission."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(113))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kPacketReceived"
        description = (
            "The packet sent from the PC was successfully received, parsed and validated and is ready for payload "
            "decoding."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(114))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kWriteObjectBufferError"
        description = (
            "Unable to write the provided object to the message payload buffer. The TransportLayer serializes the "
            "objects (data) to be sent to the PC into a shared bytes buffer. If the provided object is too large to "
            "fit into the available buffer space, this error is triggered."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(115))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kObjectWrittenToBuffer"
        description = (
            "The object (data) to be sent to the PC has been successfully serialized (written) into the message "
            "payload buffer."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(116))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kReadObjectBufferError"
        description = (
            "Unable to read the provided object's data from the received message payload buffer. The received data has "
            "to be deserialized (converted from bytes to the original format) by the TransportLayer class using "
            "provided 'prototypes' or 'containers' to infer the data format. If the container requests more data than "
            "available from the parsed message buffer, this error is triggered."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(117))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kObjectReadFromBuffer"
        description = (
            "The object (data) received from the PC has been successfully deserialized (read) from the parsed message "
            "payload buffer."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(118))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "transport_layer.status_codes.kDelimiterNotFoundError"
        description = (
            "Unable to find the unencoded delimiter byte value at the end of the incoming packet. Since TransportLayer "
            "class carries out basic data validation as it parses the packet from the serial byte stream, it checks "
            "that the packet endswith a delimiter. If this expectation is violated, this error is triggered to "
            "indicate potential data corruption."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(119))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kDelimiterFoundTooEarlyError"
        description = (
            "The delimiter byte value that is expected to be found at the end of the incoming packet is found before "
            "reaching teh end of the packet. This indicates data corruption."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(120))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "transport_layer.status_codes.kPostambleTimeoutError"
        description = (
            "Unable to parse the incoming packet, as packet reception has stalled. If reception starts before all "
            "bytes of the packet are received by the microcontroller, the parser will wait a reasonable amount of "
            "time to receive the missing bytes. If these bytes do not arrive in time, this error is triggered. This "
            "error specifically applies to parsing the CRC postamble of the packet."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(121))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @classmethod
    def _write_communication_status_codes(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(151))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kReceptionError"
        description = "Communication class ran into an error when attempting to receive a message from the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(152))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kParsingError"
        description = "Communication class ran into an error when parsing (decoding) a message received from the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(153))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kPackingError"
        description = (
            "Communication class ran into an error when writing (serializing) the message data into the transmission "
            "payload buffer."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(154))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kTransmissionError"
        description = "Communication class ran into an error when transmitting (sending) a message to the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(155))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kMessageSent"
        description = "Communication class successfully transmitted a message to the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(156))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kMessageReceived"
        description = "Communication class successfully received a message from the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(157))
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(158))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kNoBytesToReceive"
        description = (
            "Communication class did not receive enough bytes to process the message. This is not an error, most "
            "higher-end microcontrollers will spend a sizeable chunk of their runtime with no communication data to "
            "process."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(159))
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(160))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kParametersExtracted"
        description = (
            "Module parameter data has been successfully extracted and written into the module's parameter structure."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(161))
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
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=np.uint8(162))
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    @staticmethod
    def runtime_cycle(
        interface: "MicroControllerInterface",
    ) -> None:
        # Initializes the communication class. It is critical that this is done inside the method running in an
        # isolated process.
        communication = SerialCommunication(
            usb_port=interface._usb_port,
            baudrate=interface._baudrate,
            maximum_transmitted_payload_size=interface._maximum_transmitted_payload_size,
        )
        interface._terminator_array.connect()

        while True:

            try:
                while True:
                    out_data: (
                        RepeatedModuleCommand
                        | OneOffModuleCommand
                        | DequeueModuleCommand
                        | KernelCommand
                        | ModuleParameters
                        | KernelParameters
                    ) = interface._transmission_queue.get_nowait()

                    # TODO send packed data to the logger

                    communication.send_message(out_data)

            except Empty:
                pass

            # Attempts to receive the data from the microcontroller
            in_data = communication.receive_message()
            while in_data is not None:

                # TODO Send the input data payload to logger

                # Resolve additional processing steps associated with incoming data
                if isinstance(in_data, KernelState):
                    pass
                elif isinstance(in_data, KernelData):
                    pass
                elif isinstance(in_data, ModuleState):
                    pass
                elif isinstance(in_data, ModuleData):
                    pass
                elif isinstance(in_data, ReceptionCode):
                    pass
                elif isinstance(in_data, Identification):
                    pass

            # If the first element of the terminator array is true (>0), ends the communication loop.
            if interface._terminator_array.read_data(index=0, convert_output=True):
                break  # Terminates the loop if instructed to do so

    def identify_controller(self):
        self._transmission_queue.put(self._identify_command)

    def reset_controller(self):
        self._transmission_queue.put(self._reset_command)
