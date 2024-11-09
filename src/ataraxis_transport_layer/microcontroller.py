from abc import abstractmethod
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from ataraxis_base_utilities import console
from ataraxis_data_structures import NestedDictionary, SharedMemoryArray

from .communication import (
    ModuleData,
    ModuleState,
    KernelCommand,
    KernelParameters,
    ModuleParameters,
    UnityCommunication,
    OneOffModuleCommand,
    SerialCommunication,
    DequeueModuleCommand,
    RepeatedModuleCommand,
    prototypes, protocols,
)


class ModuleInterface:
    """The base class from which all custom ModuleInterface classes should inherit.

    Interface classes encapsulate module-specific parameters and data handling methods which are used by the
    MicroControllerInterface class to communicate with individual hardware modules. Overall, this arrangement is similar
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
        using 'super().__init__()'! If this is not done, the MicroControllerInterface class will likely not be able to
        properly interact with your ModuleInterface!

        All data received from or sent to the microcontroller is automatically logged as a series of byte-serialized
        numpy arrays. Do not add any additional processing flags unless you have a good reason to do so.

    Args:
        type_name: The name of the Module type (family) managed by this interface, 'e.g.: Rotary_Encoder'.
        module_type: The byte id-code of the type (family) of Modules managed by this interface. This has to match the
            code used by the module implementation in AXMC. Note, valid byte-codes range from 1 to 255.
        module_id: The instance byte-code ID of the module. This is used to identify unique instances of the same
            module type, such as different rotary encoders if more than one is used concurrently. Note, valid
            byte-codes range from 1 to 255.
        instance_description: Additional notes or description of the module. This can be used to provide further information
            about the interface module, such as the composition of its hardware or the location within the broader
            experimental system. These notes will be treated as instance-specific (unique given the module_type x
            module_id combination)!
        unity_output: Determines whether to send received Module data to Unity. If this flag is disabled, the
            send_to_unity class method will not be called.
        get_from_unity: Determines whether to receive commands from Unity to send to the Module. If this flag is
            disabled, the send_to_unity class method will not be called.
        queue_output: Determines whether to send received Module data back to central process via the output queue of
            the managing MicroControllerInterface class. If this flag is disabled, the send_to_queue class method
            will not be called.

    Attributes:
        _module_type: Stores the type (family) of the interfaced module.
        _module_id: Stores specific id of the interfaced module within the broader type (family).
        _type_name: Stores a string-name of the module_type code. This is used to make the Module type identifiable to
            humans.
        _module_notes: Stores additional notes about the module.
        _unity_output: Determines whether to send data to Unity.
        _unity_input: Determines whether to receive commands from Unity.
        _queue_output: Determines whether to send data to the output queue.

    Raises:
        ValueError: if the input module_type and module_id codes are outside the valid range of values.

    """

    def __init__(
            self,
            type_name: str,
            type_description: str,
            module_type: np.uint8,
            module_id: np.uint8,
            instance_description: str | None = None,
            *,
            unity_output: bool = False,
            unity_input: bool = False,
            queue_output: bool = False,
    ) -> None:

        # Verifies input byte-codes for validity.
        if module_type < 1:
            message = (
                f"Invalid 'module_type' argument value {module_type} encountered when initializing the ModuleInterface "
                f"class for {type_name} Modules. Valid type-codes range from 1 to 255."
            )
            console.error(message=message, error=ValueError)
        if module_id < 1:
            message = (
                f"Invalid 'module_id' argument value {module_id} encountered when initializing the ModuleInterface "
                f"class for {type_name} Modules. Valid id-codes range from 1 to 255."
            )
            console.error(message=message, error=ValueError)

        # Transfers arguments to class attributes.
        self._module_type: np.uint8 = module_type
        self._module_id: np.uint8 = module_id
        self._type_name: str = type_name
        self._module_notes: str = "" if instance_description is None else instance_description
        self._type_notes: str = type_description
        self._unity_output: bool = unity_output
        self._unity_input: bool = unity_input
        self._queue_output: bool = queue_output

    @abstractmethod
    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        """Checks the input message data and, if necessary, sends a message to Unity Game Engine.

        Unity is used to dynamically control the Virtual Reality (VR) environment used in scientific experiments.
        Currently, the communication with Unity is handled via the MQTT protocol and this method is used to
        conditionally transfer the data received from the Module class running on the microcontroller to Unity.

        Notes:
            This method should contain a series of if-else statements that determine whether the incoming message
            should be transferred to Unity. If so, this method should call specific methods of the UnityCommunication
            class to transmit the message data to Unity.

            The arguments to this method will be provided by the managing MicroControllerInterface class. Therefore, you
            should expect the UnityCommunication to be initialized and connected and the message to be properly parsed.

            If the module does not need this functionality, implement the method by calling an empty return statement.
            Remember to enable the 'additional_data_processing' flag when initializing the interface class, if the
            module does need this functionality.

        Args:
            message: The ModuleState or ModuleData object that stores the parsed message data.
            unity_communication: An initialized and connected instance of the UnityCommunication class.
        """
        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"send_to_unity method for {self._type_name} Module must be implemented when subclassing the base "
            f"ModuleInterface."
        )

    @abstractmethod
    def get_from_unity(
            self, unity_communication: UnityCommunication
    ) -> OneOffModuleCommand | RepeatedModuleCommand | None:
        """Checks whether Unity Game Engine has requested a message to be sent to the microcontroller and, if so,
        returns the packaged message structure to send.

        Unity can issue some commands as it resolves the game logic of the managed Virtual Reality task. The initialized
        UnityCommunication will monitor the MQTT (communication protocol) traffic and process incoming Unity-sent
        messages in a background thread and set certain class flags as necessary to reflect incoming command data. This
        class can be used to convert Unity messages to appropriate module-addressed command structures that will be
        sent to the microcontroller by the managing MicroControllerInterface class.

        Notes:
            This method should contain a series of if-else statements that evaluate UnityCommunication properties and,
            based on their values, decide whether to send a message to the microcontroller. If a message needs to be
            sent, the method should package the message data into the appropriate structure and return it to caller.
            Otherwise, the method should return None to indicate that no message needs to be sent.

            The arguments to this method will be provided by the managing MicroControllerInterface class. Therefore, you
            should expect the UnityCommunication to be initialized and connected and the message to be properly parsed.

            If the module does not need this functionality, implement the method by calling an empty return.
            Remember to enable the 'additional_data_processing' flag when initializing the interface class, if the
            module does need this functionality.

        Args:
            unity_communication: An initialized and connected instance of the UnityCommunication class.

        Returns:
            An initialized OneOffModuleCommand or RepeatedModuleCommand class instance that stores the message payload
            to be sent to the microcontroller. None, if there is no message to send.
        """
        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"get_from_unity method for {self._type_name} Module must be implemented when subclassing the base "
            f"ModuleInterface."
        )

    @abstractmethod
    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Checks the input message data and, if necessary, puts the message into the input queue.

        This method allows the ModuleInterface class to send received data to the main process that manages the runtime.
        In turn, this allows processing the data in addition to logging it to disk. For example, the data received from
        the module can be used to generate a live data plot that allows the user to monitor microcontroller runtime.


        Notes:
            This method should contain a series of if-else statements that determine whether the incoming message
            should be put into the queue and, if necessary, do it.

            If the module does not need this functionality, implement the method by calling an empty return.
            Remember to enable the 'additional_data_processing' flag when initializing the interface class, if the
            module does need this functionality.

        Args:
            message: The ModuleState or ModuleData object that stores the parsed message data.
            queue: An instance of the multiprocessing Queue class that allows piping data to the main process.
        """
        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"send_to_unity method for {self._type_name} Module must be implemented when subclassing the base "
            f"ModuleInterface."
        )

    @abstractmethod
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        """Writes custom module status, command, and object data information to the input code_map dictionary.

        This method is called by the MicroControllerInterface that manages the ModuleInterface to fill the shared
        code_map dictionary with module-specific data. This maps number-codes used in serialized communication to
        map commands, events, and additional data objects to human-readable names and descriptions. In turn, this
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
        """Returns the interfaced module's type (family) byte-code."""
        return self._module_type

    @property
    def type_name(self) -> str:
        """Returns the interfaced module's type (family) human-readable name."""
        return self._type_name

    @property
    def module_id(self) -> np.uint8:
        """Returns the interfaced module's ID byte-code (instance-specific identifier code)."""
        return self._module_id

    @property
    def module_notes(self) -> str:
        """Returns additional notes for the specific interfaced module instance, which are unique for each module_id
        and module_type combination.
        """
        return self._module_notes

    @property
    def type_description(self) -> str:
        return self._type_notes

    @property
    def unity_output(self) -> bool:
        """Returns True if the class is configured to send received Module data to Unity."""
        return self._unity_output

    @property
    def unity_input(self) -> bool:
        """Returns True if the class is configured to receive Module commands from Unity."""
        return self._unity_input

    @property
    def queue_output(self) -> bool:
        """Returns True if the class is configured to send received Module data to the output queue."""
        return self._queue_output


class MicroControllerInterface:
    """Exposes methods for continuously communicating with the connected Ataraxis MicroController.

    This class contains the logic that sets up a remote daemon process with SerialCommunication, UnityCommunication,
    and DataLogger bindings to facilitate bidirectional communication between Unity, Python, and the Microcontroller.
    Additionally, it exposes methods for submitting parameters and command to be sent to the Kernel and specific
    Modules of the target Microcontroller.

    Notes:
        An instance of this class has to be instantiated for each concurrently operated Microcontroller. Moreover, since
        the communication process runs on a separate core, the start() and shutdown() methods of the class have to be
        used to enable or disable communication after class initialization.

        This class uses SharedMemoryArray to control the runtime of the remote process, which makes it impossible to
        have more than one instance of this class with the same controller_name at a time. Make sure the class instance
        is deleted (to free SharedMemory buffer) before attempting to initialize a new class instance.

        This class also exposes methods used to build the shared code_map_dictionary. These methods are designed to be
        used together with similar methods from other Ataraxis libraries (notably: video-system) to build a map used for
        deserializing and interpreting logged data. It is imperative that the generated dictionary is accurate for your
        specific runtime, otherwise interpreting logged data may be challenging or impossible. It is highly advised to
        use the CodeMapDictionary class from this library to build the shared dictionary to ensure its correctness.

    Args:
        controller_name: The name of the managed Microcontroller. This name has to be unique for all concurrently active
            microcontrollers.
        controller_id: The unique identifier code of the microcontroller. This code is hardcoded via the firmware
            running on the microcontroller and should match the value provided to this class to avoid errors. Note!
            This code is also used as the source_id for the data sent from this class to the DataLogger. Therefore, it
            is important for this code to be unique across ALL concurrently active Ataraxis data producers
            (microcontrollers, video systems, etc.).
        usb_port: The serial port to which the Microcontroller is connected. You can use list_available_ports() global
            function to discover addressable usb ports.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class (via 'input_queue' property).
            This queue is used to buffer and pipe data to be logged to the logger cores.
        modules: A tuple of classes that inherit from the (base) ModuleInterface class. These classes will be used by
            the main runtime cycle to handle the incoming data from the modules running on the microcontroller.
        controller_notes: Additional description or notes about the connected microcontroller. These notes have
            to be specific to the controller_id code.
        baudrate: The baud rate at which the serial communication should be established. Note, this argument is ignored
            for boards that use the USB communication protocol, such as most Teensy boards.
        maximum_transmitted_payload_size: Should match the microcontroller serial reception buffer size. This is
            used to ensure that transmitted messages will fit inside the reception buffer of teh board. If the size is
            not set right, you may run into communication errors.
    """

    def __init__(
            self,
            controller_name: str,
            controller_id: np.uint8,
            usb_port: str,
            logger_queue: MPQueue,
            modules: tuple[ModuleInterface, ...],
            controller_notes: str | None = None,
            baudrate: int = 115200,
            maximum_transmitted_payload_size: int = 254,
    ):
        # Saves input arguments to class attributes. Mostly, this information will be used when the class starts the
        # communication cycle on a separate core.
        self._name: str = controller_name
        self._controller_id: np.uint8 = controller_id
        self._controller_notes: str = controller_notes
        self._modules = modules
        self._usb_port: str = usb_port
        self._baudrate: int = baudrate
        self._maximum_transmitted_payload_size: int = maximum_transmitted_payload_size
        self._logger_queue: MPQueue = logger_queue  # type: ignore

        # Sets up the assets used to deploy the communication runtime on a separate core and bidirectionally transfer
        # data between the communication process and the main process managing the overall runtime.
        self._mp_manager: SyncManager = Manager()
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._output_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._terminator_array: None | SharedMemoryArray = None
        self._communication_process: None | Process = None

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

    def start(self):
        # Instantiates the array used to control the runtime of the communication Process.
        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator_array",  # Uses class name to ensure the array buffer name is unique
            prototype=np.zeros(shape=1, dtype=np.uint8),
        )  # Instantiation automatically connects the main process to the array.

        # Sets up the communication process. This process continuously cycles through the communication loop until
        # terminated, enabling bidirectional communication with the controller.
        self._communication_process = Process(
            target=self._runtime_cycle,
            args=(
                self._input_queue,
                self._usb_port,
                self._baudrate,
                self._maximum_transmitted_payload_size,
                self._terminator_array,
            ),
            daemon=True,
        )
        self._communication_process.start()

        # elif isinstance(in_data, Identification):
        # if in_data.controller_id == controller_id:
        #     message = (
        #         f"Unexpected controller_id code received from the microcontroller managed by the "
        #         f"{controller_name} MicroControllerInterface instance. Expected {controller_id}, but "
        #         f"received {in_data.controller_id}."
        #     )
        #     console.error(message=message, error=ValueError)

    def stop(self):
        pass

    def parse_module_data(self):

        # Loops over all modules and builds a nested type-id-code dictionary for each used module. This information is
        # used to ensure all interface instances are uniquely identifiable.
        module_type_map = {}
        for module in self._modules:
            module_id = module.module_id
            module_type = module.module_type
            if module_type_map[module_type][module_id] is not None:
                message = (
                    f"Unable to initialize the MicroControllerInterface instance for {self._name} microcontroller with "
                    f"id code of {self._controller_id}. Encountered two ModuleInterface instance with the same "
                    f"type-code {module_type} and id-code {module_id}, which is not allowed. Make sure that each "
                    f"type+id combination is only used by a single ModuleInterface."
                )
                console.error(message=message, error=ValueError)
            else:
                # Fills the dictionary with type-id-module_name sequences.
                module_type_map[module_type][module_id] = module_type

    def identify_controller(self) -> None:
        """Sends the Identification command to the connected Microcontroller's kernel class."""
        self._input_queue.put(self._identify_command)

    def reset_controller(self) -> None:
        """Sends the reset command to the connected Microcontroller's kernel class."""
        self._input_queue.put(self._reset_command)

    def build_core_code_map(self) -> NestedDictionary:
        # Pre-initializes with a seed dictionary that includes the purpose (description) of the dictionary file
        message = (
            "This dictionary maps status, commands and data object byte-values used by the Core classes that manage "
            "microcontroller runtime to meaningful names and provides a human-friendly description for each byte-code. "
            "This information is used by parsers when decoding logged communication data."
        )
        code_dictionary = NestedDictionary(seed_dictionary={"description": message})

        # Kernel: status codes
        code_dictionary = self._write_kernel_status_codes(code_dictionary)

        # Kernel: command codes
        code_dictionary = self._write_kernel_command_codes(code_dictionary)

        # Kernel: object data
        code_dictionary = self._write_kernel_object_data(code_dictionary)

        # Module: core status codes.
        # Note, primarily, modules use custom status and command codes for each module family. This data is extracted
        # from managed ModuleInterface classes via the build_module_code_map() method. This section specifically tracks
        # the 'core' codes that each custom module inherits by subclassing the base Module class.
        code_dictionary = self._write_base_module_status_codes(code_dictionary)

        # Module: core object data
        code_dictionary = self._write_base_module_object_data(code_dictionary)

        # Communication: status codes
        # This is the final class of the 'core' triad. This class writes protocols and prototypes sections in addition
        # to the status_codes section. This information is used during data transmission and reception to decode various
        # incoming and outgoing message payloads.
        code_dictionary = self._write_communication_status_codes(code_dictionary)
        code_dictionary = prototypes.write_prototype_codes(code_dictionary)
        code_dictionary = protocols.write_protocol_codes(code_dictionary)

        # TransportLayer: status codes
        # This and the following sections track codes from classes wrapped by the Communication class. Due to the
        # importance of the communication library, we track all status codes that are (theoretically) relevant for
        # communication. TransportLayer is the source of all codes added in this way (it 'passes' CRC and COBS codes
        # as its own).
        code_dictionary = self._write_transport_layer_status_codes(code_dictionary)
        code_dictionary = self._write_cobs_status_codes(code_dictionary)
        code_dictionary = self._write_crc_status_codes(code_dictionary)

        return code_dictionary

    def build_custom_code_map(self) -> NestedDictionary:

        # Pre-initializes with a seed dictionary that includes the purpose (description) of the dictionary file
        message = (
            "This dictionary maps status, commands and data object byte-values used by custom user-defined modules "
            "that interface with specific hardware to meaningful names and provides a human-friendly description for "
            "each byte-code. This information is used by parsers when decoding logged communication data."
        )
        code_dictionary = NestedDictionary(seed_dictionary={"description": message})

        # Extracts and appends the information that is expected to be the same for each module type.
        processed_types = set()  # Ensures each module type is only mapped once
        for module in self._modules:

            # If this module type has already been mapped, skips to the next module
            if module.module_type in processed_types:
                continue

            # Adds each new module type to the processed types set
            processed_types.add(processed_types)

            # Calls the class method that should fill the status_codes, commands and data_objects module-type-specific
            # sections of the dictionary with data and return it back to caller.
            code_dictionary = module.write_code_map(code_dictionary)

        return code_dictionary

    def build_id_code_map(self) -> NestedDictionary:
        # Pre-initializes with a seed dictionary that includes the purpose (description) of the dictionary file
        message = (
            "This dictionary maps status, commands and data object byte-values used by custom user-defined modules "
            "that interface with specific hardware to meaningful names and provides a human-friendly description for "
            "each byte-code. This information is used by parsers when decoding logged communication data."
        )
        code_dictionary = NestedDictionary(seed_dictionary={"description": message})

        code_dictionary.write_nested_value(variable_path=f"{self._name}.code", value=self._controller_id)
        code_dictionary.write_nested_value(variable_path=f"{self._name}.description", value=self._controller_notes)

        # Extracts and appends the information that is expected to be the same for each module type.
        processed_types = set()  # Ensures each module type is only mapped once
        for module in self._modules:

            if module.module_type not in processed_types:
                # Adds each new module type to the processed types set
                processed_types.add(processed_types)
                section = f"{module.type_name}_module.code"
                code_dictionary.write_nested_value(variable_path=section, value=module.module_type)
                section = f"{module.type_name}_module.description"
                code_dictionary.write_nested_value(variable_path=section, value=module.type_description)

                section = f"{module.type_name}_module.instances"
                code_dictionary.write_nested_value(variable_path=section, value=[module.module_id])
                section = f"{module.type_name}_module.descriptions"
                code_dictionary.write_nested_value(variable_path=section, value=[module.module_notes])

            section = f"{module.type_name}_module.instances"
            existing_values: list[np.uint8] = code_dictionary.read_nested_value(variable_path=section)
            code_dictionary.write_nested_value(variable_path=section, value=existing_values.append(module.module_id))

            section = f"{module.type_name}_module.descriptions"
            existing_values: list[str] = code_dictionary.read_nested_value(variable_path=section)
            code_dictionary.write_nested_value(variable_path=section, value=existing_values.append(module.module_notes))

        return code_dictionary

    @staticmethod
    def _runtime_cycle(
            controller_id: np.uint8,
            input_queue: MPQueue,
            logger_queue: MPQueue,
            terminator_array: SharedMemoryArray,
            usb_port: str,
            baudrate: int,
            unity_ip: str,
            unity_port: int,
            payload_size: int,
            active_modules: tuple[ModuleInterface, ...],
            output_queue: None | MPQueue = None,
    ) -> None:

        # Sorts active module into further categories, depending on whether they are configured to receive commands from
        # Unity, send data to Unity ro send data to the output queue
        additional_input_indices = []
        additional_processing_indices = []
        unity_out = False
        unity_in = False
        queue_out = False
        for module in enumerate(active_modules):
            if module.unity_input:
                unity_in = True
                additional_input_indices.append(num)
            if module.unity_output:
                unity_out = True
                if num not in additional_processing_indices:
                    additional_processing_indices.append(num)
            if module.queue_output:
                queue_out = True
                if num not in additional_processing_indices:
                    additional_processing_indices.append(num)
        additional_input_indices = tuple(additional_input_indices)

        # Initializes the communication class and connects to the managed MicroController.
        serial_communication = SerialCommunication(
            usb_port=usb_port,
            source_id=int(controller_id),
            logger_queue=logger_queue,
            baudrate=baudrate,
            maximum_transmitted_payload_size=payload_size,
        )

        # Initializes the unity_communication class and connects to the MQTT broker, if the class is configured to
        # communicate with Unity.
        if unity_in or unity_out:
            unity_communication = UnityCommunication(
                ip=unity_ip,
                port=unity_port,
                lick_topic=True if unity_out else False,
                position_topic=True if unity_out else False,
                reward_topic=True if unity_in else False,
            )
            unity_communication.connect()
        else:
            unity_communication = None

        # Connects to the terminator array
        terminator_array.connect()

        # Initializes the main communication loop. This loop will run until the exit conditions are encountered.
        # The exit conditions for the loop require the first variable in the terminator_array to be set to True
        # and the main input queue of the interface to be empty. This ensures that all queued commands issued from
        # the central process are fully carried out before the communication is terminated.
        while terminator_array.read_data(index=0, convert_output=True) and input_queue.empty():

            # Main data sending loop. The method will sequentially retrieve the queued command and parameter data to be
            # sent to the Microcontroller and send it.
            while not input_queue.empty():
                out_data: (
                        RepeatedModuleCommand
                        | OneOffModuleCommand
                        | DequeueModuleCommand
                        | KernelCommand
                        | ModuleParameters
                        | KernelParameters
                ) = input_queue.get_nowait()
                serial_communication.send_message(out_data)  # Transmits the data to the microcontroller

            # Unity data sending loop. The loop will keep cycling until the class runs out of stored data to send to the
            # microcontroller.
            while unity_communication.has_data:

                # If UnityCommunication has received data, loops over all interfaces that were flagged as
                # capable of processing data and calls their unity data acquisition methods. The methods are expected to
                # extract the data from the communication class and translate it into a valid message format to be sent
                # to the microcontroller.
                for i in additional_input_indices:
                    out_data = modules[i].get_from_unity(unity_communication=unity_communication)

                    # It is expected that the method will return None if no message needs to be sent to the
                    # microcontroller and a valid message structure otherwise
                    if out_data is None:
                        continue

                    # If all data stored in the communication class has been consumed, stops the for loop early.
                    # Otherwise, keeps looping over modules in case not-yet-evaluated modules in the sequence can
                    # process the remaining data. This maximizes the runtime efficiency of this code section by
                    # avoiding unnecessary module cycling.
                    if not unity_communication.has_data:
                        break

            # Receives data from microcontroller
            in_data = serial_communication.receive_message()

            if in_data is None:
                continue
            # Resolve additional processing steps associated with incoming data
            if isinstance(in_data, (ModuleState, ModuleData)):

                if not unity_out and not queue_out:
                    continue

                for module in active_modules:
                    if module.module_type != in_data.module_type or module.module_id != in_data.module_id:
                        continue

                    if module.unity_output:
                        module.send_to_unity(message=in_data, unity_communication=unity_communication)
                    if module.queue_output:
                        module.send_to_queue(message=in_data, queue=output_queue)
                    break

        # Disconnects from the terminator array and terminates the process.
        terminator_array.disconnect()
        unity_communication.disconnect()

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
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(2))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.names", value=("module_type", "module_id"))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "kernel.data_objects.kDataReceptionErrorObject"
        description_1 = "The status-code of the Communication class instance."
        description_2 = "The status-code of the TransportLayer class instance or a COBS /CRC error code."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(3))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.names", value=("communication_status", "transport_layer_status")
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "kernel.data_objects.kDataSendingErrorObject"
        description_1 = "The status-code of the Communication class instance."
        description_2 = "The status-code of the TransportLayer class instance or a COBS /CRC error code."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(4))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.names", value=("communication_status", "transport_layer_status")
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "kernel.data_objects.kStateSendingErrorObject"
        description_1 = "The status-code of the Communication class instance."
        description_2 = "The status-code of the TransportLayer class instance or a COBS /CRC error code."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(5))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.names", value=("communication_status", "transport_layer_status")
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "kernel.data_objects.kServiceSendingErrorObject"
        description_1 = "The status-code of the Communication class instance."
        description_2 = "The status-code of the TransportLayer class instance or a COBS /CRC error code."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(6))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.names", value=("communication_status", "transport_layer_status")
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "kernel.data_objects.kInvalidMessageProtocolObject"
        description_1 = "The invalid protocol byte-code value that was received by the Kernel."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(7))
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
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(10))
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
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(12))
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
    def _write_base_module_object_data(cls, code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the module.object_data section of the core_codes_map dictionary with data.

        This section is used to provide additional information about the values of the data objects used by ModuleData
        messages transmitted by the method inherited from the (base) Module class. These objects usually have a
        different interpretation depending on the event-code of the message they are sent with.

        Args:
            code_dictionary: The dictionary to be filled with module object data.

        Returns:
            The updated dictionary with module object data information filled.
        """
        section = "module.data_objects.kDataSendingErrorObject"
        description_1 = "The status-code of the Communication class instance."
        description_2 = "The status-code of the TransportLayer class instance or a COBS /CRC error code."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(1))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.names", value=("communication_status", "transport_layer_status")
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

        section = "module.data_objects.kStateSendingErrorObject"
        description_1 = "The status-code of the Communication class instance."
        description_2 = "The status-code of the TransportLayer class instance or a COBS /CRC error code."
        code_dictionary.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(2))
        code_dictionary.write_nested_value(
            variable_path=f"{section}.prototype_code", value=prototypes.kTwoUnsignedBytes
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.names", value=("communication_status", "transport_layer_status")
        )
        code_dictionary.write_nested_value(
            variable_path=f"{section}.descriptions", value=(description_1, description_2)
        )

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

    def _vacate_shared_memory_buffer(self) -> None:
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed if the class is used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer. In turn, this would prevent the
        class remote Process from being started again. This method allows manually removing that buffer to reset the
        system.
        """
        buffer = SharedMemory(name=f"{self._name}_terminator_array", create=False)
        buffer.close()
        buffer.unlink()
