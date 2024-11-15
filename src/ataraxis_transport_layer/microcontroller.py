import sys
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
    Identification,
    KernelParameters,
    ModuleParameters,
    UnityCommunication,
    OneOffModuleCommand,
    SerialCommunication,
    DequeueModuleCommand,
    RepeatedModuleCommand,
    protocols,
    prototypes,
)


class ModuleInterface:
    """The base class from which all custom ModuleInterface classes should inherit.

    ModuleInterface classes encapsulate module-specific parameters and data handling methods which are used by the
    MicroControllerInterface class to communicate with individual hardware-controlling modules running on the
    microcontroller. Overall, this arrangement is similar to how custom modules inherit from the (base) Module class in
    the AtaraxisMicroController (AXMC) library.

    Due to a high degree of custom module variability, it is currently not possible to provide a 'one-fits-all' Module
    interface that is also highly efficient for real time communication. Therefore, similar to AXMC library, the
    interface for each custom module has to be implemented separately on a need-base method. The (base) class exposes
    the static API that MicroControllerInterface class can use to integrate each custom interface implementation with
    the general communication runtime cycle. To make this integration possible, this class declares a number of
    abstract (pure virtual) methods that developers have to implement for their interfaces. Follow the implementation
    guidelines in the docstrings of each abstract method and check the default modules included with the library
    distribution for guidance.

    Notes:
        When inheriting from this class, remember to call the parent's init method in the child class init method by
        using 'super().__init__()'! If this is not done, the MicroControllerInterface class will likely not be able to
        properly interact with your ModuleInterface!

        All data received from or sent to the microcontroller is automatically logged as byte-serialized numpy arrays.
        Therefore, if you do not need any other processing steps, such as sending to or receiving data from Unity,
        do not enable any custom processing flags. You will, however, have to implement all abstract methods, even if
        the class instance does not use them due to its flag-configuration.

    Args:
        module_type: The byte id-code of the type (family) of Modules managed by this interface. This has to match the
            code used by the module implementation in AXMC. Note, valid byte-codes range from 1 to 255.
        type_name: The name of the Module type (family) managed by this interface, 'e.g.: Rotary_Encoder'. This name is
            mostly used to better identify the module type to humans.
        type_description: The description of the module type. This information is saved with other runtime ID
            information and is primarily intended for human operators that will pwork with collected runtime data.
            This description can be set to the same string when using multiple instances of the same type.
        module_id: The instance byte-code ID of the module. This is used to identify unique instances of the same
            module type, such as different rotary encoders if more than one is used concurrently. Note, valid
            byte-codes range from 1 to 255.
        instance_name: The name of the specific module instance, e.g.: 'Left_Corner_Touch_Sensor'. These names are used
            to better identify different type instances to human operators that will work with the collected runtime
            data.
        instance_description: Additional description of the module instance. This can be used to provide further
            instance information, such as the composition of its hardware or the location within the broader
            experimental setup.
        unity_input_topics: A list of MQTT topics to which this module should subscribe to receive commands from Unity.
            If the module should not receive commands from Unity, set to None. This list will be used to initialize the
            UnityCommunication class instance to listen to the requested topics. If the list is provided, it is
            expected that get_from_unity() method implements the logic for accessing and handling the incoming commands.
        unity_output: Determines whether to send received data to Unity via the MQTT protocol. If this flag is True, it
            is expected that send_to_unity() method implements the logic for sending the necessary data to Unity.
        queue_output: Determines whether to send received data to other processes. If this flag is True, it is expected
            that send_to_queue() method implements the logic for sending the necessary data to the multiprocessing queue
            which pipes it to other processes.

    Attributes:
        _module_type: Stores the type (family) of the interfaced module.
        _type_name: Stores a name of the module type (family).
        _type_description: Stores the description of the module type (family).
        _module_id: Stores specific id of the interfaced module within the broader type (family).
        _instance_name: Stores the name of the specific module instance.
        _instance_description: Stores the description of the specific module instance.
        _type_id: Stores the type and id combined into a single uint16 value. This value should be unique for all
            possible type-id pairs and is used to ensure that each used module instance has a unique ID-type
            combination.
        _unity_output: Determines whether to send data to Unity.
        _unity_input: Determines whether to receive commands from Unity.
        _queue_output: Determines whether to send data to the output queue.
        _unity_input_topics: Stores the list of Unity topics to monitor for incoming commands.

    Raises:
        ValueError: if the input module_type and module_id codes are outside the valid range of values.
    """

    def __init__(
        self,
        module_type: np.uint8,
        type_name: str,
        type_description: str,
        module_id: np.uint8,
        instance_name: str,
        instance_description: str,
        unity_input_topics: tuple[str, ...] | None,
        *,
        unity_output: bool = False,
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

        # Module Type. Should be the same for all instances of this type
        self._module_type: np.uint8 = module_type
        self._type_name: str = type_name
        self._type_description: str = type_description

        # Module Instance. This should be unique for each instance within the same type
        self._module_id: np.uint8 = module_id
        self._instance_name: str = instance_name
        self._instance_description: str = instance_description

        # Combines type and ID into a 16-bit value. This is used to ensure every module instance has a unique
        # ID + Type combination. This method is position-aware, which avoids the issue of reverse pairs giving the same
        # resultant value(e.g.: 4-5 != 5-4)
        self._type_id: np.uint16 = np.uint16(
            (self._module_type.astype(np.uint16) << 8) | self._module_id.astype(np.uint16)
        )

        # Additional processing flags. Unity input is set based on whether there are input / output topics
        self._unity_input_topics: tuple[str, ...] = unity_input_topics if unity_input_topics is not None else tuple()
        self._unity_input: bool = True if len(self._unity_input_topics) > 0 else False
        self._unity_output: bool = unity_output
        self._queue_output: bool = queue_output

    def __repr__(self) -> str:
        """Returns the string representation of the ModuleInterface instance."""
        message = (
            f"ModuleInterface(type_code={self._module_type}, type_name={self._type_name}, id_code={self._module_id}, "
            f"instance_name={self._instance_name}, unity_output={self._unity_output}, unity_input={self._unity_input}, "
            f"queue_output={self._queue_output}, unity_input_topics={self._unity_input_topics})"
        )
        return message

    @abstractmethod
    def get_from_unity(
        self, topic: str, payload: bytes | bytearray
    ) -> OneOffModuleCommand | RepeatedModuleCommand | None:
        """Packages and returns a command message structure to send to the microcontroller, based on the input Unity
        message topic and payload.

        Unity can issue module commands as it resolves the game logic of the Virtual Reality (VR) task. The initialized
        UnityCommunication class will monitor the MQTT (communication protocol) traffic and process incoming Unity-sent
        messages in a background thread. The MicroControllerInterface will then read the data received from Unity and
        pass it to all ModuleInterface instances that declared the MQTT topic at which the message was received as an
        input topic. Therefore, this method is ONLY called when the topic to which the Unity sent command data exactly
        matches the topic(s) specified at ModuleInterface instantiation.

        Notes:
            This method should resolve, package, and return the appropriate ModuleCommand message structure, based on
            the input Unity topic and payload. Since this method is only called for topics declared by the class
            instance as input topics, it is expected that the method returns a valid command to send every time it is
            called.

            Remember to provide the class with topics to listen to via the 'unity_input_topics' argument when
            initializing the interface class if the instance does need this functionality. If the instance does not
            need this functionality, implement the method by calling an empty return statement and ensure that the
            'unity_input_topics' argument is set to None.

        Args:
            topic: The MQTT topic to which the Unity message was sent.
            payload: The message payload received from Unity.

        Returns:
            An initialized OneOffModuleCommand or RepeatedModuleCommand class instance that stores the message payload
            to be sent to the microcontroller.
        """
        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"get_from_unity method for {self._type_name} module interface must be implemented when subclassing the "
            f"base ModuleInterface class."
        )

    @abstractmethod
    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        """Checks the input message data and, if necessary, sends a message to Unity game engine.

        Unity is used to dynamically control the Virtual Reality (VR) environment used in scientific experiments.
        Currently, the communication with Unity is handled via the MQTT protocol and this method is used to
        conditionally transfer the data received from the Module running on the microcontroller to Unity.

        Notes:
            This method should contain the logic to determine whether the incoming message should be transferred to
            Unity. If so, this method should call the send_data() method of the input UnityCommunication class and
            send the data to the appropriate MQTT topic.

            The arguments to this method will be provided by the managing MicroControllerInterface class and, therefore,
            the UnityCommunication would be connected and appropriately configured to carry out the communication.

            Remember to enable the 'unity_output' flag when initializing the interface class if the instance does need
            this functionality. If the instance does not need this functionality, implement the method by calling an
            empty return statement and ensure that the 'unity_output' flag is disabled.

        Args:
            message: The ModuleState or ModuleData object that stores the parsed message received from the
                microcontroller.
            unity_communication: An initialized and connected instance of the UnityCommunication class to use for
                sending the data to Unity.
        """

        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"send_to_unity method for {self._type_name} module interface must be implemented when subclassing the "
            f"base ModuleInterface class."
        )

    @abstractmethod
    def send_to_queue(
        self,
        message: ModuleData | ModuleState,
        queue: MPQueue,  # type: ignore
    ) -> None:
        """Checks the input message data and, if necessary, sends a message to other processes via the provided
        multiprocessing Queue instance.

        This method allows sending received data to other processes, running in-parallel with the microcontroller
        communication process. In turn, this allows the data to be processed online, in addition to being logged to
        disk. For example, the data received from the module can be used to generate a live data plot for the user to
        monitor microcontroller runtime.

        Notes:
            This method should contain the logic to determine whether the incoming message should be transferred to
            other processes. If so, this method should call the put() method of the input queue object to pipe the
            data to the shared multiprocessing queue.

            Remember to enable the 'queue_output' flag when initializing the interface class if the instance does need
            this functionality. If the instance does not need this functionality, implement the method by calling an
            empty return statement and ensure that the 'queue_output' flag is disabled.

        Args:
            message: The ModuleState or ModuleData object that stores the parsed message received from the
                microcontroller.
            queue: An instance of the multiprocessing Queue class that allows piping data to parallel processes.
        """

        # While abstract method should prompt the user to implement this method, the default error-condition is also
        # included for additional safety.
        raise NotImplementedError(
            f"send_to_queue method for {self._type_name} module interface must be implemented when subclassing the "
            f"base ModuleInterface class."
        )

    @abstractmethod
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        """Updates the input code_map dictionary with module-type-specific status_codes, commands, and data_objects
        sections.

        This method is called by the MicroControllerInterface that manages the ModuleInterface to fill the shared
        code_map dictionary with module-specific data. The code-map dictionary is a nested dictionary that maps various
        byte-codes used during serial communication to meaningful names and human-friendly descriptions. In turn, this
        information is used to transform logged data, which is stored as serialized byte-strings, into a format more
        suitable for data analysis and long-term storage. Additionally, the map dictionaries act as a form of runtime
        documentation that should always be included with the runtime-collected data during future data processing.

        Notes:
            This method should return a one-section dictionary with all data found under the modulename_module section.
            So, if the name of the module was RotaryEncoder, all data should be saved under RotaryEncoder_module. For
            this, use a '.'-delimited path which starts with section name, e.g.:
            'RotaryEncoder_module.status_codes.kIdle'.

            This method has to be the same for all interface of the same module type (family), and it is used to store
            information expected to be the same for all instances of the same type. Therefore, this method
            should fill all relevant module-type sections: commands, status_codes, and data_objects. This method will
            only be called once for each unique module_type.

            See MicroControllerInterface class for examples on how to write this method (and fill the code_map
            dictionary). Note, if this method is not implemented properly, it may be challenging to decode the logged
            data in the future.

        """
        raise NotImplementedError(
            f"write_code_map method for {self._type_name} module interface must be implemented when subclassing the "
            f"base ModuleInterface class."
        )

    def make_deque_command(self) -> DequeueModuleCommand:
        """Returns the command that, upon being sent to the microcontroller, will clear all queued commands for this
        module instance.

        Since these Dequeue commands are universal, the method that packages and returns these command messages is
        defined as a non-abstract base ModuleInterface class method.
        """
        return DequeueModuleCommand(module_type=self._module_type, module_id=self._module_id, return_code=np.uint8(0))

    @property
    def module_type(self) -> np.uint8:
        """Returns the byte-code of the interfaced module type (family)."""
        return self._module_type

    @property
    def type_name(self) -> str:
        """Returns the human-readable name of the interfaced module type (family)."""
        return self._type_name

    @property
    def type_description(self) -> str:
        """Returns the human-readable description of the interfaced module type (family)."""
        return self._type_description

    @property
    def module_id(self) -> np.uint8:
        """Returns the byte-code identifier (ID) of the specific interfaced module instance."""
        return self._module_id

    @property
    def instance_name(self) -> str:
        """Returns the human-readable name of the specific interfaced module instance."""
        return self._instance_name

    @property
    def instance_description(self) -> str:
        """Returns the human-readable description of the interfaced module instance."""
        return self._instance_description

    @property
    def unity_output(self) -> bool:
        """Returns True if the class is configured to send the data received from the module instance to Unity."""
        return self._unity_output

    @property
    def unity_input(self) -> bool:
        """Returns True if the class is configured to receive commands from Unity and send them to module instance."""
        return self._unity_input

    @property
    def queue_output(self) -> bool:
        """Returns True if the class is configured to send the data received from the module instance to other
        processes."""
        return self._queue_output

    @property
    def unity_input_topics(self) -> tuple[str, ...]:
        """Returns the tuple of MQTT topics that should be monitored for incoming Unity commands."""
        return self._unity_input_topics

    @property
    def type_id(self) -> np.uint16:
        """Returns the unique unsigned integer value that results from combining the type-code and the id-code of the
        module instance."""
        return self._type_id


class MicroControllerInterface:
    """Exposes methods that enable continuous bidirectional communication between the connected MicroController and
    other concurrently active Ataraxis systems.

    This class contains the logic that sets up a remote daemon process with SerialCommunication, UnityCommunication,
    and DataLogger bindings to facilitate bidirectional communication between Unity, Python, and the Microcontroller.
    Additionally, it exposes methods for submitting parameters and command to be sent to the Kernel and specific
    Modules of the target Microcontroller.

    Notes:
        An instance of this class has to be instantiated for each concurrently operated Microcontroller. Moreover, since
        the communication process runs on a separate core, the start() and stop() methods of the class have to be
        used to enable or disable communication after class initialization.

        This class uses SharedMemoryArray to control the runtime of the remote process, which makes it impossible to
        have more than one instance of this class with the same controller_name at a time. Make sure the class instance
        is stopped (to free SharedMemory buffer) before attempting to initialize a new class instance.

        This class also exposes methods used to build the shared code_map dictionary. These methods are designed to be
        used together with similar methods from other Ataraxis libraries (notably: video-system) to build a map used for
        deserializing and interpreting logged data. It is imperative that the generated dictionary is accurate for your
        specific runtime, otherwise interpreting logged data may be challenging or impossible. See one of our public
        experimental runtimes for an example of how to properly build a code-map dictionary using class methods.

    Args:
        controller_id: The unique identifier code of the managed microcontroller. This information is hardcoded via the
            AtaraxisMicroController (AXMC) firmware running on the microcontroller, and this class ensures that the code
            used by the connected microcontroller matches this argument when the connection is established. Critically,
            this code is also used as the source_id for the data sent from this class to the DataLogger. Therefore, it
            is important for this code to be unique across ALL concurrently active Ataraxis data producers, such as:
            microcontrollers, video systems, and Unity game engine instances.
        controller_name: The human-readable name of the connected microcontroller. This information is used to better
            identify the microcontroller to human operators.
        controller_description: A longer human-readable description of the microcontroller. This provides additional
            information about the microcontroller, such as its general purpose or properties.
        controller_usb_port: The serial USB port to which the microcontroller is connected. This information is used to
            set up the bidirectional serial communication with the controller. You can use list_available_ports()
            function from this library to discover addressable usb ports to pass to this argument.
        logger_queue: The multiprocessing Queue object exposed by the DataLogger class via its 'input_queue' property.
            This queue is used to buffer and pipe data to be logged to the logger cores.
        modules: A tuple of classes that inherit from the (base) ModuleInterface class. These classes will be used by
            the main runtime cycle to handle the incoming data from the modules running on the microcontroller. These
            classes will also be used to build the dictionary that maps various byte-codes used during serial
            communication to human-readable names and descriptions.
        baudrate: The baudrate at which the serial communication should be established. Note, this argument is ignored
            for boards that use the USB communication protocol, such as most Teensy boards. The correct baudrate for
            boards using the UART communication protocol depends on the clock speed of the board and the specific
            UART revision supported by the board. Setting this to an unsupported value for UART boards will result in
            communication errors.
        maximum_transmitted_payload_size: The maximum size of the message payload that can be sent to the
            microcontroller as one message. This should match the microcontroller serial reception buffer size, even if
            the actual transmitted payloads do not reach that size. This is used to ensure that transmitted messages
            will fit inside the reception buffer of the board. If the size is not set right, you may run into
            communication errors.
        unity_broker_ip: The ip address of the MQTT broker used for Unity communication. Typically, this would be a
            'virtual' ip-address of the locally running MQTT broker, but the class can carry out cross-machine
            communication if necessary. Unity communication will only be initialized if any of the input modules require
            this functionality.
        unity_broker_port: The TCP port of the MQTT broker used for Unity communication. THis is used in conjunction
            with the unity_broker_ip argument to connect to the MQTT broker. Unity communication will only be
            initialized if any of the input modules require this functionality.
        verbose: Determines whether the contents of the received and outgoing serial messages are printed to console.
            This option is used during debugging and should be disabled during production runtimes.

        Attributes:
            _controller_id: Stores the id byte-code of the managed microcontroller.
            _controller_name: Stores the human-readable name of the managed microcontroller.
            _controller_description: Stores the longer description of the managed microcontroller.
            _usb_port: Stores the USB port to which the controller is connected.
            _baudrate: Stores the baudrate to use for serial communication with the controller.
            _max_tx_payload_size: Stores the maximum size the transmitted (outgoing) message payload can reach to fit
                inside the reception buffer of the microcontroller's Serial interface.
            _unity_ip: Stores the IP address of the MQTT broker used for Unity communication.
            _unity_port: Stores the port number of the MQTT broker used for Unity communication.
            _mp_manager: Stores the multiprocessing Manager used to initialize and manage input and output Queue
                objects.
            _input_queue: Stores the multiprocessing Queue used to input the data to be sent to the microcontroller into
                the communication process.
            _output_queue: Stores the multiprocessing Queue used to output the data received from the microcontroller to
                other processes.
            _terminator_array: Stores the SharedMemoryArray instance used to control the runtime of the remote
                communication process.
            _communication_process: Stores the (remote) Process instance that runs the communication cycle.
            _controller_map_section: Stores the microcontroller-specific NestedDictionary section that stores byte-code
                and id information for the microcontroller and all modules used by the microcontroller.
            _reset_command: Stores the pre-packaged Kernel-addressed command that resets the microcontroller's hardware
                and software.
            _identify_command: Stores the pre-packaged Kernel-addressed command that requests the microcontroller to
                send back its id-code.
            _disable_locks: Stores the pre-packaged Kernel parameters configuration that disables all pin locks. This
                allows writing to all microcontroller pins.
            _enable_locks: Stores the pre-packaged Kernel parameters configuration that enables all pin locks. This
                prevents every Module managed by the Kernel from writing to any of the microcontroller pins.
            _verbose: Stores the value of the verbose flag.
            _started: Tracks whether the communication process has been started. This is used to prevent calling
                the start() and stop() methods multiple times.
    """

    # Pre-packages Kernel commands into attributes. Since Kernel commands are known and fixed at compilation,
    # they only need to be defined once.
    _reset_command = KernelCommand(
        command=np.uint8(2),
        return_code=np.uint8(0),
    )
    _identify_command = KernelCommand(
        command=np.uint8(3),
        return_code=np.uint8(0),
    )

    # Also pre-packages the two most used parameter configurations (all-locked and all-unlocked). The class can
    # also send messages with partial locks (e.g.: TTl ON, Action OFF), but those are usually not used outside
    # specific debugging and testing scenarios, so are not really worth to pre-package.
    _disable_locks = KernelParameters(
        action_lock=np.bool(False),
        ttl_lock=np.bool(False),
        return_code=np.uint8(0),
    )
    _enable_locks = KernelParameters(
        action_lock=np.bool(True),
        ttl_lock=np.bool(True),
        return_code=np.uint8(0),
    )

    def __init__(
        self,
        controller_id: np.uint8,
        controller_name: str,
        controller_description: str,
        controller_usb_port: str,
        logger_queue: MPQueue,  # type: ignore
        modules: tuple[ModuleInterface, ...],
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
        unity_broker_ip: str = "127.0.0.1",
        unity_broker_port: int = 1883,
        verbose: bool = False,
    ):
        # Controller (kernel) ID information. Follows the same code-name-description format as module type and instance
        # values do.
        self._controller_id: np.uint8 = controller_id
        self._controller_name: str = controller_name
        self._controller_description: str = controller_description

        # SerialCommunication parameters. This is used to initialize the communication in the remote process.
        self._usb_port: str = controller_usb_port
        self._baudrate: int = baudrate
        self._max_tx_payload_size: int = maximum_transmitted_payload_size

        # UnityCommunication parameters. This is used to initialize the unity communication from the remote process
        # if the managed modules need this functionality.
        self._unity_ip: str = unity_broker_ip
        self._unity_port: int = unity_broker_port

        # Managed modules and data logger queue. Modules will be pre-processes as part of this initialization runtime.
        # Logger queue is fed directly into the SerialCommunication, which automatically logs all incoming and outgoing
        # data to disk.
        self._modules = modules
        self._logger_queue: MPQueue = logger_queue  # type: ignore

        # Sets up the assets used to deploy the communication runtime on a separate core and bidirectionally transfer
        # data between the communication process and the main process managing the overall runtime.
        self._mp_manager: SyncManager = Manager()
        self._input_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._output_queue: MPQueue = self._mp_manager.Queue()  # type: ignore
        self._terminator_array: None | SharedMemoryArray = None
        self._communication_process: None | Process = None

        # Stores the value of the verbose flag. This ahs to be done before parsing module data in case the process fails
        # and the class needs to be garbage-collected before initialization is over
        self._verbose: bool = verbose
        self._started: bool = False

        # Extracts information from the input modules and finalizes runtime preparations. As part of this process,
        # the method will create the microcontroller-specific code-map dictionary section.
        self._controller_map_section: NestedDictionary = self._parse_module_data()

    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
        return (
            f"MicroControllerInterface(controller_id={self._controller_id}, controller_name={self._controller_name}, "
            f"usb_port={self._usb_port}, baudrate={self._baudrate}, max_tx_payload_size={self._max_tx_payload_size}, "
            f"unity_ip={self._unity_ip}, unity_port={self._unity_port}, verbose={self._verbose}, "
            f"started={self._started})"
        )

    def __del__(self) -> None:
        """Ensures that all class resources are properly released when the class instance is garbage-collected."""
        self.stop()

    def _parse_module_data(self) -> NestedDictionary:
        """Loops over the input modules and extracts the necessary information to finish class initialization.

        Primarily, this method has two distinct goals. First, it ensures that every ModuleInterface instance contains a
        unique combination of type and instance codes, allowing to reliably identify each instance. Second, it extracts
        microcontroller and module-specific information and uses it to construct the microcontroller-specific code-map
        dictionary section.

        Returns:
            The NestedDictionary class instance that contains the microcontroller-specific information. This dictionary
            can be retrieved by accessing the microcontroller_map_section attribute.
        """

        # Seeds dictionary section with the main section description.
        # It is expected that the method building the overall mega-dictionary that integrates information from all
        # microcontrollers correctly extracts and combines each microcontroller-specific section under its
        # controller_name.
        message = (
            "This section stores information for custom assets specific to this microcontroller. It includes satus, "
            "command and data-object mappings for each used module type (family) and the information about the "
            "specific instances of each module type. It also includes information about the microcontroller itself. "
            "This section is created separately for each used microcontroller and, in general, is expected to not "
            "contain the same information as sections for other microcontrollers."
        )
        code_dict = NestedDictionary(seed_dictionary={"section_description": message})

        # Adds the id-code and description of the microcontroller
        code_dict.write_nested_value(variable_path=f"code", value=self._controller_id)
        code_dict.write_nested_value(variable_path=f"description", value=self._controller_description)

        # This set is used to limit certain operations that only need to be performed once for each module type
        processed_types: set[np.uint8] = set()

        # This set is used to ensure each module instance has a unique id within the same module type.
        processed_id_types: set[np.uint16] = set()

        # Loops over all modules. Parses and uses their information to interactively fill the code dictionary with
        # information
        for module in self._modules:
            # Extracts type and id codes of the module
            module_id = module.module_id
            module_type = module.module_type
            module_section = f"{module.type_name}_module"  # Constructs module-specific section name

            # If the module's combined type + id code is already inside the processed_id_types set, this means another
            # module with the same exact type and ID combination has already been processed. This is not allowed, so
            # aborts with an error.
            if module.type_id in processed_id_types:
                message = (
                    f"Unable to initialize the MicroControllerInterface class instance for {self._controller_name} "
                    f"microcontroller with id {self._controller_id}. Encountered two ModuleInterface instances "
                    f"with the same type-code ({module_type}) and id-code ({module_id}), which is not allowed. Make "
                    f"sure each type and id combination is only used by a single ModuleInterface class instance."
                )
                console.error(message=message, error=ValueError)

            # If the error check above was not triggered, adds the unique type + id combination to the processed set.
            processed_id_types.add(module.type_id)

            # This section only needs to be executed once for each module type (family). It will be skipped for
            # instances with already processed module type codes.
            if module_type not in processed_types:
                # Adds each new module type to the processed types set
                processed_types.add(module_type)

                # Calls the class method that should fill the status_code, command and data_object module-type-specific
                # sections of the dictionary with data and return it to caller.
                code_dict = module.write_code_map(code_dict)

                # Adds the type-code and description of the module family to the module-type-specific section.
                section = f"{module_section}.code"
                code_dict.write_nested_value(variable_path=section, value=module.module_type)
                section = f"{module_section}.description"
                code_dict.write_nested_value(variable_path=section, value=module.type_description)

                # Adds placeholder variables to store instance ID codes, names, and descriptions. All instances are
                # combined into lists for more optimized handling. These placeholders are then iteratively filled with
                # information as the method loops through instances of the same type.
                section = f"{module_section}.instance_ids"
                code_dict.write_nested_value(variable_path=section, value=[])
                section = f"{module_section}.instance_names"
                code_dict.write_nested_value(variable_path=section, value=[])
                section = f"{module_section}.instance_descriptions"
                code_dict.write_nested_value(variable_path=section, value=[])

            # If the type information for the processed module has already been handled, adds its instance-specific
            # information to the dictionary. Note, this is ALSO done for the module instance that was used to parse the
            # type information.
            section = f"{module_section}.instance_ids"
            existing_values: list[np.uint8] = code_dict.read_nested_value(variable_path=section)
            existing_values.append(module.module_id)
            code_dict.write_nested_value(variable_path=section, value=existing_values)

            section = f"{module_section}.instance_names"
            existing_strings: list[str] = code_dict.read_nested_value(variable_path=section)
            existing_strings.append(module.instance_name)
            code_dict.write_nested_value(variable_path=section, value=existing_strings)

            section = f"{module_section}.instance_descriptions"
            existing_strings = code_dict.read_nested_value(variable_path=section)
            existing_strings.append(module.instance_description)
            code_dict.write_nested_value(variable_path=section, value=existing_strings)

            # Also, for each instance, records whether additional processing flags were used for that instance.
            section = f"{module_section}.unity_input"
            code_dict.write_nested_value(variable_path=section, value=module.unity_input)
            section = f"{module_section}.unity_output"
            code_dict.write_nested_value(variable_path=section, value=module.unity_output)
            section = f"{module_section}.queue_output"
            code_dict.write_nested_value(variable_path=section, value=module.queue_output)

        # Returns filled section dictionary to caller
        return code_dict

    @property
    def general_map_section(self) -> NestedDictionary:
        """Returns the NestedDictionary instance that stores the general information about the byte-codes used by
        AtaraxisMicroController (AXMC) library classes.

        The dictionary returned by this class should be integrated into the overall code-dictionary by saving it under
        the 'general' section name (key).

        Notes:
            This method only needs to be called once, as this section is shared by all microcontrollers.
        """

        # Seeds dictionary section with the main section description.
        # It is expected that the method building the overall mega-dictionary that integrates information from all
        # microcontrollers only adds this section once.
        message = (
            "This section stores general information about the Core classes that manage microcontroller runtime. "
            "Specifically, it maps the status, command and data object codes used during PC-MicroController "
            "communication to meaningful names and descriptions. This information is the same for all used "
            "microcontrollers and, therefore, this section only needs to be filled once."
        )
        code_dict = NestedDictionary(seed_dictionary={"section_description": message})

        # Kernel: status codes
        code_dict = self._write_kernel_status_codes(code_dict)

        # Kernel: command codes
        code_dict = self._write_kernel_command_codes(code_dict)

        # Kernel: object data
        code_dict = self._write_kernel_object_data(code_dict)

        # Module: core status codes.
        # Note, primarily, modules use custom status and command codes for each module family. This data is extracted
        # from managed ModuleInterface classes via the build_module_code_map() method. This section specifically tracks
        # the 'core' codes that each custom module inherits by subclassing the base Module class.
        code_dict = self._write_base_module_status_codes(code_dict)

        # Module: core object data
        code_dict = self._write_base_module_object_data(code_dict)

        # Communication: status codes
        # This is the final class of the 'core' triad. This class writes protocols and prototypes sections in addition
        # to the status_codes section. This information is used during data transmission and reception to decode various
        # incoming and outgoing message payloads.
        code_dict = self._write_communication_status_codes(code_dict)
        code_dict = prototypes.write_prototype_codes(code_dict)
        code_dict = protocols.write_protocol_codes(code_dict)

        # TransportLayer: status codes
        # This and the following sections track codes from classes wrapped by the Communication class. Due to the
        # importance of the communication library, we track all status codes that are (theoretically) relevant for
        # communication. TransportLayer is the source of all codes added in this way (it 'passes' CRC and COBS codes
        # as its own).
        code_dict = self._write_transport_layer_status_codes(code_dict)
        code_dict = self._write_cobs_status_codes(code_dict)
        code_dict = self._write_crc_status_codes(code_dict)

        return code_dict

    @property
    def microcontroller_map_section(self) -> NestedDictionary:
        """Returns the NestedDictionary instance that stores the microcontroller-specific codes and ID information.

        The dictionary returned by this class should be integrated into the overall code-dictionary by saving it under
        the microcontroller section name (key). To get the name of the microcontroller, use the 'controlled_name'
        property.

        Notes:
            This method needs to be called once for each microcontroller.
        """
        return self._controller_map_section

    def start(self) -> None:
        """Initializes the communication with the managed MicroController, Unity game engine, and other systems.

        The MicroControllerInterface class will not be able to carry out any communications until this method is called.
        If you have submitted commands to the class before calling start(), all queue commands will be transmitted in
        one step. Multiple commands addressed to the same module will likely interfere with each-other if pre-queued in
        this fashion.
        """

        # If the process has already been started, returns without doing anything.
        if self._started:
            return

        # Instantiates the array used to control the runtime of the communication Process.
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._controller_name}_terminator_array",
            # Uses class name to ensure the array buffer name is unique
            prototype=np.zeros(shape=3, dtype=np.uint8),
        )  # Instantiation automatically connects the main process to the array.

        # Sets up the communication process. This process continuously cycles through the communication loop until
        # terminated, enabling bidirectional communication with the controller.
        self._communication_process = Process(
            target=self._runtime_cycle,
            args=(
                self._controller_id,
                self._modules,
                self._input_queue,
                self._output_queue,
                self._logger_queue,
                self._terminator_array,
                self._usb_port,
                self._baudrate,
                self._max_tx_payload_size,
                self._unity_ip,
                self._unity_port,
                self._verbose,
            ),
            daemon=True,
        )
        self._communication_process.start()

        # Sends controller identification command and ensures the (connected) controller id matches expectation.
        self.identify_controller()

        # Blocks until the microcontroller has finished all initialization steps or encounters an initialization error.
        while self._terminator_array.read_data(1) != 7:
            if not self._communication_process.is_alive():
                print(f"Initialization error code {self._terminator_array.read_data(1)}")
                raise RuntimeError("Communication died unexpectedly.")

        # Sets the started flag
        self._started = True

    def stop(self) -> None:
        """Shuts down the communication process, frees all reserved resources, and discards any unprocessed data stored
        inside input and output queues."""

        # If the process has not been started, returns without doing anything.
        if not self._started:
            return

        # Sets the terminator tracker to 1, which triggers communication process shutdown.
        if self._terminator_array is not None:
            self._terminator_array.write_data(0, np.uint8(1))

        # Waits until the communication process terminates
        if self._communication_process is not None:
            self._communication_process.join()

        # Shuts down the multiprocessing manager. This collects all active queues and discards all unprocessed data.
        self._mp_manager.shutdown()

        # Disconnects from the shared memory array and destroys its shared buffer.
        if self._terminator_array is not None:
            self._terminator_array.disconnect()
            self._terminator_array.destroy()

    def identify_controller(self) -> None:
        """Prompts the connected MicroController to identify itself by returning its id code."""
        self._input_queue.put(self._identify_command)

    def reset_controller(self) -> None:
        """Resets the connected MicroController to use default hardware and software parameters."""
        self._input_queue.put(self._reset_command)

    def lock_controller(self) -> None:
        """Configures connected MicroController parameters to prevent all modules from writing to any output pin."""
        self._input_queue.put(self._enable_locks)

    def unlock_controller(self) -> None:
        """Configures connected MicroController parameters to allow all modules to write to any output pin."""
        self._input_queue.put(self._disable_locks)

    @property
    def output_queue(self) -> MPQueue:  # type: ignore
        """Returns the multiprocessing queue used by the communication process to pipe data to other concurrently active
        processes."""
        return self._output_queue

    def send_message(
        self,
        message: (
            ModuleParameters
            | OneOffModuleCommand
            | RepeatedModuleCommand
            | DequeueModuleCommand
            | KernelParameters
            | KernelCommand
        ),
    ) -> None:
        """Sends the input arbitrary message structure to the connected Microcontroller.

        This is the primary interface for communicating with the Microcontroller. It allows sending all supported
        message structures to the Microcontroller for further processing.
        """
        self._input_queue.put(message)

    @staticmethod
    def _runtime_cycle(
        controller_id: np.uint8,
        modules: tuple[ModuleInterface, ...],
        input_queue: MPQueue,  # type: ignore
        output_queue: MPQueue,  # type: ignore
        logger_queue: MPQueue,  # type: ignore
        terminator_array: SharedMemoryArray,
        usb_port: str,
        baudrate: int,
        payload_size: int,
        unity_ip: str,
        unity_port: int,
        verbose: bool = False,
    ) -> None:
        """The main communication loop runtime of the class.

        This method is designed to run in a remote Process. It encapsulates the steps for sending and receiving the
        data from the connected microcontroller. Primarily, the method routes the data between the microcontroller and
        the multiprocessing queues (inpout and output) of the class and Unity game engine (via the binding of an MQTT
        client).

        Args:
            controller_id: The byte-code identifier of the connected Microcontroller. This is used to ensure that the
                class manages the correct controller by checking the controlled_id inside received Identification
                messages against this input byte-code.
            modules: A tuple that stores ModuleInterface classes managed by this MicroControllerInterface instance.
            input_queue: The multiprocessing queue used by other processes to issue commands to the microcontroller.
            output_queue: The multiprocessing queue used by this process to pipe received data to other processes.
            logger_queue: The queue exposed by the DataLogger class that is used to buffer and pipe received and
                outgoing messages to be logged (saved) to disk.
            terminator_array: The shared memory array used to control the communication process runtime.
            usb_port: The serial port to which the target microcontroller is connected.
            baudrate: The communication baudrate to use. This option is ignored for controllers that use USB interface,
                 but is essential for controllers that use the UART interface.
            payload_size: The maximum size of the payload the managed microcontroller can receive. This is used to
                ensure all outgoing messages will fit inside the Serial reception buffer of the microcontroller.
            unity_ip: The IP-address of the MQTT broker to use for communication with Unity game engine.
            unity_port: The port number of the MQTT broker to use for communication with Unity game engine.
            verbose: A flag that determines whether the contents of the incoming and outgoing messages should be
                printed to console. This is only used during debugging and should be disabled during most runtimes.
        """

        # If the cycle is started in verbose mode and the console is not enabled, enables the console. The console will
        # be disabled as part of this method shutdown
        is_enabled = console.is_enabled
        if not is_enabled and verbose:
            console.enable()

        # Connects to the terminator array. This is done early, as the terminator_array is used to track the
        # initialization and runtime status of the process, since errors that occur inside the process usually get
        # silenced.
        terminator_array.connect()

        # Precreates the assets used to optimize the communication runtime cycling. These assets are filled below to
        # support efficient interaction between the Communication classes and the ModuleInterface classes.
        unity_input_map: dict[str, list[int]] = {}
        unity_output_map: dict[np.uint16, int] = {}
        queue_output_map: dict[np.uint16, int] = {}

        # Loops over all modules and configures the assets instantiated above
        for num, module in enumerate(modules):
            # If the module is configured to receive data from unity, configures the necessary data structures to enable
            # monitoring the necessary topics and allow to quickly pass the data received on that topic to the
            # appropriate module class for processing.
            if module.unity_input:
                for topic in module.unity_input_topics:
                    # Extends the list of module indices that listen for that particular topic. This allows addressing
                    # multiple modules at the same time, as long as they all listen to the same topic.
                    existing_modules = unity_input_map.get(topic, [])
                    unity_input_map[topic] = existing_modules + [num]

            # If the module is configured to output data to Unity or other processes, maps its type+id combined code
            # to its index number. This is used to quickly find the module interface instance addressed by incoming
            # data, so that they can then send the data to the appropriate output stream.
            if module.unity_output:
                unity_output_map[module.type_id] = num

            if module.queue_output:
                queue_output_map[module.type_id] = num

        # Disables unused processing steps. For example, if none of the managed modules send data to Unity, that
        # processing step is disabled outright via a simple boolean if check (see the communication loop code).
        unity_input = False
        unity_output = False
        queue_output = False

        # Note, keys() essentially returns a set of keys, since the same hash-map optimizations are involved with
        # dictionary keys as with set values.
        if len(unity_input_map.keys()) != 0:
            unity_input = True
        if len(unity_output_map.keys()) != 0:
            unity_output = True
        if len(queue_output_map.keys()) != 0:
            queue_output = True

        # Reports that additional module processing steps have been resolved.
        terminator_array.write_data(index=1, data=np.uint8(1))

        # Initializes the serial communication class and connects to the managed MicroController.
        try:
            serial_communication = SerialCommunication(
                usb_port=usb_port,
                source_id=int(controller_id),
                logger_queue=logger_queue,
                baudrate=baudrate,
                maximum_transmitted_payload_size=payload_size,
            )
        except Exception as e:
            # Communicates that the process terminated at establishing serial communication
            terminator_array.write_data(index=1, data=np.uint8(2))

            # Also prints the error to terminal.
            print(e)
            sys.stdout.flush()

            sys.exit(2)

        # Reports that communication class has been successfully initialized.
        terminator_array.write_data(index=1, data=np.uint8(3))

        # Initializes the unity_communication class and connects to the MQTT broker, if the class is configured to
        # communicate with Unity.
        try:
            if unity_input or unity_output:
                # If the set is empty, the class initialization method will correctly interpret this as a case where no
                # topics need to be monitored. Therefore, it is safe to just pass the set regardless of whether it is
                # empty or not.
                unity_communication = UnityCommunication(
                    ip=unity_ip, port=unity_port, monitored_topics=tuple(tuple(unity_input_map.keys()))
                )
                unity_communication.connect()
            else:
                unity_communication = None
        except Exception as e:
            # Communicates that the process terminated at establishing unity communication
            terminator_array.write_data(index=1, data=np.uint8(4))

            # Also prints the error to terminal.
            print(e)
            sys.stdout.flush()

            sys.exit(4)

        # Reports that communication class has been successfully initialized.
        terminator_array.write_data(index=1, data=np.uint8(5))

        # Initializes the main communication loop. This loop will run until the exit conditions are encountered.
        # The exit conditions for the loop require the first variable in the terminator_array to be set to True
        # and the main input queue of the interface to be empty. This ensures that all queued commands issued from
        # the central process are fully carried out before the communication is terminated.
        while not terminator_array.read_data(index=0, convert_output=True) or not input_queue.empty():

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

            # Unity data sending loop. The loop is only executed if unity_input flag is enabled.
            if unity_input:
                while unity_communication.has_data:  # type: ignore
                    # If UnityCommunication has received data, loops over all interfaces that requested the data from
                    # this topic and calls their unity data processing method. The method is expected to extract the
                    # data from the communication class and translate it into a valid message format to be sent to the
                    # microcontroller.
                    topic, payload = unity_communication.get_data()  # type: ignore

                    # Each incoming message will be processed by each module subscribed to this topic. Since
                    # UnityCommunication is configured to only listen to topics submitted by the interface classes, the
                    # topic is guaranteed to be inside the unity_input_map dictionary and have at least one Module which
                    # can process its data.
                    for i in unity_input_map[topic]:
                        out_data = modules[i].get_from_unity(topic=topic, payload=payload)  # type: ignore
                        serial_communication.send_message(out_data)  # Transmits the data to the microcontroller

            # Attempts to receive the data from microcontroller
            in_data = serial_communication.receive_message()

            # If no data is available cycles the loop
            if in_data is None:
                continue

            # Otherwise, resolves additional processing steps associated with incoming data. Currently, only Module
            # interfaces have additional data processing steps. Therefore, limits the evaluation to ModuleState and
            # Data messages.
            if isinstance(in_data, (ModuleState, ModuleData)) and (unity_output or queue_output):
                # Only carries out additional processing steps if unity or queue output flags are enabled.

                # Computes the combined type and id code for the incoming data. This allows quickly finding the unique
                # addressee of the data payload.
                target_type_id: np.uint16 = np.uint16(
                    (in_data.module_type.astype(np.uint16) << 8) | in_data.module_type.astype(np.uint16)
                )

                # Depending on whether the combined code is inside the unity_output_map, queue_output_map or both
                # executes the necessary module's method to handle data output.
                if target_type_id in unity_output_map.keys():
                    modules[unity_output_map[target_type_id]].send_to_unity(
                        message=in_data,
                        unity_communication=unity_communication,  # type: ignore
                    )
                if target_type_id in queue_output_map.keys():
                    modules[queue_output_map[target_type_id]].send_to_queue(message=in_data, queue=output_queue)

            # Whenever the incoming message is the Identification message, ensures that the received controller_id
            # matches the ID expected by the class.
            elif isinstance(in_data, Identification):

                if in_data.controller_id != controller_id:
                    # Reports that the controller-sent id code does not match the expected controller_id.
                    terminator_array.write_data(index=1, data=np.uint8(6))

                    # Raises the error.
                    message = (
                        f"Unexpected controller_id code received from the microcontroller managed by the "
                        f"MicroControllerInterface instance. Expected {controller_id}, but received "
                        f"{in_data.controller_id}."
                    )
                    console.error(message=message, error=ValueError)

                else:
                    # Reports that communication class has been successfully initialized. Seeing this code means that
                    # the communication appears to be functioning correctly, at least in terms of the data reaching the
                    # Kernel.
                    terminator_array.write_data(index=1, data=np.uint8(7))

        # If this point is reached, the loop has received the shutdown command and successfully escaped the
        # communication cycle.
        # Disconnects from the terminator array and shuts down Unity communication.
        terminator_array.disconnect()
        if unity_communication is not None:
            unity_communication.disconnect()

        # If this method enabled the console to comply with the verbose flag, disables the console.
        if is_enabled and verbose:
            console.disable()

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
        buffer = SharedMemory(name=f"{self._controller_name}_terminator_array", create=False)
        buffer.close()
        buffer.unlink()
