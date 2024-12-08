from abc import abstractmethod
from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from ataraxis_data_structures import NestedDictionary, SharedMemoryArray

from .communication import (
    ModuleData as ModuleData,
    ModuleState as ModuleState,
    KernelCommand as KernelCommand,
    Identification as Identification,
    KernelParameters as KernelParameters,
    ModuleParameters as ModuleParameters,
    UnityCommunication as UnityCommunication,
    OneOffModuleCommand as OneOffModuleCommand,
    SerialCommunication as SerialCommunication,
    DequeueModuleCommand as DequeueModuleCommand,
    RepeatedModuleCommand as RepeatedModuleCommand,
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

    _module_type: Incomplete
    _type_name: Incomplete
    _type_description: Incomplete
    _module_id: Incomplete
    _instance_name: Incomplete
    _instance_description: Incomplete
    _type_id: Incomplete
    _unity_input_topics: Incomplete
    _unity_input: Incomplete
    _unity_output: Incomplete
    _queue_output: Incomplete
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
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns the string representation of the ModuleInterface instance."""
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
            to be sent to the microcontroller. While the signature contains None as a return, None is NOT a valid
            return value. MicroControllerInterface class is designed to ONLY call this method if it expects a non-None
            return. Make sure this method is properly implemented if your module includes a list of Unity topics to
            listen for.
        """
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
    @abstractmethod
    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
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

        Args:
            code_map: The shared NestedDictionary instance that aggregates all information from a single
                MicroControllerInterface class, including the information from all ModuleInterface instances managed by
                the class.

        Returns:
            The updated NestedDictionary instance. It is assumed that all valid modules always update the input
            dictionary as part of this method runtime.

        """
    @abstractmethod
    def write_instance_variables(self, code_map: NestedDictionary) -> NestedDictionary:
        """Updates the input code_map dictionary with module-instance-specific runtime variables.

        This method allows writing instance-specific variables to the global code map. For example, this method can
        be used to store specific conversion factors used to translate pulses from a specific EncoderModule into
        centimeters. At the very least, each module has to use this method to write whether it uses unity input / output
        queue output, as well as any other instance variables that may be helpful when parsing the logged data.

        Notes:
            This method functions identically to write_code_map(), except that it is called for every instance with the
            same module_type.

            Make sure all data is written under modulename_module.instancename.instance_variables root path. Otherwise,
            some parser features available for all instances (for example, parsing unity input / output boolean flags)
            will not work correctly.

            See one of the default custom ModuleInterface classes for examples on how to implement this method, both
            for modules that do and do not use this functionality.

        Args:
            code_map: The shared NestedDictionary instance that aggregates all information from a single
                MicroControllerInterface class, including the information from all ModuleInterface instances managed by
                the class.

        Returns:
            An updated NestedDictionary instance. All valid module instances are expected to use this method to at least
            write whether they use unity input / output and queue output flags.

        """
    def dequeue_command(self) -> DequeueModuleCommand:
        """Returns the command that, upon being sent to the microcontroller, will clear all queued commands for this
        module instance.

        Since these Dequeue commands are universal, the method that packages and returns these command messages is
        defined as a non-abstract base ModuleInterface class method.
        """
    @property
    def module_type(self) -> np.uint8:
        """Returns the byte-code of the interfaced module type (family)."""
    @property
    def type_name(self) -> str:
        """Returns the human-readable name of the interfaced module type (family)."""
    @property
    def type_description(self) -> str:
        """Returns the human-readable description of the interfaced module type (family)."""
    @property
    def module_id(self) -> np.uint8:
        """Returns the byte-code identifier (ID) of the specific interfaced module instance."""
    @property
    def instance_name(self) -> str:
        """Returns the human-readable name of the specific interfaced module instance."""
    @property
    def instance_description(self) -> str:
        """Returns the human-readable description of the interfaced module instance."""
    @property
    def unity_output(self) -> bool:
        """Returns True if the class is configured to send the data received from the module instance to Unity."""
    @property
    def unity_input(self) -> bool:
        """Returns True if the class is configured to receive commands from Unity and send them to module instance."""
    @property
    def queue_output(self) -> bool:
        """Returns True if the class is configured to send the data received from the module instance to other
        processes."""
    @property
    def unity_input_topics(self) -> tuple[str, ...]:
        """Returns the tuple of MQTT topics that should be monitored for incoming Unity commands."""
    @property
    def type_id(self) -> np.uint16:
        """Returns the unique unsigned integer value that results from combining the type-code and the id-code of the
        module instance."""

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
        verbose: Determines whether the communication cycle reports runtime progress, including the contents of all
            incoming and outgoing messages, by printing messages to the console. This option is used during debugging
            and should be disabled during production runtimes.

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
            _watchdog_thread: A thread used to monitor the runtime status of the remote communication process.
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
            _started: Tracks whether the communication process has been started. This is used to prevent calling
                the start() and stop() methods multiple times.
    """

    _reset_command: Incomplete
    _identify_command: Incomplete
    _disable_locks: Incomplete
    _enable_locks: Incomplete
    _controller_id: Incomplete
    _controller_name: Incomplete
    _controller_description: Incomplete
    _usb_port: Incomplete
    _baudrate: Incomplete
    _max_tx_payload_size: Incomplete
    _unity_ip: Incomplete
    _unity_port: Incomplete
    _verbose: Incomplete
    _started: bool
    _modules: Incomplete
    _logger_queue: Incomplete
    _mp_manager: Incomplete
    _input_queue: Incomplete
    _output_queue: Incomplete
    _terminator_array: Incomplete
    _communication_process: Incomplete
    _watchdog_thread: Incomplete
    _controller_map_section: Incomplete
    def __init__(
        self,
        controller_id: np.uint8,
        controller_name: str,
        controller_description: str,
        controller_usb_port: str,
        logger_queue: MPQueue,
        modules: tuple[ModuleInterface, ...],
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
        unity_broker_ip: str = "127.0.0.1",
        unity_broker_port: int = 1883,
        verbose: bool = False,
    ) -> None: ...
    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
    def __del__(self) -> None:
        """Ensures that all class resources are properly released when the class instance is garbage-collected."""
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
    def identify_controller(self) -> None:
        """Prompts the connected MicroController to identify itself by returning its id code."""
    def reset_controller(self) -> None:
        """Resets the connected MicroController to use default hardware and software parameters."""
    def lock_controller(self) -> None:
        """Configures connected MicroController parameters to prevent all modules from writing to any output pin."""
    def unlock_controller(self) -> None:
        """Configures connected MicroController parameters to allow all modules to write to any output pin."""
    def send_message(
        self,
        message: ModuleParameters
        | OneOffModuleCommand
        | RepeatedModuleCommand
        | DequeueModuleCommand
        | KernelParameters
        | KernelCommand,
    ) -> None:
        """Sends the input arbitrary message structure to the connected Microcontroller.

        This is the primary interface for communicating with the Microcontroller. It allows sending all supported
        message structures to the Microcontroller for further processing.
        """
    @property
    def output_queue(self) -> MPQueue:
        """Returns the multiprocessing queue used by the communication process to pipe received data to other
        concurrently active processes."""
    def _watchdog(self) -> None:
        """This function should be used by the watchdog thread to ensure the communication process is alive during
        runtime.

        This function will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """
    def start(self) -> None:
        """Initializes the communication with the managed MicroController, Unity game engine, and other systems.

        The MicroControllerInterface class will not be able to carry out any communications until this method is called.
        If you have submitted commands to the class before calling start(), all queued commands will be transmitted in
        one step. Multiple commands addressed to the same module will likely interfere with each-other if pre-queued in
        this fashion.

        Note:
            As part of this method runtime, the interface emits an identification request and ensures that the
            connected microController responds with the id_code that exactly matches the id code used during class
            initialization.

        Raises:
            RuntimeError: If the class is not able to properly initialize the communication runtime. The actual cause
                of this error is usually one of the numerous subclasses used in the process. Use 'verbose' flag
                during class initialization to view the detailed error message that aborts the initialization process.
        """
    def stop(self) -> None:
        """Shuts down the communication process, frees all reserved resources, and discards any unprocessed data stored
        inside input and output queues."""
    @staticmethod
    def _runtime_cycle(
        controller_id: np.uint8,
        modules: tuple[ModuleInterface, ...],
        input_queue: MPQueue,
        output_queue: MPQueue,
        logger_queue: MPQueue,
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
    def _vacate_shared_memory_buffer(self) -> None:
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed if the class is used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer. In turn, this would prevent the
        class remote Process from being started again. This method allows manually removing that buffer to reset the
        system.

        This method is designed to do nothing if the buffer with the same name as the microcontroller does not exist.
        """
