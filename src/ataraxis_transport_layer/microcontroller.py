"""This module provides the ModuleInterface and MicroControllerInterface classes that aggregate the methods to
bidirectionally transmit data between PC, MicroControllers, and Unity game engine.

Specifically, each microcontroller module that manages physical hardware should be matched to a specialized instance
of the ModuleInterface class. Similarly, for each concurrently active microcontroller, there has to be a specific
MicroControllerInterface instance that manages the ModuleInterface instances for the modules of that controller.

In addition to carrying out the communication, these classes also jointly create dictionary maps that match byte-codes
used during communication to human-readable names and descriptions. This is necessary to properly decode the
communication logs that store transmitted data as byte-serialized payloads.
"""

from abc import abstractmethod
import sys
from threading import Thread
from multiprocessing import (
    Queue as MPQueue,
    Manager,
    Process,
)
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console
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
            to be sent to the microcontroller. While the signature contains None as a return, None is NOT a valid
            return value. MicroControllerInterface class is designed to ONLY call this method if it expects a non-None
            return. Make sure this method is properly implemented if your module includes a list of Unity topics to
            listen for.
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

        Args:
            code_map: The shared NestedDictionary instance that aggregates all information from a single
                MicroControllerInterface class, including the information from all ModuleInterface instances managed by
                the class.

        Returns:
            The updated NestedDictionary instance. It is assumed that all valid modules always update the input
            dictionary as part of this method runtime.

        """
        raise NotImplementedError(
            f"write_code_map method for {self._type_name} module interface must be implemented when subclassing the "
            f"base ModuleInterface class."
        )

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
        raise NotImplementedError(
            f"write_instance_variables method for {self._type_name} module interface must be implemented when "
            f"subclassing the base ModuleInterface class."
        )

    def dequeue_command(self) -> DequeueModuleCommand:
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

        # Verbose flag and started trackers
        self._verbose: bool = verbose
        self._started: bool = False

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
        self._watchdog_thread: None | Thread = None

        # Extracts information from the input modules and finalizes runtime preparations. As part of this process,
        # the method will create the microcontroller-specific code-map dictionary section.
        self._controller_map_section: NestedDictionary = self._parse_module_data()

    def __repr__(self) -> str:
        """Returns a string representation of the class instance."""
        return (
            f"MicroControllerInterface(controller_id={self._controller_id}, controller_name={self._controller_name}, "
            f"usb_port={self._usb_port}, baudrate={self._baudrate}, max_tx_payload_size={self._max_tx_payload_size}, "
            f"unity_ip={self._unity_ip}, unity_port={self._unity_port}, started={self._started})"
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

            # For each module instance, adds its instance-specific information to the dictionary.
            section = f"{module_section}.{module.instance_name}.code"
            code_dict.write_nested_value(variable_path=section, value=module.module_id)
            section = f"{module_section}.{module.instance_name}.description"
            code_dict.write_nested_value(variable_path=section, value=module.instance_description)

            # Finally, adds the custom variables section for each instance by calling the appropriate method.
            code_dict = module.write_instance_variables(code_dict)

        # Returns filled section dictionary to caller
        return code_dict

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

    @property
    def output_queue(self) -> MPQueue:  # type: ignore
        """Returns the multiprocessing queue used by the communication process to pipe received data to other
        concurrently active processes."""
        return self._output_queue

    def _watchdog(self) -> None:
        """This function should be used by the watchdog thread to ensure the communication process is alive during
        runtime.

        This function will raise a RuntimeError if it detects that a process has prematurely shut down. It will verify
        process states every ~20 ms and will release the GIL between checking the states.
        """
        timer = PrecisionTimer(precision="ms")

        # The watchdog function will run until the global shutdown command is issued.
        while not self._terminator_array.read_data(index=0):  # type: ignore
            # Checks process state every 20 ms. Releases the GIL while waiting.
            timer.delay_noblock(delay=20, allow_sleep=True)

            if not self._started:
                continue

            # Only checks that the process is alive if it is started. The shutdown() flips the started tracker
            # before actually shutting down the process, so there should be no collisions here.
            if self._communication_process is not None and not self._communication_process.is_alive():
                message = (
                    f"The communication process for the MicroControllerInterface {self._controller_name} with id "
                    f"{self._controller_id} has been prematurely shut down. This likely indicates that the process has "
                    f"encountered a runtime error that terminated the process."
                )
                console.error(message=message, error=RuntimeError)

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

        # If the process has already been started, returns without doing anything.
        if self._started:
            return

        # Instantiates the array used to control the runtime of the communication Process.
        self._terminator_array = SharedMemoryArray.create_array(
            name=f"{self._controller_name}_terminator_array",
            # Uses class name to ensure the array buffer name is unique
            prototype=np.zeros(shape=2, dtype=np.uint8),  # Index 0 = terminator, index 1 = initialization status
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

        # Creates the watchdog thread.
        self._watchdog_thread = Thread(target=self._watchdog, daemon=True)

        # Initializes the communication process.
        self._communication_process.start()

        # Sends controller identification command and ensures the (connected) controller id matches expectation.
        self.identify_controller()

        start_timer = PrecisionTimer("s")
        start_timer.reset()
        # Blocks until the microcontroller has finished all initialization steps or encounters an initialization error.
        while self._terminator_array.read_data(1) != 1:
            # Generally, there are two ways initialization failure is detected. One is if the managed process
            # terminates, which would be the case if any subclass used in the communication process raises an exception.
            # Another way if the status tracker never reaches success code (1). This latter case would likely indicate
            # that there is a communication issue where the data does not reach the controller or the PC. The
            # initialization process should be VERY fast, likely on the order of hundreds of microseconds. Waiting for
            # 5 seconds is likely complete overkill.
            if not self._communication_process.is_alive() or start_timer.elapsed > 5:
                message = (
                    f"MicroControllerInterface for {self._controller_name} (id={self._controller_id}) failed to "
                    f"initialize the communication runtime with the microcontroller. If the class was initialized with "
                    f"the 'verbose' flag disabled, enable the flag to get more information and debug the "
                    f"initialization process error."
                )
                console.error(error=RuntimeError, message=message)

        # Starts the process watchdog thread once the initialization is complete
        self._watchdog_thread.start()

        # Sets the started flag
        self._started = True

    def stop(self) -> None:
        """Shuts down the communication process, frees all reserved resources, and discards any unprocessed data stored
        inside input and output queues."""

        # If the process has not been started, returns without doing anything.
        if not self._started:
            return

        self.reset_controller()  # Resets the controller. This automatically locks all pins and resets modules.

        # There is no need for additional delays as the communication loop will make sure the reset command is sent
        # to the controller before shutdown

        # Changes the started tracker value. Amongst other things this soft-inactivates the watchdog thread.
        self._started = False

        # Sets the terminator trigger to 1, which triggers communication process shutdown. This also shuts down the
        # watchdog thread.
        if self._terminator_array is not None:
            self._terminator_array.write_data(0, np.uint8(1))

        # Waits until the communication process terminates
        if self._communication_process is not None:
            self._communication_process.join()

        # Shuts down the multiprocessing manager. This collects all active queues and discards all unprocessed data.
        self._mp_manager.shutdown()

        # Waits for the watchdog thread to terminate.
        if self._watchdog_thread is not None:
            self._watchdog_thread.join()

        # Disconnects from the shared memory array and destroys its shared buffer.
        if self._terminator_array is not None:
            self._terminator_array.disconnect()
            self._terminator_array.destroy()

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

        # Connects to the terminator array. This is done early, as the terminator_array is used to track the
        # initialization and runtime status of the process.
        terminator_array.connect()

        # If the runtime is called in the verbose mode, ensures the console is enabled.
        was_enabled = console.enabled
        if verbose and not was_enabled:
            console.enable()

            # Also sends a message to notify that the initialization has started.
            console.echo(
                message=f"Starting MicroControllerInterface for controller {controller_id}...", level=LogLevel.INFO
            )

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

        # Initializes the serial communication class and connects to the managed MicroController.
        serial_communication = SerialCommunication(
            usb_port=usb_port,
            source_id=controller_id,
            logger_queue=logger_queue,
            baudrate=baudrate,
            maximum_transmitted_payload_size=payload_size,
            verbose=verbose,
        )

        # Initializes the unity_communication class and connects to the MQTT broker. If the interface does not
        # need Unity communication, this initialization will only statically reserve some RAM with no other
        # adverse effects.
        # If the set is empty, the class initialization method will correctly interpret this as a case where no
        # topics need to be monitored. Therefore, it is safe to just pass the set regardless of whether it is
        # empty or not.
        unity_communication = UnityCommunication(
            ip=unity_ip, port=unity_port, monitored_topics=tuple(tuple(unity_input_map.keys()))
        )

        # Only connects to the class if managed modules need to send or receive data from unity.
        if unity_input or unity_output:
            unity_communication.connect()

        # This notifies the user that subclass initialization is complete. The interface still needs to verify that the
        # connected controller has the correct ID (see below).
        if verbose:
            console.echo(
                message=f"MicroControllerInterface {controller_id} initialization complete.", level=LogLevel.SUCCESS
            )

        try:
            # Initializes the main communication loop. This loop will run until the exit conditions are encountered.
            # The exit conditions for the loop require the first variable in the terminator_array to be set to True
            # and the main input queue of the interface to be empty. This ensures that all queued commands issued from
            # the central process are fully carried out before the communication is terminated.
            while not terminator_array.read_data(index=0, convert_output=True) or not input_queue.empty():
                # Main data sending loop. The method will sequentially retrieve the queued command and parameter data
                # to be sent to the Microcontroller and send it.
                while not input_queue.empty():
                    out_data = input_queue.get()

                    serial_communication.send_message(out_data)  # Transmits the data to the microcontroller

                # Unity data sending loop. This loop will be O(1) if unity never has data. In turn, this will always be
                # teh case if no module supports unity inputs. Therefore, there is no need to both have an 'if' and a
                # 'while' check here for optimal runtime speed.
                while unity_communication.has_data:
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
                        out_data = modules[i].get_from_unity(topic=topic, payload=payload)
                        if out_data is not None:
                            serial_communication.send_message(out_data)  # Transmits the data to the microcontroller

                # Attempts to receive the data from microcontroller
                in_data = serial_communication.receive_message()

                # If no data is available cycles the loop
                if in_data is None:
                    continue

                # Otherwise, resolves additional processing steps associated with incoming data. Currently, only Module
                # interfaces have additional data processing steps that are expected more than once during runtime. The
                # Kernel Identification message also has a unique processing step, but it should only be executed once,
                # during runtime initialization.
                if isinstance(in_data, (ModuleState, ModuleData)) and (unity_output or queue_output):
                    # Computes the combined type and id code for the incoming data. This is used to find the specific
                    # ModuleInterface to which the message is addressed and, if necessary, invoke interface-specific
                    # additional processing method.
                    target_type_id: np.uint16 = np.uint16(
                        (in_data.module_type.astype(np.uint16) << 8) | in_data.module_id.astype(np.uint16)
                    )

                    # Depending on whether the combined code is inside the unity_output_map, queue_output_map, or both,
                    # executes the necessary module's method to handle data output.
                    if target_type_id in unity_output_map.keys():
                        modules[unity_output_map[target_type_id]].send_to_unity(
                            message=in_data,
                            unity_communication=unity_communication,
                        )
                    if target_type_id in queue_output_map.keys():
                        modules[queue_output_map[target_type_id]].send_to_queue(message=in_data, queue=output_queue)

                # Whenever the incoming message is the Identification message, ensures that the received controller_id
                # matches the ID expected by the class.
                elif isinstance(in_data, Identification):
                    if in_data.controller_id != controller_id:
                        # Raises the error.
                        message = (
                            f"Unexpected controller_id code received from the microcontroller managed by the "
                            f"MicroControllerInterface instance. Expected {controller_id}, but received "
                            f"{in_data.controller_id}."
                        )
                        console.error(message=message, error=ValueError)

                    else:
                        # Reports that communication class has been successfully initialized. Seeing this code means
                        # that the communication appears to be functioning correctly, at least in terms of the data
                        # reaching the Kernel and back. While this does not guarantee the runtime will continue running
                        # without errors, it is very likely to be so.
                        terminator_array.write_data(index=1, data=np.uint8(1))

        # If an unknown and unhandled exception occurs, prints and flushes the exception message to the terminal
        # before re-raising the exception to terminate the process.
        except Exception as e:
            sys.stderr.write(str(e))
            sys.stderr.flush()
            raise e

        # If this point is reached, the loop has received the shutdown command and successfully escaped the
        # communication cycle.
        # Disconnects from the terminator array and shuts down Unity communication.
        terminator_array.disconnect()
        unity_communication.disconnect()

        # If this runtime had to enable the console to comply with 'verbose' flag, disables it before ending the
        # runtime.
        if verbose and not was_enabled:
            # Notifies the user that the runtime Process has been successfully terminated
            console.echo(
                message=f"MicroControllerInterface {controller_id} communication runtime terminated.",
                level=LogLevel.SUCCESS,
            )
            console.disable()

    def _vacate_shared_memory_buffer(self) -> None:
        """Clears the SharedMemory buffer with the same name as the one used by the class.

        While this method should not be needed if the class is used correctly, there is a possibility that invalid
        class termination leaves behind non-garbage-collected SharedMemory buffer. In turn, this would prevent the
        class remote Process from being started again. This method allows manually removing that buffer to reset the
        system.

        This method is designed to do nothing if the buffer with the same name as the microcontroller does not exist.
        """
        try:
            buffer = SharedMemory(name=f"{self._controller_name}_terminator_array", create=False)
            buffer.close()
            buffer.unlink()
        except FileNotFoundError:
            pass
