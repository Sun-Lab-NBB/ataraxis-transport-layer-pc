import numpy as np
from src.ataraxis_transport_layer.communication import (
    SerialCommunication,
    CommandMessage,
    IdentificationMessage,
    DataMessage,
)
from ataraxis_data_structures import NestedDictionary, SharedMemoryArray
from dataclasses import dataclass
from typing import Any, Optional
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
    ProcessError,
)
from multiprocessing.managers import SyncManager
from ataraxis_base_utilities import console

class Module:

    def __init__(self, module_type: np.uint8, module_id: np.uint8, custom_command_map: dict):
        self._module_type: np.uint8 = module_type
        self._module_id: np.uint8 = module_id
        self._commands: dict = {}
        self._status_codes: dict = {}
        self._prototypes: dict = {}
        self._unity_channels_map: dict = {}

    def make_command(
            self,
            command_code: np.uint8,
            return_code: np.uint8 = 0,
            noblock: np.bool = True,
            cycle: np.bool = False,
            cycle_delay: np.uint32 = 0,
    ) -> None:
        pass

    @staticmethod
    def write_status_codes_map():
        raise NotImplementedError #TODO

    @staticmethod
    def write_command_codes_map():
        pass

    def make_parameters(
            self,
            return_code: np.uint8 = 0,
    ) -> None:
        pass  # TODO Virtual (not implemented error)

    def process_data(self, message: DataMessage):
        pass  # TODO Virtual (not implemented error)


class MicroController:
    _kernel_type = np.uint8(1)
    _kernel_id = np.uint8(0)
    _identify_command_code = np.uint8(4)

    def __init__(
            self,
            name: str,
            usb_port: str,
            baud_rate: int,
            reception_buffer_size: int,
            id_code: int,
            modules: tuple[Module, ...],
    ):

        self._name: str = name

        self._communication = SerialCommunication(usb_port, baud_rate, reception_buffer_size)

        self._modules = modules

        identify_command = CommandMessage(
            module_type=self._kernel_type,
            module_id=self._kernel_id,
            command=self._identify_command_code,
            return_code=np.uint8(0),
            noblock=False,
            cycle=False,
            cycle_delay=np.uint32(0),
        )

        self._communication.send_message(identify_command)

        while True:
            message = self._communication.receive_message()
            if message is None:
                continue
            elif isinstance(message, IdentificationMessage):
                if message.controller_id == id_code:
                    break
                else:
                    raise ValueError("Invalid controller ID!")
            else:
                raise ValueError("Received unexpected message type")

        # Sets up the multiprocessing Queue, which is used to buffer and pipe images from the producer (camera) to
        # one or more consumers (savers). Uses Manager() instantiation as it has a working qsize() method for all
        # supported platforms.
        self._mp_manager: SyncManager = multiprocessing.Manager()
        self._image_queue: MPQueue = self._mp_manager.Queue()  # type: ignore

        # Instantiates an array shared between all processes. This array is used to control all child processes.
        # Index 0 (element 1) is used to issue global process termination command, index 1 (element 2) is used to
        # flexibly enable or disable saving camera frames.
        self._terminator_array: SharedMemoryArray = SharedMemoryArray.create_array(
            name=f"{self._name}_terminator_array",  # Uses class name with an additional specifier
            prototype=np.array([0, 0], dtype=np.int32),
        )  # Instantiation automatically connects the main process to the array.

        # Sets up the image producer Process. This process continuously executes a loop that conditionally grabs frames
        # from camera, optionally displays them to the user, and queues them up to be saved by the consumers.
        self._producer_process: Process = Process(
            target=self.runtime_cycle,
            args=(self._image_queue, self._terminator_array),
            daemon=True,
        )

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
        message = f"The byte-code that identifies messages sent by or to the Kernel module of the MicroController."
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
        code_dictionary = MicroController._write_module_status_codes(code_dictionary)

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
        If there is a mismatch, you may be unable to interpret logged data during offline parsing or interpret it
        incorrectly.

        Args:
            code_dictionary: The dictionary to be filled with kernel status codes.

        Returns:
            The updated dictionary with kernel status codes information filled.
        """

        section = "kernel.status_codes.kStandBy"
        description = "The value used to initialize internal Kernel status tracker."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=0)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kSetupComplete"
        description = (
            "The microcontroller hardware (e.g.: pin modes) and software (e.g.: command queue) parameters were "
            "successfully (re)set to hardcoded defaults."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=1)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleSetupError"
        description = (
            "The microcontroller was not able to (re)set its hardware and software parameters due to one of the "
            "managed custom modules failing its' setup method runtime."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=2)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kNoDataToReceive"
        description = (
            "The Kernel did not receive any data to parse during the current execution loop cycle. This is not an "
            "error, this is one of the common default states of the Kernel!"
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=3)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kDataReceptionError"
        description = (
            "The Kernel failed to parse the data sent from the PC. This can be due to a number of errors, including "
            "corruption in transmission and invalid data format."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=4)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kDataSendingError"
        description = (
            "The Kernel failed to send a DataMessage to the PC. Usually, this indicates that the chosen data payload "
            "format is not valid."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=5)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kInvalidDataProtocol"
        description = (
            "The Kernel has received a message from the PC that does not use a valid (supported) message protocol. "
            "The message protocol is communicated by the first variable of each message payload and determines how to "
            "parse the rest of the payload. This error typically indicates a mismatch between the PC and "
            "Microcontroller codebase versions or data corruption errors."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=6)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kKernelParametersSet"
        description = (
            "New parameter-values addressed to the Kernel (controller-wide parameters) were received and "
            "applied successfully."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=7)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kKernelParametersError"
        description = (
            "The Kernel-addressed parameter-values do not match the format of the structure used to store Kernel "
            "parameters. In turn, this makes it impossible for the Kernel to decode parameter-values from the "
            "serialized message payload."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=8)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kModuleParametersSet"
        description = (
            "New parameter-values addressed to a custom (user-defined) Module class instance were received and "
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

        section = "kernel.status_codes.kParametersTargetNotFound"
        description = (
            "Unable to find the addressee of the new parameter-values sent from the PC. The module_type and module_id "
            "fields of the message did not match the Kernel class or any of the custom Modules. Usually, this "
            "indicates a malformed message (user-error)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=11)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kControllerReset"
        description = (
            "The Kernel has successfully (re)set the hardware (e.g.: pin mode) and software (e.g.: command queue) "
            "parameters of the managed microcontroller back to default hardcoded values. This procedure is identical "
            "to the original controller setup, but is triggered in response to an explicit PC-sent command."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=12)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kKernelCommandUnknown"
        description = (
            "The Kernel has received an unknown Kernel command code from the PC. Usually, this indicates data "
            "corruption or a mismatch between the PC and Microcontroller codebase versions."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=13)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kModuleCommandQueued"
        description = (
            "The Kernel has received a command code addressed to one of the custom Modules and successfully queued it "
            "to be evaluated and executed when the module finishes any currently active commands. Note, command code 0 "
            "would trigger kModuleCommandsReset status (22)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=14)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kCommandTargetNotFound"
        description = (
            "Unable to find the addressee of the command-code sent from the PC. The module_type and module_id "
            "fields of the message did not match the Kernel class or any of the custom Modules. Usually, this "
            "indicates a malformed message (user-error)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=15)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kServiceSendingError"
        description = (
            "The Kernel failed to send a service message (e.g.: identificationMessage) to the PC. Usually, this "
            "indicates that the chosen message payload format is not valid."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=16)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kControllerIDSent"
        description = (
            "The Kernel has successfully sent the hardcoded controller ID byte-code to the PC, following the "
            "identification request (command)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=17)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleCommandError"
        description = (
            "A custom Module has failed to execute a command. Currently, the only way to produce this error is for the "
            "custom module to not recognize the command code. In turn, this usually indicates a user-error or a "
            "mismatch between the PC and Microcontroller codebase versions."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=18)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kModuleCommandsCompleted"
        description = (
            "All managed custom Modules have successfully resolved (determined which command to run) and completed the"
            "necessary stage(s) of their active commands during this runtime cycle."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=19)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kModuleAssetResetError"
        description = (
            "Failed to reset the custom assets of a module. Custom assets are any assets not exposed through the "
            "default API inherited by custom Modules from the base Module class. These assets are usually unique for "
            "each type of modules and track hardware-specific parameters (for example, whether a valve is "
            "normally-open or closed). The ability of the Kernel to work with these assets entirely depends on the "
            "user writing the custom module code to provide the necessary API."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=20)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kModuleCommandsReset"
        description = (
            "This is a special status invoked by the Kernel when it receives a module-addressed command code 0. "
            "Instead of triggering the usual command queue runtime, code 0 is interpreted as command reset state and "
            "the Kernel then clears all queued command(s) for that Module. The active Module command is allowed to "
            "complete gracefully, but there will be no further command execution until new commands are queued."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=21)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "kernel.status_codes.kStateSendingError"
        description = "The Kernel failed to send a StateMessage to the PC."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=22)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "kernel.status_codes.kResetModuleQueueTargetNotFound"
        description = (
            "Unable to find the addressee of the module (command) queue reset command sent from the PC. The "
            "module_type and module_id fields of the message did not match any of the custom Modules. Usually, this "
            "indicates a malformed message (user-error)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=23)
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

        section = "kernel.commands.kSetup"
        description = (
            "Carries out the necessary hardware and software parameter initialization. This command is executed "
            "automatically when the setup() method of main.cpp / .ino is executed."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=1)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = "kernel.commands.kReceiveData"
        description = (
            "Attempts to receive and parse the command and parameters data from the PC. This command is automatically "
            "triggered during controller runtime cycling, together with RunModuleCommands command. Note, this command "
            "is always triggered before RunModuleCommands."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=2)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = "kernel.commands.kResetController"
        description = (
            "Resets the hardware and software parameters of the Kernel and all managed modules. This command works "
            "similarly to kSetup (the logic is the same), but it is externally addressable (by the PC)."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=3)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = "kernel.commands.kIdentifyController"
        description = (
            "Transmits the unique ID of the controller that was hardcoded in the microcode firmware version running on "
            "the microcontroller. This command is used to verify the identity of the connected controller from the PC."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=4)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = "kernel.commands.kRunModuleCommands"
        description = (
            "Attempts to resolve (determine what to run) and execute a command for each of the managed custom modules. "
            "This command is automatically triggered during controller runtime cycling, together with ReceiveData "
            "command. Note, this command is always triggered after ReceiveData."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=5)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.addressable", value=False)

        return code_dictionary

    @staticmethod
    def _write_module_status_codes(code_dictionary: NestedDictionary) -> NestedDictionary:
        """Fills the module.status_codes section of the core_codes_map dictionary with data.

        Module classes directly control the hardware connected to the microcontroller. Their information is primarily
        mapped by custom_codes_map dictionary, which aggregates the codes that are unique for each custom Module type
        (family). However, since all custom Modules inherit from the same (base) Module class, some status codes used
        by custom modules come from the shared parent class. These shared status codes are added by this method, as
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

        section = "module.status_codes.kCommandAlreadyRunning"
        description = (
            "The Module already has an active command and does not need to activate a new command. This is an "
            "internal status intended for testing purposes."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=2)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kNewCommandActivated"
        description = (
            "The Module already has successfully activated a newly queued command. This is an internal status intended "
            "for testing purposes."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=3)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kRecurrentCommandActivated"
        description = (
            "The Module already has successfully (re)activated a recently completed command. This is an internal "
            "status intended for testing purposes."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=4)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kNoQueuedCommands"
        description = (
            "The Module is currently not running any command and does not have any queued commands to activate. "
            "This is an internal status intended for testing purposes."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=5)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kRecurrentTimerNotExpired"
        description = (
            "The Module's recurrent command activation timeout has not expired and there are no newly queued commands. "
            "Recently completed command (re)activation is disabled until the timeout expires. This is an internal "
            "status intended for testing purposes."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=6)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kNotImplemented"
        description = (
            "Critical! A custom Module class inheriting from the (base) Module class has not implemented (overloaded) "
            "one of the virtual API methods. These API methods are used by the Kernel to interface with custom modules "
            "and all API methods have to be overloaded (implemented) separately by each custom module class. This "
            "error indicates that the microcontroller firmware is not valid."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=7)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "module.status_codes.kParametersSet"
        description = (
            "New parameter-values addressed to the Module (custom module parameters) were received and applied "
            "successfully."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=8)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kSetupComplete"
        description = "Hardware and Software parameters of the Module have been successfully (re)set."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=9)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kModuleAssetsReset"
        description = "Custom assets of the Module have been (re)set to hardcoded defaults."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=10)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "module.status_codes.kStateSendingError"
        description = (
            "The Module failed to send a StateMessage to the PC. State messages work similar to Data messages, but "
            "they are used in cases where data objects do not need to be included with event-codes to optimize "
            "transmission."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=11)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "module.status_codes.kCommandCompleted"
        description = (
            "Indicates that the active command of the module has been completed. This status is reported whenever a "
            "command is replaced by a new command or terminates with no further queued or recurring commands."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=12)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

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
            "Failed to encode payload because payload size is too small. Valid payloads have to include at least one "
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
            "Failed to calculate the CRC checksum, because the size of the buffer that holds the packet is  too small."
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

        section = "transport_layer.status_codes.kPacketStartByteNotFoundError"
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
            "upper limit is due to COBS specifications). Encountering a payload size set to 255 or 0 would, therefore, "
            "trigger this error."
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
        section = "communication.status_codes.kCommunicationStandby"
        description = "Standby placeholder used to initialize the Communication class status tracker."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=151)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kCommunicationReceptionError"
        description = "Communication class ran into an error when receiving a message."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=152)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kCommunicationParsingError"
        description = "Communication class ran into an error when parsing (reading) a message."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=153)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kCommunicationPackingError"
        description = "Communication class ran into an error when writing a message to payload."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=154)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kCommunicationTransmissionError"
        description = "Communication class ran into an error when transmitting a message."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=155)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kCommunicationTransmitted"
        description = "Communication class successfully transmitted a message."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=156)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kCommunicationReceived"
        description = "Communication class successfully received a message."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=157)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kCommunicationInvalidProtocolError"
        description = "The received or transmitted protocol code is not valid for that type of operation."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=158)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kCommunicationNoBytesToReceive"
        description = "Communication class did not receive enough bytes to process the message. This is NOT an error."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=159)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kCommunicationParameterSizeMismatchError"
        description = "The number of extracted parameter bytes does not match the size of the input structure."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=160)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        section = "communication.status_codes.kCommunicationParametersExtracted"
        description = "Parameter data has been successfully extracted."
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=161)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=False)

        section = "communication.status_codes.kCommunicationParameterExtractionInvalidMessage"
        description = (
            "Unable to extract Module-addressed parameters, as the class currently holds a different message type in "
            "its reception buffer. Currently, only Module-addressed parameters need to be extracted by a separate "
            "method call. Calling the method for Kernel-addressed parameters (or any other message) will produce this "
            "error."
        )
        code_dictionary.write_nested_value(variable_path=f"{section}.code", value=162)
        code_dictionary.write_nested_value(variable_path=f"{section}.description", value=description)
        code_dictionary.write_nested_value(variable_path=f"{section}.error", value=True)

        return code_dictionary

    def runtime_cycle(self, command_queue: MPQueue, terminator_array: SharedMemoryArray) -> None:
        pass


print(MicroController.build_core_code_map())
