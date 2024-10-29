import numpy as np
from ataraxis_data_structures import NestedDictionary, SharedMemoryArray
from dataclasses import dataclass
from .communication import DataMessage
from typing import Any, Optional
import multiprocessing
from multiprocessing import (
    Queue as MPQueue,
    Process,
    ProcessError,
)
from multiprocessing.managers import SyncManager


class MicroController:
    def __init__(self, name, serial_port, baud_rate, id_code, modules):

        self._name: str = name

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

        self._core_module_status_codes = {
            "kStandBy": 0,  # The code used to initialize the module_status variable.
            "kDataSendingError": 1,  # An error has occurred when sending Data to the connected Ataraxis system.
            "kCommandAlreadyRunning": 2,  # The module cannot activate new commands as one is already running.
            "kNewCommandActivated": 3,  # The module has successfully activated a new command.
            "kRecurrentCommandActivated": 4,  # The module has successfully activated a recurrent command.
            "kNoQueuedCommands": 5,  # The module does not have any new or recurrent commands to activate.
            "kRecurrentTimerNotExpired": 6,  # The module's recurrent activation timeout has not expired yet
            "kNotImplemented": 7,  # Derived class has not implemented the called virtual method
            "kParametersSet": 8,  # The custom parameters of the module were overwritten with PC-sent data.
            "kSetupComplete": 9,  # Module hardware was successfully configured.
            "kModuleAssetsReset": 10,  # All custom assets of the module ahs been reset.
        }

        self._kernel_status_codes = {
            "kStandby": 0,  # Standby code used during class initialization.
            "kSetupComplete": 1,  # Setup() method runtime succeeded.
            "kModuleSetupError": 2,  # Setup() method runtime failed due to a module setup error.
            "kNoDataToReceive": 3,  # ReceiveData() method succeeded without receiving any data.
            "kDataReceptionError": 4,  # ReceiveData() method failed due to a data reception error.
            "kDataSendingError": 5,  # SendData() method failed due to a data sending error.
            "kInvalidDataProtocol": 6,  # Received message uses an unsupported (unknown) protocol.
            "kKernelParametersSet": 7,  # Received and applied the parameters addressed to the Kernel class.
            "kKernelParametersError": 8,  # Unable to apply the received Kernel parameters.
            "kModuleParametersSet": 9,  # Received and applied the parameters addressed to a managed Module class.
            "kModuleParametersError": 10,  # Unable to apply the received Module parameters.
            "kParametersTargetNotFound": 11,  # The addressee of the parameters' message could not be found.
            "kControllerReset": 12,  # The Kernel has reset the controller's software and hardware states.
            "kKernelCommandUnknown": 13,  # The Kernel has received an unknown command.
            "kModuleCommandQueued": 14,  # The received command was queued to be executed by the Module.
            "kCommandTargetNotFound": 15,  # The addressee of the command message could not be found.
            "kServiceSendingError": 16,  # Error sending a service message to the connected system.
            "kModuleResetError": 17,  # Unable to reset the custom assets of a module.
            "kControllerIDSent": 18,  # The requested controller ID was sent ot to the connected system.
            "kModuleCommandError": 19,  # Error executing an active module command.
            "kModuleCommandsCompleted": 20,  # All active module commands have been executed.
            "kModuleAssetResetError": 21,  # Resetting custom assets of a module failed.
            "kModuleCommandsReset": 22,  # Module command structure has been reset. Queued commands cleared.
        }

        self._kernel_commands = {
            "kStandby": 0,  # Standby code used during class initialization. Not externally addressable.
            "kSetup": 1,  # Module setup command. Not externally addressable.
            "kReceiveData": 2,  # Receive data command. Not externally addressable.
            "kResetController": 3,  # Resets the software and hardware state of all modules. Externally addressable.
            "kIdentifyController": 4,  # Transmits the ID of the controller back to caller. Externally addressable.
            "kRunModuleCommands": 5,  # Executes active module commands. Not externally addressable.
        }

    def runtime_cycle(self, command_queue: MPQueue, terminator_array: SharedMemoryArray) -> None:
        pass


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

    def make_parameters(
        self,
        return_code: np.uint8 = 0,
    ) -> None:
        pass  # TODO Virtual (not implemented error)

    def process_data(self, message: DataMessage):
        pass  # TODO Virtual (not implemented error)
