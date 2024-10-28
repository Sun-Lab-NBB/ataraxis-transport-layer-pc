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

    def runtime_cycle(self, command_queue: MPQueue, terminator_array: SharedMemoryArray) -> None:
        pass


@dataclass(frozen=True)
class Module:
    module_type: np.uint8
    module_id: np.uint8
    data_prototypes: NestedDictionary
    commands: NestedDictionary

    def get_prototype(self, message: DataMessage) -> Optional[Any]:
        if not message.module_type == self.module_type or not message.module_id == self.module_id:
            return None

        return self.data_prototypes.read_nested_value(variable_path=(message.command.item(), message.event.item()))

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
        pass
