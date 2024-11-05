from .microcontroller import ModuleInterface
import numpy as np
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


class EncoderModule(ModuleInterface):
    def __init__(self, module_type: np.uint8, module_id: np.uint8):
        # Call parent's __init__ first
        super().__init__(type_name="Encoder", module_type=module_type, module_id=module_id)

        # Yes.

    def process_data(self, message: ModuleData | ModuleState):
        pass

    def make_command_message(
        self,
        command_code: np.uint8,
        noblock: np.bool,
        return_code: np.uint8 = np.uint8(0),
        recurrent_delay: np.uint32 = np.uint32(0),
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        """Creates a Repeated or OneOff ModuleCommand message object.

        This method can be sued to create Module-addressed command objects. These objects can then be passed to the
        appropriate MicroControllerInterface class instance for them to be sent to and executed by the microcontroller
        that manages the module.

        Notes:
            The type of the created object determines on the value of the recurrent_delay argument. When it is set to
            0, a OneOffModuleCommand is created. When it is set to a non-zero value, a RepeatedModuleCommand is created.

        Args:
            command_code: The byte-code of the command to execute.
            noblock: Whether the message should be sent in a non-blocking manner.
            return_code: The byte-code of the return code for the command. This is optional and defaults to 0.
            recurrent_delay: The delay in milliseconds between sending repeated ModuleCommand messages. This is optional
                and defaults to 0.
        """
        if recurrent_delay != 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                command=command_code,
                noblock=noblock,
                return_code=return_code,
            )
        else:
            return RepeatedModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                command=command_code,
                noblock=noblock,
                return_code=return_code,
                cycle_delay=recurrent_delay,
            )

    def make_deque_message(self, return_code: np.uint8 = np.uint8(0)) -> DequeueModuleCommand:
        return DequeueModuleCommand(module_type=self._module_type, module_id=self._module_id, return_code=return_code)

    def make_parameter_message(
        self,
        parameter_values: tuple[np.unsignedinteger, np.signedinteger, np.floating, np.bool],
        return_code: np.uint8 = np.uint8(0),
    ) -> ModuleParameters:
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            parameter_data=parameter_values,
            return_code=return_code,
        )
