from multiprocessing import Queue as MPQueue

import numpy as np
from ataraxis_data_structures import NestedDictionary

from .communication import (
    ModuleData,
    ModuleState,
    ModuleParameters,
    OneOffModuleCommand,
    RepeatedModuleCommand,
    UnityCommunication,
)
from .microcontroller import ModuleInterface


class TTLModule(ModuleInterface):
    """The class that exposes methods for interfacing with TTLModule instances running on Ataraxis MicroControllers.

    TTLModule facilitate exchanging Transistor-to-Transistor Logic (TTL) signals between various hardware systems, such
    as microcontrollers, cameras and other microchip-bundled hardware. The module contains methods for both sending and
    receiving the TTL pulses, and all are accessible through the interaction of this specific interface implementation
    and the MicroControllerInterface class that manages the microcontroller.

    Args:
        module_id: The unique byte identifier code of the managed TTLModule instance.
        instance_name: The human-readable name of the TTLModule instance.
        instance_description: A longer human-readable description of the TTLModule instance.
    """

    def __init__(self, module_id: np.uint8, instance_name: str, instance_description: str) -> None:

        # Statically defines the TTLModule type description.
        type_description = (
            f"The family of modules that sends or receives Transistor-to-Transistor Logic (TTL) signals using the "
            f"specified digital pin."
        )

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="TTLModule",
            module_type=np.uint8(1),
            type_description=type_description,
            module_id=module_id,
            instance_name=instance_name,
            instance_description=instance_description,
            unity_input_topics=None,
            unity_output=False,
            queue_output=False,
        )

    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        """Not used."""
        return

    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        # Status Codes
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kOutputLocked"
        description = "Unable to output the requested digital signal, as the global TTL lock is enabled."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=True)

        section = f"{module_section}.kInputOn"
        description = "The monitored digital pin detects a HIGH incoming signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(52))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kInputOff"
        description = "The monitored digital pin detects a LOW incoming signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(53))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kInvalidPinMode"
        description = (
            "The requested command is not valid for the managed digital pin mode. This error would be triggered if the "
            "module that manages an output pin receives a command to check the input pin state. Similarly, this would "
            "be triggered if the module that monitors an input pin receives a command to output a TTL signal."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(54))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=True)

        # Commands
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kSendPulse"
        description = (
            "Pulses the managed digital pin by first setting it to HIGH and, after a configurable delay, re-setting it "
            "to LOW. The delay is specified by the pulse_duration class execution parameter and is given in "
            "microseconds. When this command is received from the PC, a single square pulse will be sent per each "
            "command activation."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = f"{module_section}.kToggleOn"
        description = "Sets the managed digital pin to perpetually output the HIGH signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = f"{module_section}.kToggleOff"
        description = "Sets the managed digital pin to perpetually output the LOW signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=False)

        section = f"{module_section}.kCheckState"
        description = (
            "Checks the current state of the monitored digital input pin. If the state of the pin changed since the "
            "last check, sends the new pin state to the PC. If this command does not result in the new message sent to "
            "the PC, this means that the state ahs not changed. When this command is called for the first time, the "
            "initial pin state is always sent to the PC."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(4))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=False)

        #  TTLModule does not send Data messages, so no additional sections to add.

        return code_map

    def set_parameters(self, pulse_duration: np.uint32, averaging_pool_size: np.uint8) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Args:
            pulse_duration: The time, in microseconds, for the HIGH phase of emitted TTL pulses. This is used by the
                send_pulse() command to control the duration of emitted pulses.
            averaging_pool_size: The number of digital pin readouts to average together when checking pin state. This
                is used by the check_state() command to smooth the pin readout. This should be disabled for most use
                cases.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(pulse_duration, averaging_pool_size),
        )

    def send_pulse(
        self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = True
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        """Sends a one-off or recurrent (repeating) TTL pulse using the digital pin managed by the module instance.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If this is set to 0,
                the command will only run once.
            noblock: Determines whether the command should block the microcontroller while emitting the high phase of
                the pulse or not. Blocking ensures precise pulse duration, non-blocking allows the microcontroller to
                perform other operations while waiting, increasing its throughput.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message to be sent to the microcontroller.
        """

        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=noblock,
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=noblock,
            cycle_delay=repetition_delay,
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Sets the digital pin managed by the module instance to continuously output the desired signal.

        Args:
            state: The signal to output. Set to True for HIGH and False for LOW.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=False,
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the state of the digital pin managed by the module, and if it is different from the last checked
        state, notifies the PC.

        It is highly advised to issue this command as recurrent to continuously monitor the pin state, rather than
        repeatedly calling it as a one-off command for best runtime efficiency.

         Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If this is set to 0,
                the command will only run once.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message to be sent to the microcontroller.
        """
        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(4),
                noblock=False,
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=False,
            cycle_delay=repetition_delay,
        )


class EncoderModule(ModuleInterface):
    def __init__(self, module_type: np.uint8, module_id: np.uint8):
        pass
