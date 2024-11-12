from multiprocessing import Queue as MPQueue

import numpy as np
from ataraxis_data_structures import NestedDictionary

from .communication import (
    ModuleData,
    ModuleState,
    ModuleParameters,
    OneOffModuleCommand,
    DequeueModuleCommand,
    RepeatedModuleCommand,
    UnityCommunication,
)
from .microcontroller import ModuleInterface


class TTLModule(ModuleInterface):
    def __init__(self, module_id: np.uint8, instance_name: str, instance_description: str):

        type_description = (
            f"Sends or receives Transistor-to-Transistor Logic (TTL) signals using the specified digital pin."
        )

        # Call parent's __init__ first
        super().__init__(
            type_name="TTLModule",
            module_type=np.uint8(1),
            type_description=type_description,
            module_id=module_id,
            instance_name=instance_name,
            instance_description=instance_description,
            unity_output=False,
            unity_input_topics=None,
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

        section = f"{module_section}.kOutputOn"
        description = "The managed digital pin has been set to output the HIGH signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kOutputOff"
        description = "The managed digital pin has been set to output the LOW signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(52))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kInputOn"
        description = "The monitored digital pin detects a HIGH incoming signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(53))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kInputOff"
        description = "The monitored digital pin detects a LOW incoming signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(54))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kOutputLocked"
        description = "Unable to output the requested digital signal, as the global TTL lock is enabled."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(55))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=True)

        section = f"{module_section}.kInvalidPinMode"
        description = (
            "The requested command is not valid for the managed digital pin mode. This error would be triggered if the "
            "module that manages an output pin receives a command to check the input pin state. Similarly, this would "
            "be triggered if the module that monitors an input pin receives a command to output a TTL signal."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(56))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=True)

        # Commands
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kSendPulse"
        description = (
            "Attempts to receive and parse the command and parameters data sent from the PC. This command is "
            "automatically triggered at the beginning of each controller runtime cycle. Note, this command "
            "is always triggered before running any queued or newly received module commands."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=False)


class EncoderModule(ModuleInterface):
    def __init__(self, module_type: np.uint8, module_id: np.uint8):
        pass
