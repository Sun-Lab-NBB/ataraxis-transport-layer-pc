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
    prototypes,
)
from .microcontroller import ModuleInterface
from json import dumps


class TTLModule(ModuleInterface):
    """The class that exposes methods for interfacing with TTLModule instances running on Ataraxis MicroControllers.

    TTLModule facilitates exchanging Transistor-to-Transistor Logic (TTL) signals between various hardware systems, such
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
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kSendPulse"
        description = (
            "Pulses the managed digital pin by first setting it to HIGH and, after a configurable delay, re-setting it "
            "to LOW. The delay is specified by the pulse_duration class execution parameter and is given in "
            "microseconds. When this command is received from the PC, a single square pulse will be sent per each "
            "command activation."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kToggleOn"
        description = "Sets the managed digital pin to perpetually output the HIGH signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kToggleOff"
        description = "Sets the managed digital pin to perpetually output the LOW signal."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kCheckState"
        description = (
            "Checks the current state of the monitored digital input pin. If the state of the pin changed since the "
            "last check, sends the new pin state to the PC. If this command does not result in the new message sent to "
            "the PC, this means that the state ahs not changed. When this command is called for the first time, the "
            "initial pin state is always sent to the PC."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(4))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  TTLModule does not send Data messages, so no additional sections to add.

        return code_map

    def set_parameters(
        self, pulse_duration: np.uint32 = np.uint32(10000), averaging_pool_size: np.uint8 = np.uint8(0)
    ) -> ModuleParameters:
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
    """The class that exposes methods for interfacing with EncoderModule instances running on Ataraxis MicroControllers.

    EncoderModule is used to interface with quadrature encoders used to monitor the direction and magnitude of circular
    motion. To achieve the highest resolution, the module relies on hardware interrupt pins to detect and handle the
    pulses sent by the encoder. Overall, the EncoderModule is tasked with determining how much the tracked circular
    object has moved since the last check and to which direction.

    Notes:
        This interface comes pre-set to send incoming motion data to Unity. If you do not need this functionality,
        override the unity_output flag at class instantiation!

    Args:
        module_id: The unique byte identifier code of the managed EncoderModule instance.
        instance_name: The human-readable name of the EncoderModule instance.
        instance_description: A longer human-readable description of the EncoderModule instance.
        unity_output: Determines whether the EncoderModule instance should send the motion data to Unity.

    Attributes:
        _motion_topic: The MQTT topic to which the Encoder should send the motion data received from the
            microcontroller.
    """

    def __init__(
        self, module_id: np.uint8, instance_name: str, instance_description: str, unity_output: bool = True
    ) -> None:

        # Statically defines the module type description.
        type_description = (
            f"The family of modules that allows interfacing with a rotary encoder to track the direction and "
            f"displacement of circular motion."
        )

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="EncoderModule",
            module_type=np.uint8(2),
            type_description=type_description,
            module_id=module_id,
            instance_name=instance_name,
            instance_description=instance_description,
            unity_input_topics=None,
            unity_output=unity_output,
            queue_output=False,
        )

        self._motion_topic = "LinearTreadmill/Data"

    def send_to_unity(self, message: ModuleData, unity_communication: UnityCommunication) -> None:
        # If the incoming message is not a CCW or CW motion report, aborts processing
        if message.event != np.uint8(51) or message.event != np.uint8(52):
            return

        # The motion sign is encoded via the message event code. CW motion (code 51) is interpreted as negative motion
        # and CCW as positive.
        sign = -1 if message.event == np.uint8(51) else 1

        # Translates the absolute motion into the CW / CCW vector
        signed_motion = int(message.data_object) * sign

        # Encodes the motion data into the format expected by the GIMBL Unity code.
        json_string = dumps(obj=signed_motion)
        byte_array = json_string.encode("utf-8")

        # Publishes the motion to the appropriate MQTT topic.
        unity_communication.send_data(topic=self._motion_topic, payload=byte_array)

    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        # Status Codes
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kCCWDelta"
        description = (
            "The monitored encoder has moved in the Counter-Clockwise (CCW) direction relative to the last encoder "
            "readout or module class initialization."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kCWDelta"
        description = (
            "The monitored encoder has moved in the Clockwise (CW) direction relative to the last encoder readout or "
            "module class initialization."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(52))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        # Commands
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kReadEncoder"
        description = (
            "Reads the number and direction of pulses registered by the encoder since the last readout or class reset, "
            "whichever is more recent. If the obtained number is greater than the minimum reporting threshold and "
            "reporting displacement in the recorded direction (CW or CCW) is enabled, sends the absolute displacement "
            "value and direction to the PC."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kResetEncoder"
        description = (
            "Resets the pulse tracker to 0 without reading its current value. Note, kReadEncoder command also resets "
            "the encoder tracker, so this command should generally only be called when there is a need to reset the "
            "tracker without reading its value."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  Data Objects
        module_section = f"{self.type_name}_module.data_objects"

        section = f"{module_section}.kCWMovementObject"
        description_1 = (
            "The number of pulses by which the encoder has moved in the Clockwise (CW) direction, relative to last "
            "encoder readout or reset."
        )
        code_map.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedLong)
        code_map.write_nested_value(variable_path=f"{section}.names", value=("movement_pulse_count",))
        code_map.write_nested_value(variable_path=f"{section}.descriptions", value=(description_1,))

        section = f"{module_section}.kCCWMovementObject"
        description_1 = (
            "The number of pulses by which the encoder has moved in the Counter-Clockwise (CCW) direction, relative to "
            "last encoder readout or reset."
        )
        code_map.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(52))
        code_map.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedLong)
        code_map.write_nested_value(variable_path=f"{section}.names", value=("movement_pulse_count",))
        code_map.write_nested_value(variable_path=f"{section}.descriptions", value=(description_1,))

        return code_map

    def set_parameters(
        self, report_ccw: bool = True, report_cw: bool = True, delta_threshold: np.uint32 = np.uint32(1)
    ) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Args:
            report_ccw: Determines whether to report motion in the CCW (positive) direction.
            report_cw: Determines whether to report motion in the CW (negative) direction.
            delta_threshold: The minimum number of pulses required for the motion to be reported. Depending on encoder
                resolution, this allows setting the 'minimum distance' threshold for reporting. Note, if the change is
                0 (the encoder readout did not change), it will not be reported, regardless of the delta_threshold.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(report_ccw, report_cw, delta_threshold),
        )

    def reset_encoder(self) -> OneOffModuleCommand:
        """Resets the current pulse tracker of the encoder to 0, clearing any currently stored motion data."""
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2),
            noblock=False,
        )

    def read_encoder(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the number and direction of pulses recorded by the managed encoder class since the last readout or
        module class reset.

        If there has been a significant change in the tracker direction, reports the change and direction to the PC. It
        is highly advised to issue this command as recurrent to continuously monitor the encoder motion, rather than
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
                command=np.uint8(1),
                noblock=False,
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=False,
            cycle_delay=repetition_delay,
        )


class BreakModule(ModuleInterface):
    """The class that exposes methods for interfacing with BreakModule instances running on Ataraxis MicroControllers.

    BreakModule is used to interface with various break systems. Overall, the class is designed to be connected to a
    Field-Effect Transistor (FET) gated relay that controls the delivery of voltage to the break system. Therefore, the
    class can be used to either enable or disable the breaks or to output a PWM_modulated signal to variably adjust the
    breaking power.

    Args:
        module_id: The unique byte identifier code of the managed BreakModule instance.
        instance_name: The human-readable name of the BreakModule instance.
        instance_description: A longer human-readable description of the BreakModule instance.
    """

    def __init__(self, module_id: np.uint8, instance_name: str, instance_description: str) -> None:

        # Statically defines the TTLModule type description.
        type_description = (
            f"The family of modules that sends Pulse-Width Modulated (PWM) signals to variably engage the connected "
            f"breaking system."
        )

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="BreakModule",
            module_type=np.uint8(3),
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
        description = "Unable to output the requested digital or analog signal, as the global Action lock is enabled."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=True)

        # Commands
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kEnable"
        description = (
            "Sets the output pin to perpetually output the necessary HIGH or LOW signal to permanently engage the "
            "breaks. The output signal depends on whether the break is normally engaged."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kDisable"
        description = (
            "Sets the output pin to perpetually output the necessary HIGH or LOW signal to permanently disengage the "
            "breaks. The output signal depends on whether the break is normally engaged."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kSetPower"
        description = (
            "Sets the output pin to perpetually output a square wave pulse modulated by the PWM value defined by "
            "the custom parameters structure of the instance. The PWM value controls the amount of time the break "
            "is engaged, which, in turn, variably controls the overall breaking power."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  BreakModule does not send Data messages, so no additional sections to add.

        return code_map

    def set_parameters(self, pwm_strength: np.uint8 = np.uint8(255)) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Args:
            pwm_strength: The Pulse-Width Modulation (PWM) value to use when the module is triggered to deliver variable
                breaking power. Depending on this value, the breaking power can be adjusted from none (when 0) to
                maximum (when 255). This is only used when the break is triggered using set_power() command.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(pwm_strength,),
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Sets the pin connected to the break system to continuously output the necessary signal to engage or disengage
        the break.

        Args:
            state: The signal to output. Set to True to permanently enable the break with maximum power, set to False to
                permanently disable the break.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1 if state else 2),
            noblock=False,
        )

    def set_power(self) -> OneOffModuleCommand:
        """Sets the pin connected to the break system to continuously engage the break with the pwm specified by module
        instance pwm_strength parameter.

        This engages the break with the relative strength that depends on the pwm duty cycle value. This command only
        activates the variable breaking strength mode, to adjusty the power send the updated pwm_strength value as
        part of the ModuleParameters message (via set_parameters() command).

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=False,
        )


class SensorModule(ModuleInterface):
    """The class that exposes methods for interfacing with SensorModule instances running on Ataraxis MicroControllers.

    SensorModule facilitates receiving data recorded by hardware sensors that output analog voltage signal(s), such as
    voltage or current sensors. Overall, any sensor designed to output a single unipolar non-binary voltage signal
    can be monitored using SensorModule instance. This class functions similar to the TTLModule input mode, but is
    uniquely specialized to working with analog data. If you need to monitor a binary digital signal, use TTLModule
    instead for higher efficiency.

    Notes:
        This interface comes pre-set to send triggers to Unity when it receives sensor data. If you do not need this
        functionality, override the unity_output flag at class instantiation!

    Args:
        module_id: The unique byte identifier code of the managed SensorModule instance.
        instance_name: The human-readable name of the SensorModule instance.
        instance_description: A longer human-readable description of the SensorModule instance.
        unity_output: Determines whether the SensorModule instance should send the motion data to Unity.

    Attributes:
        _sensor_topic: The MQTT topic to which the Sensor should send the triggers based on the sensor data received =
            from the microcontroller.
    """

    def __init__(
        self, module_id: np.uint8, instance_name: str, instance_description: str, unity_output: bool = True
    ) -> None:

        # Statically defines the TTLModule type description.
        type_description = (
            f"The family of modules that receives unidirectional analog signals from connected sensor hardware."
        )

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="SensorModule",
            module_type=np.uint8(4),
            type_description=type_description,
            module_id=module_id,
            instance_name=instance_name,
            instance_description=instance_description,
            unity_input_topics=None,
            unity_output=unity_output,
            queue_output=False,
        )

        self._sensor_topic: str = f"LickPort/"

    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        # If the incoming message is not a signal value report, aborts processing
        if message.event != np.uint8(51):
            return

        # Sends an empty message to the sensor MQTT topic, which acts as a binary trigger message.
        unity_communication.send_data(topic=self._sensor_topic, payload=None)

    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        # Status Codes
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kInput"
        description = (
            "The monitored analog pin has detected a signal that is within the reporting threshold and is "
            "significantly different from the signal detected during a previous significant readout."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        # Commands
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kCheckState"
        description = (
            "Checks the signal received by the monitored analog input pin. If the signal is within the threshold "
            "boundaries and is significantly different from the signal detected during a previous significant readout, "
            "sends detected signal value to the PC. The signal value is always sent to the PC when it is checked the "
            "first time after class initialization."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  Data Objects
        module_section = f"{self.type_name}_module.data_objects"

        section = f"{module_section}.kInputObject"
        description_1 = (
            "The raw analog value that specifies the level of the input signal detected by the monitored analog pin."
        )
        code_map.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedShort)
        code_map.write_nested_value(variable_path=f"{section}.names", value=("sensor_signal",))
        code_map.write_nested_value(variable_path=f"{section}.descriptions", value=(description_1,))

        return code_map

    def set_parameters(
        self,
        lower_threshold: np.uint16 = np.uint16(0),
        upper_threshold: np.uint16 = np.uint16(65535),
        delta_threshold: np.uint16 = np.uint16(1),
        averaging_pool_size: np.uint8 = np.uint8(0),
    ) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Args:
            lower_threshold: The minimum strength of the signal, in raw analog units of the sensor pin, to be considered
                valid. Setting this threshold to a number above zero allows high-pass filtering the incoming signals.
                Note, the threshold is inclusive.
            upper_threshold: The maximum strength of the signal, in raw analog units of the sensor pin, to be considered
                valid. Setting this threshold to a number below 65535 allows low-pass filtering the incoming signals.
                Note, the threshold is inclusive, and due to the typically used analog resolution of 10-12 bytes, the
                realistic ceiling of detected signals will likely not exceed 1000-5000 analog units.
            delta_threshold: The minimum difference between the current and previous levels of the signal received by
                the sensor pin to be reported to the PC. This allows filtering out sensor pin noise. Note, the
                threshold is inclusive and if the delta is 0, the signal will not be reported to the PC, regardless of
                the delta_threshold value.
            averaging_pool_size: The number of analog pin readouts to average together when checking pin state. This
                is used by the check_state() command to smooth the pin readout. It is highly advised to have this
                enabled and set to at least 10 readouts.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(upper_threshold, lower_threshold, delta_threshold, averaging_pool_size),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the signal received by the analog pin managed by the module, and if it is different from the last
        checked detected signal, notifies the PC.

        It is highly advised to issue this command as recurrent to continuously monitor the received signal, rather than
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
                command=np.uint8(1),
                noblock=False,
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=False,
            cycle_delay=repetition_delay,
        )
