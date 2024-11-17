"""This module provides ModuleInterface implementations for teh default modules shipped with the
AtaraxisMicroController library. Primarily, they are included to showcase the correct usage of this library.
"""

from json import dumps
from multiprocessing import Queue as MPQueue

import numpy as np
from ataraxis_data_structures import NestedDictionary

from .communication import (
    ModuleData,
    ModuleState,
    ModuleParameters,
    UnityCommunication,
    OneOffModuleCommand,
    RepeatedModuleCommand,
    prototypes,
)
from .microcontroller import ModuleInterface


class TTLModule(ModuleInterface):
    """The class that exposes methods for interfacing with TTLModule instances running on Ataraxis MicroControllers.

    TTLModule facilitates exchanging Transistor-to-Transistor Logic (TTL) signals between various hardware systems, such
    as microcontrollers, cameras and other hardware. The module contains methods for both sending and receiving the TTL
    pulses, but each TTLModule instance can only do one of these functions at a time.

    Args:
        module_id: The unique identifier code of the managed TTLModule instance.
        instance_name: The human-readable name of the TTLModule instance.
        instance_description: A longer human-readable description of the TTLModule instance.
    """

    def __init__(self, module_id: np.uint8, instance_name: str, instance_description: str) -> None:
        # Statically defines the TTLModule type description.
        type_description = (
            f"The family of modules that sends or receives Transistor-to-Transistor Logic (TTL) signals using the "
            f"managed digital input or output pin."
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

    def send_to_queue(
        self,
        message: ModuleData | ModuleState,
        queue: MPQueue,  # type: ignore
    ) -> None:
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
            "last check, sends the new pin state to the PC. When this command is called for the first time, the "
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
        """Sets PC-addressable runtime parameters of the module instance running on the microcontroller.

        Args:
            pulse_duration: The time, in microseconds, for the HIGH phase of emitted TTL pulses. This is used during the
                execution of send_pulse() command to control the length of emitted pulses.
            averaging_pool_size: The number of digital pin readouts to average together when checking pin state. This
                is used during the execution of check_state() command to smooth the pin readout. This should be disabled
                for most use cases as digital logic signals are already comparatively jitter-free.

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

        Generally, this command is well-suited to carry out most forms of TTL communication. It is, however, adapted for
        comparatively low-frequency communication of 10-200 Hz, compared to PWM outputs capable of mHz or even Khz pulse
        oscillation frequencies.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If this is set to 0,
                the command will only run once. Note, the exact repetition delay will be further affected by other
                modules managed by the same microcontroller and may not be perfectly accurate.
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
                noblock=np.bool(noblock),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(noblock),
            cycle_delay=repetition_delay,
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Sets the digital pin managed by the module instance to continuously output the desired signal level.

        Use this to permanently activate or inactivate the pin. Since this is a lock-in command, it does not make sense
        to make it recurrent.

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
            noblock=np.bool(False),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the state of the digital pin managed by the module.

        This command will evaluate the state of the digital pin and, if it is significantly different from the state
        recorded during a previous check, report it to the PC. This approach ensures that only significant changes are
        communicated to the PC, preserving communication bandwidth. It is highly advised to issue this command to
        repeat (recur) at a desired interval to continuously monitor the pin state, rather than repeatedly calling it
        as a one-off command for best runtime efficiency.

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
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )


class EncoderModule(ModuleInterface):
    """The class that exposes methods for interfacing with EncoderModule instances running on Ataraxis MicroControllers.

    EncoderModule allows interfacing with quadrature encoders used to monitor the direction and magnitude of connected
    object's rotation. To achieve the highest resolution, the module relies on hardware interrupt pins to detect and
    handle the pulses sent by the two encoder channels. Overall, the EncoderModule is tasked with determining how much
    the tracked rotating object has moved since the last check and to which direction.

    Notes:
        This interface comes pre-configured to send incoming motion data to Unity. If you do not need this
        functionality, override the unity_output flag at class instantiation!

    Args:
        module_id: The unique identifier code of the managed EncoderModule instance.
        instance_name: The human-readable name of the EncoderModule instance.
        instance_description: A longer human-readable description of the EncoderModule instance.
        unity_output: Determines whether the EncoderModule instance should send the motion data to Unity.
        motion_topic: The MQTT topic to which the instance should send the motion data received from the
            microcontroller.
        encoder_ppr: The resolution of the managed quadrature encoder, in Pulses per Revolution (PPR). Specifically,
            this is the number of quadrature pulses the encoder emits per full 360-degree rotation. If this number is
            not known, provide a 'dummy' value and use get_ppr() command to estimate the PPR using the index channel
            of the encoder.

    Attributes:
        _motion_topic: Stores the MQTT motion topic.
        _ppr: Stores the resolution of the managed quadrature encoder.
    """

    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        instance_description: str,
        unity_output: bool = True,
        motion_topic: str = "LinearTreadmill/Data",
        encoder_ppr: int = 2048,
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

        # Saves additional data to class attributes.
        self._motion_topic = motion_topic
        self._ppr = encoder_ppr

    def send_to_unity(self, message: ModuleState | ModuleData, unity_communication: UnityCommunication) -> None:
        # If the incoming message is not a CCW or CW motion report, aborts processing
        if message.event != np.uint8(51) and message.event != np.uint8(52):
            return

        # The rotation direction is encoded via the message event code. CW rotation (code 51) is interpreted as negative
        # and CCW as positive.
        sign = 1 if message.event == np.uint8(51) else -1

        # Translates the absolute motion into the CW / CCW vector and converts from raw pulse count to degrees using the
        # PPR.
        signed_motion = (float(message.data_object) / self._ppr) * sign  # type: ignore

        # Encodes the motion data into the format expected by the GIMBL Unity module and serializes it into a
        # byte-string.
        json_string = dumps(obj={"movement": signed_motion})
        byte_array = json_string.encode("utf-8")

        # Publishes the motion to the appropriate MQTT topic.
        unity_communication.send_data(topic=self._motion_topic, payload=byte_array)

    def send_to_queue(
        self,
        message: ModuleData | ModuleState,
        queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        # Status Codes
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kRotatedCCW"
        description = (
            "The monitored encoder has rotated in the Counter-Clockwise (CCW) direction relative to the last encoder "
            "readout or module class initialization. CCW rotation is interpreted as positive rotation."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kRotatedCW"
        description = (
            "The monitored encoder has rotated in the Clockwise (CW) direction relative to the last encoder readout or "
            "module class initialization. CW rotation is interpreted as negative rotation."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(52))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        section = f"{module_section}.kPPR"
        description = "The encoder has finished estimating the Pulse-Per-Revolution (PPR) resolution of the encoder."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(53))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        # Commands
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kCheckState"
        description = (
            "Reads the number and direction of pulses registered by the encoder since the last readout or class reset, "
            "whichever is more recent. If the absolute obtained number is greater than the minimum reporting threshold "
            "and reporting rotation in the recorded direction (CW or CCW) is enabled, sends the absolute displacement "
            "value and direction to the PC."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kReset"
        description = (
            "Resets the pulse tracker to 0 without reading its current value. Note, kCheckState command also resets "
            "the encoder tracker, so this command should generally only be called when there is a need to reset the "
            "tracker without reading its value."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kGetPPR"
        description = (
            "Estimates the Pulse-Per-Revolution (PPR) resolution of the encoder over up to 11 full revolutions."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  Data Objects
        module_section = f"{self.type_name}_module.data_objects"

        section = f"{module_section}.kRotatedCWObject"
        description_1 = (
            "The number of pulses by which the encoder has moved in the Clockwise (CW) direction, relative to last "
            "encoder readout or reset."
        )
        code_map.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedLong)
        code_map.write_nested_value(variable_path=f"{section}.names", value=("Pulses",))
        code_map.write_nested_value(variable_path=f"{section}.descriptions", value=(description_1,))

        section = f"{module_section}.kRotatedCCWObject"
        description_1 = (
            "The number of pulses by which the encoder has moved in the Counter-Clockwise (CCW) direction, relative to "
            "last encoder readout or reset."
        )
        code_map.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(52))
        code_map.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedLong)
        code_map.write_nested_value(variable_path=f"{section}.names", value=("Pulses",))
        code_map.write_nested_value(variable_path=f"{section}.descriptions", value=(description_1,))

        section = f"{module_section}.kPPRObject"
        description_1 = (
            "The estimated number of pulses the encoder emits per one full revolution of the tracked object."
        )
        code_map.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(53))
        code_map.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedShort)
        code_map.write_nested_value(variable_path=f"{section}.names", value=("Pulses",))
        code_map.write_nested_value(variable_path=f"{section}.descriptions", value=(description_1,))

        return code_map

    def set_parameters(
        self,
        report_ccw: np.bool = np.bool(True),
        report_cw: np.bool = np.bool(True),
        delta_threshold: np.uint32 = np.uint32(1),
    ) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Args:
            report_ccw: Determines whether to report rotation in the CCW (positive) direction.
            report_cw: Determines whether to report rotation in the CW (negative) direction.
            delta_threshold: The minimum number of pulses required for the motion to be reported. Depending on encoder
                resolution, this allows setting the 'minimum rotation distance' threshold for reporting. Note, if the
                change is 0 (the encoder readout did not change), it will not be reported, regardless of the
                value of this parameter.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(report_ccw, report_cw, delta_threshold),
        )

    def check_state(self, repetition_delay: np.uint32 = np.uint32(0)) -> OneOffModuleCommand | RepeatedModuleCommand:
        """Checks the number and direction of pulses recorded by the encoder since the last readout or module reset.

        If there has been a significant change in the absolute count of pulses, reports the change and direction to the
        PC. It is highly advised to issue this command to repeat (recur) at a desired interval to continuously monitor
        the pin state, rather than repeatedly calling it as a one-off command for best runtime efficiency.

        This command allows continuously monitoring the rotation of the object connected to the encoder. It is designed
        to return the absolute raw count of pulses emitted by the encoder in response to the object ration. This allows
        avoiding floating-point arithmetic on the microcontroller, but requires it to be implemented on the PC to
        convert emitted pulses into a meaningful circular distance value. The specific conversion algorithm depends on
        the encoder and the tracker object.

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
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )

    def reset_pulse_count(self) -> OneOffModuleCommand:
        """Resets the current pulse tracker of the encoder to 0, clearing any currently stored rotation data.

        Primarily, this command is helpful if you need to reset the encoder without evaluating its current pulse count.
        For example, this would be the case if there is a delay between the initialization of the module and the start
        of the encoder monitoring. Resetting the encoder before evaluating its pulse count for the first time discards
        a nonsensical pulse count aggregated before the monitoring has started. Similarly, this can be used to re-base
        the encoder pulse count without reading the aggregate data.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2),
            noblock=np.bool(False),
        )

    def get_ppr(self) -> OneOffModuleCommand:
        """Uses the index channel of the encoder to estimate its Pulse-per-Revolution (PPR).

        The PPR allows converting raw pulse counts reported by other commands of this module to real life circular
        distances. This is a service command not intended to be used during most production runtimes if the PPR is
        already known. It relies on the encoder completing up to 11 full rotations and uses the index channel of the
        encoder to detect each revolution completion.

        Notes:
            Make sure the calibrated encoder revolves at a steady slow speed until this command completes. Similar to
            other service commands, it is designed to deadlock the controller until the command completes.

            The direction of the rotation is not relevant for this command, as long as it make the full 360-degree
            revolution.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )


class BreakModule(ModuleInterface):
    """The class that exposes methods for interfacing with BreakModule instances running on Ataraxis MicroControllers.

    BreakModule allows interfacing with a wide range of breaks attached to rotating objects. To enable this, the class
    is designed to be connected to a Field-Effect-Transistor (FET) gated relay that controls the delivery of voltage to
    the break. The module can be used to either fully engage or disengage the breaks or to output a PWM signal to
    engage the break with the desired strength.

    Args:
        module_id: The unique identifier code of the managed BreakModule instance.
        instance_name: The human-readable name of the BreakModule instance.
        instance_description: A longer human-readable description of the BreakModule instance.
    """

    def __init__(self, module_id: np.uint8, instance_name: str, instance_description: str) -> None:
        # Statically defines the TTLModule type description.
        type_description = (
            f"The family of modules that sends digital or analog Pulse-Width-Modulated (PWM) signals to engage the "
            f"managed break with the desirable strength."
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

    def send_to_queue(
        self,
        message: ModuleData | ModuleState,
        queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        # Status Codes
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kOutputLocked"
        description = (
            "Unable to sends the requested digital or analog signal to the break, as the global Action lock is enabled."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=True)

        # Commands
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kToggleOn"
        description = (
            "Sets the output pin to continuously deliver the necessary signal to permanently engage the break at "
            "maximum strength. The output signal depends on whether the break is normally engaged."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kToggleOff"
        description = (
            "Sets the output pin to continuously deliver the necessary signal to permanently disengage the break. The "
            "output signal depends on whether the break is normally engaged."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kSetPower"
        description = (
            "Sets the output pin to perpetually output a square wave pulse modulated by the PWM value defined by "
            "the module instance custom parameters structure. The PWM value controls the amount of time the break "
            "is engaged, which, in turn, adjusts the overall breaking power."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  BreakModule does not send Data messages, so no additional sections to add.

        return code_map

    def set_parameters(self, pwm_strength: np.uint8 = np.uint8(255)) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Notes:
            When the manage BreakModule is running the set_power() command, updating the pwm_strength parameter will
            tune the strength at which the managed break is engaged.

        Args:
            pwm_strength: The Pulse-Width-Modulation (PWM) value to use when the module is triggered to deliver variable
                breaking power. Depending on this value, the breaking power can be adjusted from none (0) to maximum
                (255). This is only used when the break is engaged via set_power() command.

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
        """Toggles the managed break to be constantly engaged at maximum strength or disengaged.

        Notes:
            This command does not the pwm_strength parameter and always uses either maximum or minimum breaking power.
            To set the break to a specific power level, use the set_power() command.

        Args:
            state: The desired state of the break. Set to True to permanently enable the break with maximum power, or to
                False to permanently disable the break.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1 if state else 2),
            noblock=np.bool(False),
        )

    def set_power(self) -> OneOffModuleCommand:
        """Sets the managed break to engage with the desired strength, depending on the pwm_strength module parameter.

        Unlike the toggle() command, this command allows precisely controlling the power applied by the break by using
        a PWM signal to adjust the power. Depending on your specific use case, this command may either be very useful or
        not useful at all. For binary engage / disengage control the toggle() command is a more efficient choice.

        Notes:
            This command switches the break to run in the variable strength mode, but it does not determine the breaking
            power. To control the power, adjust the pwm_strength parameter by sending a ModuleParameters message with
            the new pwm_strength value. By default, the break power is set to engage the break with maximum power.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(3),
            noblock=np.bool(False),
        )


class SensorModule(ModuleInterface):
    """The class that exposes methods for interfacing with SensorModule instances running on Ataraxis MicroControllers.

    SensorModule facilitates receiving data recorded by any sensor that outputs analog unidirectional logic signals.
    This class functions similar to the TTLModule, but is specialized for working with analog signals. If you need to
    monitor a binary digital signal, use TTLModule instead for higher efficiency.

    Notes:
        This interface comes pre-configured to send triggers to Unity when it receives sensor data. If you do not need
        this functionality, override the unity_output flag at class instantiation!

    Args:
        module_id: The unique identifier code of the managed SensorModule instance.
        instance_name: The human-readable name of the SensorModule instance.
        instance_description: A longer human-readable description of the SensorModule instance.
        unity_output: Determines whether the SensorModule instance should send triggers to Unity when it receives sensor
            data.
        sensor_topic: The MQTT topic to which the instance should send the triggers based on the received sensor data.

    Attributes:
        _sensor_topic: Stores the output MQTT topic.
    """

    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        instance_description: str,
        unity_output: bool = True,
        sensor_topic: str = "LickPort/",
    ) -> None:
        # Statically defines the TTLModule type description.
        type_description = (
            f"The family of modules that receives unidirectional analog logic signals from a wide range of sensors."
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

        self._sensor_topic: str = sensor_topic

    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        # If the incoming message is not reporting a change in signal (code 51), aborts processing
        if message.event != np.uint8(51):
            return

        # Sends an empty message to the sensor MQTT topic, which acts as a binary trigger.
        unity_communication.send_data(topic=self._sensor_topic, payload=None)

    def send_to_queue(
        self,
        message: ModuleData | ModuleState,
        queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
        return

    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        # Status Codes
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kChanged"
        description = (
            "The monitored analog pin has detected a significant change in signal level relative to previous readout."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=False)

        # Commands
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kCheckState"
        description = (
            "Checks the state of the monitored analog pin. If the signal received by the pin is within the threshold "
            "boundaries and is significantly different from the signal detected during a previous readout, sends "
            "detected signal value to the PC. The signal value is always sent to the PC when it is checked the "
            "first time after class initialization."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  Data Objects
        module_section = f"{self.type_name}_module.data_objects"

        section = f"{module_section}.kChangedObject"
        description_1 = "The raw analog value of the signal received by the monitored input pin."
        code_map.write_nested_value(variable_path=f"{section}.event_code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.prototype_code", value=prototypes.kOneUnsignedShort)
        code_map.write_nested_value(variable_path=f"{section}.names", value=("signal",))
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

        Mostly, these parameters are used to filter incoming signals to minimize the number of messages sent to the PC.

        Notes:
            All threshold parameters are inclusive!

        Args:
            lower_threshold: The minimum signal level, in raw analog units, to be considered valid. Setting this
                threshold to a number above zero allows high-pass filtering the incoming signals.
            upper_threshold: The maximum signal level, in raw analog units, to be considered valid. Setting this
                threshold to a number below 65535 allows low-pass filtering the incoming signals. Note, due to the
                typically used analog readout resolution of 10-14 bytes, the practical ceiling of detected signals will
                likely not exceed 1000-5000 analog units.
            delta_threshold: The minimum value by which the signal has to change, relative to the previous check, for
                the change to be reported to the PC. Note, if the change is 0, the signal will not be reported to the
                PC, regardless of this parameter value.
            averaging_pool_size: The number of analog pin readouts to average together when checking pin state. This
                is used to smooth the recorded signal and eliminate analog communication noise. It is highly advised to
                have this enabled and set to at least 10 readouts.

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
        """Checks the state of the analog pin managed by the module.

        This command will evaluate the signal received by the analog pin and, if it is significantly different from the
        signal recorded during a previous check, report it to the PC. This approach ensures that only significant
        changes are communicated to the PC, preserving communication bandwidth. It is highly advised to issue this
        command to repeat (recur) at a desired interval to continuously monitor the pin state, rather than repeatedly
        calling it as a one-off command for best runtime efficiency.

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
                noblock=np.bool(False),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),
            cycle_delay=repetition_delay,
        )


class ValveModule(ModuleInterface):
    """The class that exposes methods for interfacing with ValveModule instances running on Ataraxis MicroControllers.

    ValveModule allows interfacing with a wide range of solenoid fluid or gas valves. To enable this, the class is
    designed to be connected to a Field-Effect-Transistor (FET) gated relay that controls the delivery of voltage to
    the valve. The module can be used to either permanently open or close the valve or to cycle opening and closing in
    a way that ensures a specific amount of gas or fluid passes through the valve.

    Notes:
        This interface comes pre-configured to receive valve pulse triggers from Unity. If you do not need this
        functionality, set the input_unity_topics argument to None when initializing the class!

    Args:
        module_id: The unique identifier code of the managed ValveModule instance.
        instance_name: The human-readable name of the ValveModule instance.
        instance_description: A longer human-readable description of the ValveModule instance.
        input_unity_topics: A tuple of Unity topics that the module should monitor to receive activation triggers.
    """

    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        instance_description: str,
        input_unity_topics: tuple[str, ...] | None = ("Gimbl/Reward/",),
    ) -> None:
        # Statically defines the TTLModule type description.
        type_description = (
            f"The family of modules that sends digital signals to open or close the managed solenoid fluid or gas "
            f"valve."
        )

        # Initializes the subclassed ModuleInterface using the input instance data. Type data is hardcoded.
        super().__init__(
            type_name="ValveModule",
            module_type=np.uint8(1),
            type_description=type_description,
            module_id=module_id,
            instance_name=instance_name,
            instance_description=instance_description,
            unity_input_topics=input_unity_topics,
            unity_output=False,
            queue_output=False,
        )

    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        """Not used."""
        return

    def send_to_queue(
        self,
        message: ModuleData | ModuleState,
        queue: MPQueue,  # type: ignore
    ) -> None:
        """Not used."""
        return

    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> OneOffModuleCommand:
        # Currently, the only processed topic is "Gimbl/Reward/". If more supported topics are needed, this needs to be
        # rewritten to use if-else conditions.

        # If the received message was sent to the reward topic, this is a binary (empty payload) trigger to
        # pulse the valve. It is expected that the valve parameters are configured so that this delivers the
        # desired amount of water reward.
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(False),  # Blocks to ensure reward delivery precision.
        )

    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary:
        # Status Codes
        module_section = f"{self.type_name}_module.status_codes"

        section = f"{module_section}.kOutputLocked"
        description = "Unable to sends the requested digital signal to the valve, as the global Action lock is enabled."
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(51))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.error", value=True)

        # Commands
        module_section = f"{self.type_name}_module.commands"

        section = f"{module_section}.kSendPulse"
        description = (
            "Pulses the managed valve by first opening it and, after a configurable delay, closing it. The delay "
            "is specified by the pulse_duration class execution parameter and is given in microseconds. When this "
            "command is received from the PC, a single open-close cycle will be carried out per each command. The "
            "specific digital signals used to open and close the valve depend on whether the valve is normally closed."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(1))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kToggleOn"
        description = (
            "Sets the output pin to continuously deliver the necessary signal to permanently open the managed valve. "
            "The output signal depends on whether the valve is normally closed."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(2))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kToggleOff"
        description = (
            "Sets the output pin to continuously deliver the necessary signal to permanently close the managed valve. "
            "The output signal depends on whether the valve is normally closed."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(3))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        section = f"{module_section}.kCalibrate"
        description = (
            "Calibrates the valve by issuing consecutive open-close cycles without delays. This is used to map "
            "different pulse durations to how much fluid or gas is release from the valve during the open phase of the "
            "pulse cycle. This command uses additional instance parameters to determine how many times to pulse the "
            "valve and how many microseconds to wait between consecutive pulses."
        )
        code_map.write_nested_value(variable_path=f"{section}.code", value=np.uint8(4))
        code_map.write_nested_value(variable_path=f"{section}.description", value=description)
        code_map.write_nested_value(variable_path=f"{section}.addressable", value=True)

        #  ValveModule does not send Data messages, so no additional sections to add.

        return code_map

    def set_parameters(
        self,
        pulse_duration: np.uint32 = np.uint32(10000),
        calibration_delay: np.uint32 = np.uint32(10000),
        calibration_count: np.uint16 = np.uint16(1000),
    ) -> ModuleParameters:
        """Sets PC-addressable runtime parameters of the module instance running on the microcontroller.

        Args:
            pulse_duration: The time, in microseconds, the valve stays open during pulsing. This is used during the
                execution of send_pulse() command to control the amount of dispensed gas or fluid.
            calibration_delay: The time, in microseconds, to wait between consecutive pulses during calibration.
                Calibration works by repeatedly pulsing the valve the requested number of times. Delaying after closing
                the valve ensures the valve hardware has enough time to respond to the inactivation phase, before the
                activation phase of the next pulse starts.
            calibration_count: The number of times to pulse the valve during calibration.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
        return ModuleParameters(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),  # Generally, return code is only helpful for debugging.
            parameter_data=(pulse_duration, calibration_delay, calibration_count),
        )

    def send_pulse(
        self, repetition_delay: np.uint32 = np.uint32(0), noblock: bool = False
    ) -> RepeatedModuleCommand | OneOffModuleCommand:
        """Delivers the predetermined amount of gas or fluid once or repeatedly (recurrently) by opening and closing
        (pulsing) teh valve.

        After calibration, this command allows delivering precise amounts of fluid accurate in the microliter and,
        depending on the used valve and relay hardware, even nanoliter ranges. Generally, this is the most common way
        of using the solenoid valves. If you need to pulse the valve over even intervals, issue a repeating (recurrent)
        command to maximize the repetition precision.

        Notes:
            To ensure the accuracy of fluid or gas delivery, it is recommended to run the valve in the blocking mode
            and, if possible, isolate it to a controller that is not busy with running other modules.

        Args:
            repetition_delay: The time, in microseconds, to delay before repeating the command. If this is set to 0,
                the command will only run once. Note, the exact repetition delay will be further affected by other
                modules managed by the same microcontroller and may not be perfectly accurate.
            noblock: Determines whether the command should block the microcontroller while the valve is kept open or
                not. Blocking ensures precise pulse duration, non-blocking allows the microcontroller to perform other
                operations while waiting, increasing its throughput.

        Returns:
            The RepeatedModuleCommand or OneOffModuleCommand message to be sent to the microcontroller.
        """

        if repetition_delay == 0:
            return OneOffModuleCommand(
                module_type=self._module_type,
                module_id=self._module_id,
                return_code=np.uint8(0),
                command=np.uint8(1),
                noblock=np.bool(noblock),
            )

        return RepeatedModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(1),
            noblock=np.bool(noblock),
            cycle_delay=repetition_delay,
        )

    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Toggles the managed valve to be constantly opened or closed.

        Args:
            state: The desired state of the valve. Set to True to permanently open the valve, or to False to permanently
                close the valve.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(2 if state else 3),
            noblock=np.bool(False),
        )

    def calibrate(self) -> OneOffModuleCommand:
        """Calibrates the valve by repeatedly pulsing it a certain number of times.

        This command is used to build the calibration map of the valve that matches pulse_durations to the amount of
        fluid or gas dispensed during the opened phase of the pulse. To do so, the command repeatedly issues a high
        number of pulses to dispense a large volume of fluid or gas. The number of pulses carried out during this
        command is specified by the calibration_count parameter, and the delay between pulses is specified by the
        calibration_delay parameter.

        Notes:
            When activated, this command will block in-place until the calibration cycle is completed. Currently, there
            is no way to interrupt the command, and it may take a prolonged period of time (minutes) to complete.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
        return OneOffModuleCommand(
            module_type=self._module_type,
            module_id=self._module_id,
            return_code=np.uint8(0),
            command=np.uint8(4),
            noblock=np.bool(False),
        )
