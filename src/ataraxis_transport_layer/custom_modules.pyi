from multiprocessing import Queue as MPQueue

import numpy as np
from _typeshed import Incomplete
from ataraxis_data_structures import NestedDictionary

from .communication import (
    ModuleData as ModuleData,
    ModuleState as ModuleState,
    ModuleParameters as ModuleParameters,
    UnityCommunication as UnityCommunication,
    OneOffModuleCommand as OneOffModuleCommand,
    RepeatedModuleCommand as RepeatedModuleCommand,
<<<<<<< HEAD
    prototypes as prototypes,
=======
>>>>>>> 5b8062e (Added Communication module tests)
)
from .microcontroller import ModuleInterface as ModuleInterface

class TTLModule(ModuleInterface):
    """The class that exposes methods for interfacing with TTLModule instances running on Ataraxis MicroControllers.

    TTLModule facilitates exchanging Transistor-to-Transistor Logic (TTL) signals between various hardware systems, such
    as microcontrollers, cameras, and other hardware. The module contains methods for both sending and receiving the TTL
    pulses, but each TTLModule instance can only do one of these functions at a time.

    Args:
        module_id: The unique identifier code of the managed TTLModule instance.
        instance_name: The human-readable name of the TTLModule instance.
        instance_description: A longer human-readable description of the TTLModule instance.
    """
    def __init__(self, module_id: np.uint8, instance_name: str, instance_description: str) -> None: ...
    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        """Not used."""
    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary: ...
    def set_parameters(self, pulse_duration: np.uint32 = ..., averaging_pool_size: np.uint8 = ...) -> ModuleParameters:
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
    def send_pulse(
        self, repetition_delay: np.uint32 = ..., noblock: bool = True
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
    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Sets the digital pin managed by the module instance to continuously output the desired signal level.

        Use this to permanently activate or inactivate the pin. Since this is a lock-in command, it does not make sense
        to make it recurrent.

        Args:
            state: The signal to output. Set to True for HIGH and False for LOW.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
    def check_state(self, repetition_delay: np.uint32 = ...) -> OneOffModuleCommand | RepeatedModuleCommand:
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
        encoder_ppr: The resolution of the managed quadrature encoder, in Pulses Per Revolution (PPR). Specifically,
            this is the number of quadrature pulses the encoder emits per full 360-degree rotation. If this number is
            not known, provide a placeholder value and use get_ppr() command to estimate the PPR using the index channel
            of the encoder.
        object_diameter: The diameter of the rotating object connected to the encoder, in cm. This is used to
            convert encoder pulses into rotated distance in cm.
        cm_per_unity_unit: The conversion factor to translate the distance traveled by the edge of the circular object
             into the Unity units. This value works together with object_diameter and encoder_ppr to translate raw
             encoder pulses received from the microcontroller into Unity-compatible units, used by the game engine to
             move the VirtualReality agent.

    Attributes:
        _motion_topic: Stores the MQTT motion topic.
        _ppr: Stores the resolution of the managed quadrature encoder.
        _object_diameter: Stores the diameter of the object connected to the encoder.
        _cm_per_unity_unit: Stores the conversion factor that translates centimeters into Unity units.
        _unity_unit_per_pulse: Stores the conversion factor to translate encoder pulses into Unity units.
    """

    _motion_topic: Incomplete
    _ppr: Incomplete
    _object_diameter: Incomplete
    _cm_per_unity_unit: Incomplete
    _unity_unit_per_pulse: Incomplete
    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        instance_description: str,
        unity_output: bool = True,
        motion_topic: str = "LinearTreadmill/Data",
        encoder_ppr: int = 8192,
        object_diameter: float = 15.0333,
        cm_per_unity_unit: float = 10.0,
    ) -> None: ...
    def send_to_unity(self, message: ModuleState | ModuleData, unity_communication: UnityCommunication) -> None: ...
    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary: ...
    def write_instance_variables(self, code_map: NestedDictionary) -> NestedDictionary: ...
    def set_parameters(
        self, report_ccw: np.bool | bool = ..., report_cw: np.bool | bool = ..., delta_threshold: np.uint32 | int = ...
    ) -> ModuleParameters:
        """Sets PC-addressable parameters of the module instance running on the microcontroller.

        Args:
            report_ccw: Determines whether to report rotation in the CCW (positive) direction.
            report_cw: Determines whether to report rotation in the CW (negative) direction.
            delta_threshold: The minimum number of pulses required for the motion to be reported. Depending on encoder
                resolution, this allows setting the 'minimum rotation distance' threshold for reporting. Note, if the
                change is 0 (the encoder readout did not change), it will not be reported, regardless of the
                value of this parameter. Sub-threshold motion will be aggregated (summed) across readouts until a
                significant overall change in position is reached to justify reporting it to the PC.

        Returns:
            The ModuleParameters message to be sent to the microcontroller.
        """
    def check_state(self, repetition_delay: np.uint32 = ...) -> OneOffModuleCommand | RepeatedModuleCommand:
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
    def reset_pulse_count(self) -> OneOffModuleCommand:
        """Resets the current pulse tracker of the encoder to 0, clearing any currently stored rotation data.

        Primarily, this command is helpful if you need to reset the encoder without evaluating its current pulse count.
        For example, this would be the case if there is a delay between the initialization of the module and the start
        of the encoder monitoring. Resetting the encoder before evaluating its pulse count for the first time discards
        a nonsensical pulse count aggregated before the monitoring has started. Similarly, this can be used to re-base
        the encoder pulse count without reading the aggregate data.
        """
    def get_ppr(self) -> OneOffModuleCommand:
        """Uses the index channel of the encoder to estimate its Pulse-per-Revolution (PPR).

        The PPR allows converting raw pulse counts reported by other commands of this module to real life circular
        distances. This is a service command not intended to be used during most production runtimes if the PPR is
        already known. It relies on the encoder completing up to 11 full rotations and uses the index channel of the
        encoder to detect each revolution completion.

        Notes:
            Make sure the calibrated encoder rotates at a steady slow speed until this command completes. Similar to
            other service commands, it is designed to deadlock the controller until the command completes. Since this
            interface exclusive works with the encoder, you have to provide the encoder rotation separately (manually).

            The direction of the rotation is not relevant for this command, as long as it make the full 360-degree
            revolution.

            The command is optimized for the object to be rotated with a human hand at a steady rate, so it delays
            further index pin polling for 100 milliseconds each time the index pin is triggered. Therefore, the object
            should not make more than 10 rotations per second and ideally should stay within 1-3 rotations per second.
            It is also possible to evaluate motor-assisted rotation if the motor does not leak the
            magnetic field that would interfere with the index signaling from the encoder and spins the object at the
            speed discussed above.
        """

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
    def __init__(self, module_id: np.uint8, instance_name: str, instance_description: str) -> None: ...
    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        """Not used."""
    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary: ...
    def set_parameters(self, pwm_strength: np.uint8 = ...) -> ModuleParameters:
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

    _sensor_topic: Incomplete
    def __init__(
        self,
        module_id: np.uint8,
        instance_name: str,
        instance_description: str,
        unity_output: bool = True,
        sensor_topic: str = "LickPort/",
    ) -> None: ...
    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None: ...
    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> None:
        """Not used."""
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary: ...
    def set_parameters(
        self,
        lower_threshold: np.uint16 = ...,
        upper_threshold: np.uint16 = ...,
        delta_threshold: np.uint16 = ...,
        averaging_pool_size: np.uint8 = ...,
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
    def check_state(self, repetition_delay: np.uint32 = ...) -> OneOffModuleCommand | RepeatedModuleCommand:
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
    ) -> None: ...
    def send_to_unity(self, message: ModuleData | ModuleState, unity_communication: UnityCommunication) -> None:
        """Not used."""
    def send_to_queue(self, message: ModuleData | ModuleState, queue: MPQueue) -> None:
        """Not used."""
    def get_from_unity(self, topic: str, payload: bytes | bytearray) -> OneOffModuleCommand: ...
    def write_code_map(self, code_map: NestedDictionary) -> NestedDictionary: ...
    def set_parameters(
        self, pulse_duration: np.uint32 = ..., calibration_delay: np.uint32 = ..., calibration_count: np.uint16 = ...
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
    def send_pulse(
        self, repetition_delay: np.uint32 = ..., noblock: bool = False
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
    def toggle(self, state: bool) -> OneOffModuleCommand:
        """Toggles the managed valve to be constantly opened or closed.

        Args:
            state: The desired state of the valve. Set to True to permanently open the valve, or to False to permanently
                close the valve.

        Returns:
            The OneOffModuleCommand message to be sent to the microcontroller.
        """
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
