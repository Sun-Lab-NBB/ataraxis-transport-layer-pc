import numpy as np
from ataraxis_data_structures import DataLogger
from ataraxis_time.precision_timer.timer_class import PrecisionTimer

from ataraxis_transport_layer import KernelParameters, ModuleParameters, RepeatedModuleCommand, MicroControllerInterface

# print(SerialCommunication.list_available_ports())

set_output_params = KernelParameters(
    action_lock=False,
    ttl_lock=False,
    return_code=np.uint8(11),
)

ttl_params = ModuleParameters(
    module_type=np.uint8(2),
    module_id=np.uint8(1),
    return_code=np.uint8(22),
    parameter_data=(np.uint32(5000000), np.uint8(0)),
)

ttl_pulse = RepeatedModuleCommand(
    module_type=np.uint8(2),
    module_id=np.uint8(1),
    return_code=np.uint8(121),
    command=np.uint8(1),
    noblock=np.uint8(1),
    cycle_delay=np.uint32(5000000),
)

timeout = PrecisionTimer(precision="ms")
if __name__ == "__main__":
    logger = DataLogger

    interface = MicroControllerInterface(
        controller_id=np.uint8(123),
        controller_name="TestController",
        controller_description="The controller used to test our code and hardware assembly.",
        controller_usb_port="/dev/cu.usbmodem142013801",
        baudrate=115200,
        maximum_transmitted_payload_size=254,
    )

    interface.start()

    interface.identify_controller()
    interface.send_message(set_output_params)
    interface.send_message(ttl_params)
    interface.send_message(ttl_pulse)

    timeout.delay_noblock(delay=20000, allow_sleep=True)

    interface.stop()
