import sys

import numpy as np
from ataraxis_time.precision_timer.timer_class import PrecisionTimer
from ataraxis_transport_layer import MicroControllerInterface, SerialCommunication, ModuleParameters, KernelParameters, RepeatedModuleCommand
from ataraxis_time import PrecisionTimer

from src.ataraxis_transport_layer.communication import KernelCommand

#print(SerialCommunication.list_available_ports())

set_output_params = KernelParameters(
    action_lock=False,
    ttl_lock=False,
    return_code=np.uint8(11),
)

ttl_params = ModuleParameters(
    module_type=np.uint8(2),
    module_id=np.uint8(1),
    return_code=np.uint8(22),
    parameter_data=(np.uint32(5000000), np.uint8(0))
)

ttl_pulse = RepeatedModuleCommand(
    module_type=np.uint8(2),
    module_id=np.uint8(1),
    return_code=np.uint8(121),
    command=np.uint8(1),
    noblock=np.uint8(1),
    cycle_delay=np.uint32(5000000),
)

timeout = PrecisionTimer(precision='ms')
if __name__ == '__main__':
    interface = MicroControllerInterface(
        name="Test",
        controller_id=np.uint8(123),
        usb_port="/dev/cu.usbmodem142013801",
        baudrate=115200,
        maximum_transmitted_payload_size=254,

    )

    interface.identify_controller()
    interface._transmission_queue.put(set_output_params)
    interface._transmission_queue.put(ttl_params)
    interface._transmission_queue.put(ttl_pulse)

    timeout.delay_noblock(delay=20000, allow_sleep=True)

    print('Terminating')

    interface._terminator_array.write_data(0, 1)
    timeout.delay_noblock(delay=2000, allow_sleep=True)
    interface._terminator_array.disconnect()
    interface._terminator_array.destroy()

