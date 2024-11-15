import sys
from pathlib import Path

import numpy as np
from ataraxis_data_structures import DataLogger
from ataraxis_time.precision_timer.timer_class import PrecisionTimer

from ataraxis_transport_layer import EncoderModule, MicroControllerInterface

# print(SerialCommunication.list_available_ports())

timeout = PrecisionTimer(precision="s")
if __name__ == "__main__":
    # Initializes the logger
    output_directory = Path("/path/to/output/directory")
    logger = DataLogger(output_directory)

    module = EncoderModule(
        module_id=np.uint8(1), instance_name="TestEncoder", instance_description="You test encoders!"
    )

    interface = MicroControllerInterface(
        controller_id=np.uint8(123),
        controller_name="TestController",
        controller_description="The controller used to test our code and hardware assembly.",
        logger_queue=logger.input_queue,
        modules=(module,),
        controller_usb_port="/dev/cu.usbmodem142013801",
        baudrate=115200,
        maximum_transmitted_payload_size=254,
        unity_broker_ip="127.0.0.1",
        unity_broker_port=1883,
        verbose=True,
    )

    interface.start()

    # This is used to test dictionary generation :)
    # print(interface.general_map_section)
    # print()
    # print(interface.microcontroller_map_section)
    #
    # sys.exit(0)

    interface.identify_controller()
    interface.unlock_controller()

    check_command = module.check_state(repetition_delay=np.uint32(5000000))

    interface.send_message(check_command)

    timeout.delay_noblock(delay=60, allow_sleep=True)

    interface.stop()
