from pathlib import Path

import numpy as np
from ataraxis_data_structures import DataLogger
from ataraxis_time.precision_timer.timer_class import PrecisionTimer

from ataraxis_transport_layer import EncoderModule, MicroControllerInterface

timeout = PrecisionTimer(precision="s")
if __name__ == "__main__":
    # Initializes and starts the logger
    output_directory = Path("/home/cybermouse/Desktop/TestLog")
    logger = DataLogger(output_directory)
    logger.start()

    # Initializes the tested module interface
    module = EncoderModule(
        module_id=np.uint8(1), instance_name="TestEncoder", instance_description="You test encoders!"
    )

    # Initializes and starts the microcontroller interface. Provides it with the tested module instance initialized
    # above.
    interface = MicroControllerInterface(
        controller_id=np.uint8(123),
        controller_name="TestController",
        controller_description="The controller used to test our code and hardware assembly.",
        logger_queue=logger.input_queue,
        modules=(module,),
        controller_usb_port="/dev/ttyACM0",
        baudrate=115200,
        maximum_transmitted_payload_size=254,
        unity_broker_ip="127.0.0.1",
        unity_broker_port=1883,
        verbose=True,
    )
    interface.start()
    interface.unlock_controller()

    # Sends out the tested command
    check_command = module.check_state(repetition_delay=np.uint32(1000000))
    interface.send_message(check_command)

    # Statically blocks for 20 seconds while running recurrent commands.
    timeout.delay_noblock(delay=20, allow_sleep=True)

    # Shuts down the communication interface and the logger
    interface.stop()
    logger.shutdown()

    # Compresses and lists the logged data. This is done mostly to verify the integrity of the logs by printing them to
    # terminal.
    logger.compress_logs(remove_sources=True, verbose=True)
