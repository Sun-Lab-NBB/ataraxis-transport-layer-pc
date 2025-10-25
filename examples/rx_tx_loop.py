# This example is intended to be sued together with the quickstart loop for the companion library:
# https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc#quickstart.
# See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc for more details.
# API documentation: https://ataraxis-transport-layer-pc-api-docs.netlify.app/.
# Authors: Ivan Kondratyev (Inkaros), Katlynn Ryu.

from dataclasses import field, dataclass
import numpy as np
from ataraxis_time import PrecisionTimer
from ataraxis_transport_layer_pc import TransportLayer
from ataraxis_base_utilities import console, LogLevel

# Activates the console to print messages to the terminal during runtime.
if not console.enabled:
    console.enable()

# Instantiates a new TransportLayer object. Most class initialization arguments are set to use optimal default values
# for most microcontrollers and assume that the companion library uses the default parameters. Consult the ReadMe and
# the API documentation to learn about fine-tuning the TransportLayer's parameters to better match the intended
# use-case.
tl_class = TransportLayer(port="/dev/ttyACM1", baudrate=115200, microcontroller_serial_buffer_size=256)

# Note, buffer size 256 is set for an Arduino Due board. Most Arduino boards have buffers capped at 64 or 256
# bytes. During production runtimes, it is critically important to set the buffer size to the actual size used by the
# interfaced microcontroller.

# Similarly, the baudrate used here is not optimal for all UART microcontrollers. For the communication to be stable,
# the baudrate must be set to an optimal value for the specific microcontroller participating in the communication
# cycle. Use the https://wormfood.net/avrbaudcalc.php tool to find the best baudrate for your AVR board or consult the
# manufacturer's documentation.

# Pre-creates the objects used for the demonstration below.
test_scalar = np.uint32(123456789)
test_array = np.zeros(4, dtype=np.uint8)  # [0, 0, 0, 0]


# While Python does not have C++-like structures, it has dataclasses that fulfill a similar role.
@dataclass()  # It is important for the class to NOT be frozen!
class TestStruct:
    test_flag: np.bool = field(default_factory=lambda: np.bool(True))
    test_float: np.float32 = field(default_factory=lambda: np.float32(6.66))

    def __repr__(self) -> str:
        return f"TestStruct(test_flag={self.test_flag}, test_float={round(float(self.test_float), ndigits=2)})"


test_struct = TestStruct()

# Some Arduino boards reset after receiving a connection request. To make this example universal, sleeps for 2 seconds
# to ensure the microcontroller is ready to receive data.
timer = PrecisionTimer("s")
timer.delay(delay=2, allow_sleep=True, block=False)

console.echo("Transmitting the data to the microcontroller...")

# Executes one transmission and one data reception cycle. During production runtime, this code would typically run in
# a function or loop.

# Writes objects to the TransportLayer's transmission buffer, staging them to be sent with the next
# send_data() command. Note, the objects are written in the order they are read by the microcontroller.
tl_class.write_data(test_scalar)
tl_class.write_data(test_array)
tl_class.write_data(test_struct)

# Packages and sends the contents of the transmission buffer that were written above to the Microcontroller.
tl_class.send_data()

console.echo("Data transmission: Complete.", level=LogLevel.SUCCESS)

# Waits for the microcontroller to receive the data and respond by sending its data back to the PC.
console.echo("Waiting for the microcontroller to respond...")
while not tl_class.available:
    continue  # If no data is available, the loop blocks until it becomes available.

# If the data is available, carries out the reception procedure (reads the received byte-stream, parses the
# payload, and makes it available for reading).
data_received = tl_class.receive_data()

# If the reception was successful, reads the data, assumed to contain serialized test objects. Note, this
# example is intended to be used together with the example script from the ataraxis-transport-layer-mc library.
if data_received:
    console.echo("Data reception: Complete.", level=LogLevel.SUCCESS)

    # Overwrites the memory of the objects that were sent to the microcontroller with the response data
    test_scalar = tl_class.read_data(test_scalar)
    test_array = tl_class.read_data(test_array)
    test_struct = tl_class.read_data(test_struct)

    # Verifies the received data
    assert test_scalar == np.uint32(987654321)  # The microcontroller overwrites the scalar with reverse order.

    # The rest of the data is transmitted without any modifications.
    assert np.array_equal(test_array, np.array([0, 0, 0, 0]))
    assert test_struct.test_flag == np.bool(True)
    assert test_struct.test_float == np.float32(6.66)

# Prints the received data values to the terminal for visual inspection.
console.echo("Data reading: Complete.", level=LogLevel.SUCCESS)
console.echo("Received data values:")
console.echo(f"test_scalar = {test_scalar}")
console.echo(f"test_array = {test_array}")
console.echo(f"test_struct = {test_struct}")
