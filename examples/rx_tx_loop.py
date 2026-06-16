"""Demonstrates bidirectional serial communication with a microcontroller using TransportLayer.

Intended to be used together with the quickstart loop for the companion library:
https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc#quickstart.
See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc for more details.
API documentation: https://ataraxis-transport-layer-pc-api-docs.netlify.app/.
Authors: Ivan Kondratyev (Inkaros), Katlynn Ryu.
"""

from dataclasses import field, dataclass

import numpy as np
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from ataraxis_transport_layer_pc import TransportLayer

# Activates the console to print messages to the terminal during runtime.
if not console.enabled:
    console.enable()

# Instantiates a new TransportLayer object. Most class initialization arguments are set to use optimal default values
# for most microcontrollers and assume that the companion library uses the default parameters. Consult the ReadMe and
# the API documentation to learn about fine-tuning the TransportLayer's parameters to better match the intended
# use-case.
transport_layer = TransportLayer(port="/dev/ttyACM1", baudrate=115200, microcontroller_serial_buffer_size=256)

# Note, buffer size 256 is set for an Arduino Due board. Most Arduino boards have buffers capped at 64 or 256
# bytes. During production runtimes, it is critically important to set the buffer size to the actual size used by the
# interfaced microcontroller.

# Similarly, the baudrate used here is not optimal for all UART microcontrollers. For the communication to be stable,
# the baudrate must be set to an optimal value for the specific microcontroller participating in the communication
# cycle. Use the https://wormfood.net/avrbaudcalc.php tool to find the best baudrate for your AVR board or consult the
# manufacturer's documentation.

# Pre-creates the objects used for the demonstration below.
test_scalar = np.uint32(123456789)
test_array = np.zeros(4, dtype=np.uint8)


# While Python does not have C++-like structures, it has dataclasses that fulfill a similar role. This dataclass
# must not be frozen, because read_data() overwrites its fields in place during the reception procedure.
@dataclass(slots=True)
class TestStruct:
    """Groups the test values used to demonstrate dataclass serialization."""

    test_flag: np.bool_ = field(default_factory=lambda: np.bool_(True))
    """Demonstrates serialization of a numpy boolean value."""
    test_float: np.float32 = field(default_factory=lambda: np.float32(6.66))
    """Demonstrates serialization of a numpy 32-bit floating-point value."""

    def __repr__(self) -> str:
        """Returns a string representation of the TestStruct instance."""
        return f"TestStruct(test_flag={self.test_flag}, test_float={round(float(self.test_float), ndigits=2)})"


test_struct = TestStruct()

# Some Arduino boards reset after receiving a connection request. To make this example universal, sleeps for 2 seconds
# to ensure the microcontroller is ready to receive data.
timer = PrecisionTimer(precision="s")
timer.delay(delay=2, allow_sleep=True, block=False)

console.echo(message="Transmitting the data to the microcontroller...")

# Executes one transmission and one data reception cycle. During production runtime, this code would typically run in
# a function or loop.

# Writes objects to the TransportLayer's transmission buffer, staging them to be sent with the next
# send_data() command. Note, the objects are written in the order they are read by the microcontroller.
transport_layer.write_data(test_scalar)
transport_layer.write_data(test_array)
transport_layer.write_data(test_struct)

# Packages and sends the contents of the transmission buffer that were written above to the Microcontroller.
transport_layer.send_data()

console.echo(message="Data transmission: Complete.", level=LogLevel.SUCCESS)

# Waits for the microcontroller to receive the data and respond by sending its data back to the PC.
console.echo(message="Waiting for the microcontroller to respond...")

# If no data is available, the loop blocks until the microcontroller's response becomes available.
while not transport_layer.available:
    continue

# If the data is available, carries out the reception procedure (reads the received byte-stream, parses the
# payload, and makes it available for reading).
data_received = transport_layer.receive_data()

# If the reception was successful, reads the data, assumed to contain serialized test objects. Note, this
# example is intended to be used together with the example script from the ataraxis-transport-layer-mc library.
if data_received:
    console.echo(message="Data reception: Complete.", level=LogLevel.SUCCESS)

    # Overwrites the memory of the objects that were sent to the microcontroller with the response data.
    test_scalar = transport_layer.read_data(test_scalar)
    test_array = transport_layer.read_data(test_array)
    test_struct = transport_layer.read_data(test_struct)

    # The microcontroller replaces the scalar with a new fixed value before sending it back.
    assert test_scalar == np.uint32(987654321)

    # The rest of the data is transmitted without any modifications.
    assert np.array_equal(test_array, np.array([0, 0, 0, 0], dtype=np.uint8))
    assert test_struct.test_flag == np.bool_(True)
    assert test_struct.test_float == np.float32(6.66)

# Prints the received data values to the terminal for visual inspection.
console.echo(message="Data reading: Complete.", level=LogLevel.SUCCESS)
console.echo(message="Received data values:")
console.echo(message=f"test_scalar = {test_scalar}")
console.echo(message=f"test_array = {test_array}")
console.echo(message=f"test_struct = {test_struct}")
