# This example is intended to be used together with the rx_tx_loop example of the Python companion library
# ataraxis-transport-layer-pc. When used correctly, the example code will continuously transmit and receive data
# between the microcontroller and the PC.

# This example is intended to be sued together with the quickstart loop for the companion library:
# https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc#quickstart.
# See https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc for more details.
# API documentation: https://ataraxis-transport-layer-pc-api-docs.netlify.app/.
# Authors: Ivan Kondratyev (Inkaros), Katlynn Ryu.

# Imports the TransportLayer class
# Imports sleep to delay execution after establishing the connection
from time import sleep

# Imports dataclass to demonstrate struct-like data transmission
from dataclasses import dataclass

# Imports numpy, which is used to generate data payloads
import numpy as np

from ataraxis_transport_layer_pc import TransportLayer

# Instantiates a new TransportLayer object. Most class initialization arguments should scale with any microcontroller.
# However, you do need to provide the USB port name (can be discovered via 'axtl-ports' CLI command)
# and the microcontroller's Serial buffer size (can be obtained from the microcontroller's manufacturer). Check the API
# documentation website if you want to fine-tune other class parameters to better match your use case.
tl_class = TransportLayer(port="/dev/ttyACM2", baudrate=115200, microcontroller_serial_buffer_size=8192)

# Note, the buffer size 8192 assumes you are using Teensy 3.0+. Most Arduino boards have buffers capped at 64 or 256
# bytes. While this demonstration will likely work even if the buffer size is not valid, it is critically
# important to set this value correctly for production runtimes.

# Similarly, the baudrate here will likely need to be adjusted for UART microcontrollers. If baudrate is not set
# correctly, the communication will not be stable (many packets will be corrupted in transmission). You can use this
# https://wormfood.net/avrbaudcalc.php tool to find the best baudrate for your AVR board or consult the manufacturer's
# documentation.

# Pre-creates the objects used for the demonstration below.
test_scalar = np.uint32(123456789)
test_array = np.zeros(4, dtype=np.uint8)  # [0, 0, 0, 0]


# While Python does not have C++-like structures, dataclasses can be used for a similar purpose.
@dataclass()  # It is important for the class to NOT be frozen!
class TestStruct:
    test_flag: np.bool = np.bool(True)
    test_float: np.float32 = np.float32(6.66)

    def __repr__(self) -> str:
        return f"TestStruct(test_flag={self.test_flag}, test_float={round(float(self.test_float), ndigits=2)})"


test_struct = TestStruct()


# Some Arduino boards reset after receiving a connection request. To make this example universal, sleeps for 2 seconds
# to ensure the microcontroller is ready to receive data.
sleep(2)

print("Transmitting the data to the microcontroller...")

# Executes one transmission and one data reception cycle. During production runtime, this code would typically run in
# a function or loop.

# Writes objects to the TransportLayer's transmission buffer, staging them to be sent with the next
# send_data() command. Note, the objects are written in the order they will be read by the microcontroller.
next_index = 0  # Starts writing from the beginning of the transmission buffer.
next_index = tl_class.write_data(test_scalar, next_index)
next_index = tl_class.write_data(test_array, next_index)
# Since test_struct is the last object in the payload, we do not need to save the new next_index.
next_index = tl_class.write_data(test_struct, next_index)

# Packages and sends the contents of the transmission buffer that were written above to the Microcontroller.
tl_class.send_data()  # This also returns a boolean status that we discard for this example.

print("Data transmission complete.")

# Waits for the microcontroller to receive the data and respond by sending its data.
while not tl_class.available:
    continue  # If no data is available, the loop blocks until it becomes available.

# If the data is available, carries out the reception procedure (reads the received byte-stream, parses the
# payload, and makes it available for reading).
data_received = tl_class.receive_data()

# If the reception was successful, reads the data, assumed to contain serialized test objects. Note, this
# example is intended to be used together with the example script from the ataraxis-transport-layer-mc library.
if data_received:
    print("Data reception complete.")

    # Overwrites the memory of the objects that were sent to the microcontroller with the response data
    next_index = 0  # Resets the index to 0.
    test_scalar, next_index = tl_class.read_data(test_scalar, next_index)
    test_array, next_index = tl_class.read_data(test_array, next_index)
    test_struct, _ = tl_class.read_data(test_struct, next_index)  # Again, the index after last object is not saved.

    # Verifies the received data
    assert test_scalar == np.uint32(987654321)  # The microcontroller overwrites the scalar with reverse order.

    # The rest of the data is transmitted without any modifications.
    assert np.array_equal(test_array, np.array([0, 0, 0, 0]))
    assert test_struct.test_flag == np.bool(True)
    assert test_struct.test_float == np.float32(6.66)

# Prints the received data values to the terminal for visual inspection.
print("Data reading complete.")
print(f"test_scalar = {test_scalar}")
print(f"test_array = {test_array}")
print(f"test_struct = {test_struct}")
