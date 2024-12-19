# ataraxis-transport-layer-pc

A Python library that provides methods for establishing and maintaining bidirectional communication with Arduino and 
Teensy microcontrollers over USB or UART serial interfaces.

![PyPI - Version](https://img.shields.io/pypi/v/ataraxis-transport-layer-pc)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ataraxis-transport-layer-pc)
[![uv](https://tinyurl.com/uvbadge)](https://github.com/astral-sh/uv)
[![Ruff](https://tinyurl.com/ruffbadge)](https://github.com/astral-sh/ruff)
![type-checked: mypy](https://img.shields.io/badge/type--checked-mypy-blue?style=flat-square&logo=python)
![PyPI - License](https://img.shields.io/pypi/l/ataraxis-transport-layer-pc)
![PyPI - Status](https://img.shields.io/pypi/status/ataraxis-transport-layer-pc)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/ataraxis-transport-layer-pc)
___

## Detailed Description

This is the Python implementation of the ataraxis-transport-layer (AXTL) library, designed to run on 
host-computers (PCs). It provides methods for bidirectionally communicating with a microcontroller running the 
[ataraxis-transport-layer-mc](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc) companion library written in 
C++. The library abstracts most steps necessary for data transmission, such as serializing data into payloads, 
packing the payloads into packets, and transmitting packets as byte-streams to the receiver. It also abstracts the 
reverse sequence of steps necessary to verify and decode the payload from the packet received as a stream of bytes. The 
library is specifically designed to support time-critical applications, such as scientific experiments, and can achieve 
microsecond communication speeds for newer microcontroller-PC configurations.
___

## Features

- Supports Windows, Linux, and macOS.
- Uses Consistent Overhead Byte Stuffing (COBS) to encode payloads.
- Supports Circular Redundancy Check (CRC) 8-, 16- and 32-bit polynomials to ensure data integrity during transmission.
- Uses JIT-compilation and NumPy to optimize data processing and communication speeds.
- Wraps JIT-compiled methods into pure-python interfaces to improve user experience.
- Has a [companion](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc) libray written in C++ to simplify 
  PC-MicroController communication.
- GPL 3 License.
___

## Table of Contents

- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Developers](#developers)
- [Versioning](#versioning)
- [Authors](#authors)
- [License](#license)
- [Acknowledgements](#Acknowledgments)
___

## Dependencies

For users, all library dependencies are installed automatically by all supported installation methods 
(see [Installation](#installation) section). 

For developers, see the [Developers](#developers) section for information on installing additional development 
dependencies.
___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.
Developers should see the [Developers](#Developers) section for more details on installing from source. The instructions
below assume you are ***not*** a developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file, to install the 
   wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install ataraxis-transport-layer-pc```.
___

## Usage

### TransportLayer
The TransportLayer class provides an intermediate-level API for bidirectional communication over USB or UART serial 
interfaces. It ensures proper encoding and decoding of data packets using the Consistent Overhead Byte Stuffing (COBS) 
protocol and ensures transmitted packet integrity via Cyclic Redundancy Check (CRC).

#### Packet Anatomy:
This class sends and receives data in the form of packets. Each packet adheres to the following general layout:

`[START] [PAYLOAD SIZE] [COBS OVERHEAD] [PAYLOAD (1 to 254 bytes)] [DELIMITER] [CRC CHECKSUM (1 to 4 bytes)]`

To optimize runtime efficiency, the class generates two buffers at initialization time that store encoded and 
decoded payloads. TransportLayer’s write_data() and read_data() methods work with payload data buffers. The rest of 
the packet data is processed exclusively by send_data() and receive_data() methods and is not accessible to users.
Therefore, users can safely ignore all packet-related information and focus on working with transmitted and received
serialized payloads.

#### JIT Compilation:

The class uses numba under-the-hood to compile many data processing steps to efficient C-code the first time these
methods are called. Since compilation is expensive, the first call to each numba-compiled method will be very slow, but
all further calls will be much faster. For optimal performance, call all TransportLayer methods at least once before 
entering the time-critical portion of your runtime so that it has time to precompile the code.

#### Initialization Delay
Some microcontrollers, such as Arduino AVR boards, reset upon establishing UART connection. If TransportLayer attempts
to transmit the data to a microcontroller undergoing the reset, the data may not reach the microcontroller at all or 
become corrupted. If you are using a microcontroller with UART interface, delay further code execution by ~2–5 seconds 
after initializing the TransportLayer class to allow the microcontroller to finish its reset sequence.

#### Baudrates
For microcontrollers using the UART serial interface, it is essential to set the baudrate to a value supported 
by the microcontroller’s hardware. Usually, manufactures provide a list of supported baudrates for each 
microcontroller. Additionally, the baudrate values used in the microcontroller code and the PC code have to match. 
If any of these conditions are not satisfied, the communication will not be stable and many transmitted packets 
will be corrupted.

#### Quickstart
This is a minimal example of how to use this library. It is designed to be used together with the quickstart example
of the [companion](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc#quickstart) library. 
See the [rx_tx_loop.py](./examples/rx_tx_loop.py) for .py implementation:
```
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
```

#### Key Methods

##### Sending Data
There are two key methods associated with sending data to the microcontroller:
- The `write_data()` method serializes the input object into bytes and writes the resultant byte sequence into 
  the `_transmission_buffer` starting at the specified `start_index`.
- The `send_data()` method encodes the payload into a packet using COBS, calculates the CRC checksum for the encoded 
  packet, and transmits the packet and the CRC checksum to microcontroller. The method requires that at least one byte 
  of data is written to the staging buffer via the WriteData() method before it can be sent to the microcontroller.

The example below showcases the sequence of steps necessary to send the data to the microcontroller and assumes
TransportLayer 'tl_class' was initialized following the steps in the [Quickstart](#quickstart) example:
```
# Generates the test array to simulate the payload.
test_array = np.array(object=[1, 2, 3, 0, 0, 6, 0, 8, 0, 0], dtype=np.uint8)

# Writes the data into the _transmission_buffer. The method returns the index (next_index) that can be used to add
# another object directly behind the current object. This supports chained data writing operations, where the
# returned index of the previous write_data call is used as the start_index of the next write_data call.
next_index = tl_class.write_data(test_array, start_index=0)

# Sends the payload to the pySerial transmission buffer. If all steps of this process succeed, the method returns
# 'true' and the data is handed off to the serial interface to be transmitted.
sent_status = tl_class.send_data()  # Returns True if the data was sent
```

#### Receiving Data
There are three key methods associated with receiving data from the microcontroller:
- The `available` property checks if the serial interface has received enough bytes to justify parsing the data. If this
  property is False, calling receive_data() will likely fail.
- The `receive_data()` method reads the encoded packet from the byte-stream stored in pySerial interface buffer, 
  verifies its integrity with CRC, and decodes the payload from the packet using COBS. If the packet was successfully
  received and unpacked, this method returns True.
- The `read_data()` method recreates the input object with the data extracted from the received payload. To do so, 
  the method reads the number of bytes necessary to 'fill' the object with data from the payload, starting at the
  `start_index` and uses the object type to recreate the instance with new data. Following this procedure, the new 
  object whose memory matches the read data will be returned to caller. Note, this is different from the C++ library,
  where the object instance is modified by reference, instead of being recreated.

The example below showcases the sequence of steps necessary to receive data from the microcontroller and assumes
TransportLayer 'tl_class' was initialized following the steps in the [Quickstart](#quickstart) example: 
```
# Generates the test array to which the received data will be written.
test_array[10] = np.array([1, 2, 3, 0, 0, 6, 0, 8, 0, 0], dtype=np.uint8)

# Blocks until the data is received from the microcontroller.
while not tl_class.available:
    continue

# Parses the received data. Note, this method internally checks 'available' property', so it is safe to call 
# receive_data() instead of available in the 'while' loop above without changing how this example behaves.
receive_status = tl_class.receive_data()  # Returns True if the data was received and passed verification.

# Recreates and returns the new test_array instance using the data received from the microcontroller. Also returns the 
# index that can be used to read the next object in the received data payload. This supports chained data reading 
# operations, where the returned index of the previous read_data call can be used as the start_index for the next 
# read_data call.
updated_array, next_index = tl_class.read_data(test_array, 0)  # Start index is 0.
```
___

## API Documentation

See the [API documentation](https://ataraxis-transport-layer-pc-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.
___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library.

### Installing the library

The easiest way to ensure you have most recent development dependencies and library source files is to install the 
python environment for your OS (see below). All environments used during development are exported as .yml files and as 
spec.txt files to the [envs](envs) folder. The environment snapshots were taken on each of the three explicitly 
supported OS families: Windows 11, OSx Darwin, and GNU Linux.

**Note!** Since the OSx environment was built for the Darwin platform (Apple Silicon), it may not work on Intel-based 
Apple devices.

1. If you do not already have it installed, install [tox](https://tox.wiki/en/latest/user_guide.html) into the active
   python environment. The rest of this installation guide relies on the interaction of local tox installation with the
   configuration files included in with this library.
2. Download this repository to your local machine using your preferred method, such as git-cloning. If necessary, unpack
   and move the project directory to the appropriate location on your system.
3. ```cd``` to the root directory of the project using your command line interface of choice. Make sure it contains
   the `tox.ini` and `pyproject.toml` files.
4. Run ```tox -e import``` to automatically import the os-specific development environment included with the source 
   distribution. Alternatively, you can use ```tox -e create``` to create the environment from scratch and automatically
   install the necessary dependencies using pyproject.toml file. 
5. If either step 4 command fails, use ```tox -e provision``` to fix a partially installed environment.

**Hint:** while only the platforms mentioned above were explicitly evaluated, this project will likely work on any 
common OS, but may require additional configurations steps.

### Additional Dependencies

In addition to installing the development environment, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. These 
   versions will be installed in-addition to the main Python version installed in the development environment.
   The easiest way to get tox to work as intended is to have separate python distributions, but using 
   [pyenv](https://github.com/pyenv/pyenv) is a good alternative. This is needed for the 'test' task to work as 
   intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check [tox.ini file](tox.ini) for details about 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All commits to this project have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime duration for this task, use ```tox --parallel```.

For more information, check the 'Usage' section of the 
[ataraxis-automation project](https://github.com/Sun-Lab-NBB/ataraxis-automation#Usage) documentation.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself are prone to various failures. In 
most cases, this is related to their caching behavior. Despite a considerable effort to disable caching behavior known 
to be problematic, in some cases it cannot or should not be eliminated. If you run into an unintelligible error with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a cli command is very likely to fix the issue.
___

## Versioning

We use [semantic versioning](https://semver.org/) for this project. For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc/tags).

---

## Authors

- Ivan Kondratyev ([Inkaros](https://github.com/Inkaros))
- Katlynn Ryu ([katlynn-ryu](https://github.com/KatlynnRyu))

___

## License

This project is licensed under the GPL3 License: see the [LICENSE](LICENSE) file for details.
___

## Acknowledgments

- All Sun lab [members](https://neuroai.github.io/sunlab/people) for providing the inspiration and comments during the
  development of this library.
- [PowerBroker2](https://github.com/PowerBroker2) and his 
  [pySerialTransfer](https://github.com/PowerBroker2/pySerialTransfer) for inspiring this library and serving as an 
  example and benchmark. Check pySerialTransfer as a good alternative with non-overlapping functionality that may be 
  better for your project.
- The creators of all other projects used in our development automation pipelines and source code 
  [see pyproject.toml](pyproject.toml).
---
