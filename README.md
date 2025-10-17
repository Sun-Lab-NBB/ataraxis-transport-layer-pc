# ataraxis-transport-layer-pc

A Python library that provides methods for establishing and maintaining bidirectional communication with Arduino and 
Teensy microcontrollers over USB and UART serial interfaces.

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
host-computers (PCs). It provides methods for bidirectionally communicating with microcontrollers running the 
[ataraxis-transport-layer-mc](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc) companion library written in 
C++. The library abstracts all steps necessary to safely send and receive data over the USB and UART communication
interfaces. It is specifically designed to support time-critical applications, such as scientific experiments, and can 
achieve microsecond communication speeds for modern microcontroller-PC hardware combinations.

___

## Features

- Supports Windows, Linux, and macOS.
- Uses Consistent Overhead Byte Stuffing (COBS) to encode payloads during transmission.
- Supports Circular Redundancy Check (CRC) 8-, 16- and 32-bit polynomials to ensure data integrity during transmission.
- Allows fine-tuning all library components to support a wide range of application contexts.
- Uses Just-in-Time (JIT) compilation and NumPy to optimized runtime performance in time-critical applications.
- Has a [companion](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc) microcontroller libray written in C++.
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
(see the [Installation](#installation) section).

***Note!*** Developers should see the [Developers](#developers) section for information on installing additional 
development dependencies.

___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for anyone who is not an active project developer.

1. Download this repository to the local machine using the preferred method, such as git-cloning. Use one of the 
   [stable releases](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc/releases) that include precompiled 
   binary and source code distribution (sdist) wheels.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Run ```python -m pip install .``` to install the project. Alternatively, if using a distribution with precompiled
   binaries, use ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file.

### pip
Use the following command to install the library using pip: ```pip install ataraxis-transport-layer-pc```

___

## Usage

### TransportLayer
The TransportLayer class provides the API for bidirectional communication over USB or UART serial interfaces. It 
ensures proper encoding and decoding of data packets using the Consistent Overhead Byte Stuffing (COBS) 
scheme and ensures transmitted packet integrity through the use of the Cyclic Redundancy Check (CRC) checksums.

#### Packet Anatomy:
The TransportLayer class sends and receives data in the form of packets. Each packet adheres to the following general 
layout:

`[START] [PAYLOAD SIZE] [COBS OVERHEAD] [PAYLOAD (1 to 254 bytes)] [DELIMITER] [CRC CHECKSUM (1 to 4 bytes)]`

To optimize runtime efficiency, the class generates two buffers at initialization time that store the incoming and 
outgoing data packets. Additionally, the class generates a static lookup table to speed up the CRC checksum calculations
at runtime.

***Note!*** TransportLayer’s write_data() and read_data() methods ***exclusively*** work with the **PAYLOAD** region of 
each data buffer. End users can safely ignore all packet-related information and focus on working with transmitted and
received serialized payloads, as it is impossible to access and manipulate packet metadata via the public API.

#### JIT Compilation:
The class uses numba under-the-hood to compile many data processing steps to efficient C-code the first time these
methods are called. Since compilation is expensive, the first call to each numba-compiled method is typically very slow,
but all further calls are considerably faster. For optimal performance, call all TransportLayer methods at least once 
before entering the time-critical portion of the runtime so that it has time to precompile the code.

#### Initialization Delay
Some microcontrollers, such as Arduino AVR boards, reset upon establishing connection over the UART interface. If 
TransportLayer attempts to transmit the data to a microcontroller undergoing the reset, the data may not reach the 
microcontroller at all or become corrupted. When using a microcontroller with the UART interface, delay further code 
execution by ~2–5 seconds after initializing the TransportLayer class to allow the microcontroller to finish its reset 
sequence.

#### Baudrates
For microcontrollers using the UART interface, it is essential to set the baudrate to a value supported by the 
microcontroller’s hardware. Usually, manufactures provide a list of supported baudrates for each 
microcontroller. Additionally, the baudrate values used in the microcontroller and PC versions of the library have to 
match. If any of these conditions are not satisfied, the connection can become unstable, leading to the corruption of 
exchanged data packets.

#### Quickstart
This minimal example demonstrates how to use this library to send and receive data. It is designed to be used together 
with the quickstart example of the [companion](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc#quickstart) 
library. See the [rx_tx_loop.py](./examples/rx_tx_loop.py) for the .py implementation of this example:
```
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
```

#### Key Methods

##### Sending Data
There are two key methods associated with sending data to the microcontroller:
- The `write_data()` method serializes the input object and writes the resultant byte sequence to the 
  transmission buffer’s payload region. Each call appends the data to the end of the payload already stored in the 
  transmission buffer.
- The `send_data()` method encodes the payload stored in the transmission buffer into a packet using COBS, calculates 
  and adds the CRC checksum to the encoded packet, and transmits the packet to the microcontroller. The method requires 
  at least one byte of data to be written to the staging buffer before it can be sent to the microcontroller.

The example below showcases the sequence of steps necessary to send the data to the microcontroller and assumes
TransportLayer 'tl_class' was initialized following the steps in the [Quickstart](#quickstart) example:
```
# Generates the test array to simulate the payload.
test_array = np.array(object=[1, 2, 3, 0, 0, 6, 0, 8, 0, 0], dtype=np.uint8)

# Writes the data into the instance's transmission buffer. The method raises an error if it is unable to write the 
# data.
tl_class.write_data(test_array)

# Constructs and hands the packet to the communication interface to be transmitted to the microcontroller.
tl_class.send_data()
```

***Note!*** The transmission buffer is reset when the data is transmitted or via the call to the 
`reset_transmission_buffer()` method. Resetting the transmission buffer discards all data stored in the buffer.

#### Receiving Data
There are three key methods associated with receiving data from the microcontroller:
- The `available` property checks if the serial interface has received enough bytes to justify parsing the data.
- The `receive_data()` method reads the encoded packet from the byte-stream stored in Serial interface buffer, verifies 
  its integrity with the CRC checksum, and decodes the payload from the packet using COBS. If the packet was 
  successfully received and unpacked, this method returns True.
- The `read_data()` method overwrites the memory (data) of the input object with the data extracted from the received 
  payload. To do so, the method reads and consumes the number of bytes necessary to 'fill' the object with data from 
  the payload. Following this procedure, the object stores the new value(s) that match the read data and the consumed
  bytes are discarded, meaning it is only possible to read the same data **once**.

The example below showcases the sequence of steps necessary to receive data from the microcontroller and assumes
TransportLayer 'tl_class' was initialized following the steps in the [Quickstart](#quickstart) example: 
```
# Generates the test array to which the received data will be written.
test_array[10] = np.array([1, 2, 3, 0, 0, 6, 0, 8, 0, 0], dtype=np.uint8)

# Blocks until the data is received from the microcontroller.
while not tl_class.available:
    continue

# Parses the received data. Note, this method internally accesses the 'available' property, so it is safe to call 
# receive_data() instead of 'available' in the 'while' loop above without changing how this example behaves.
receive_status = tl_class.receive_data()  # Returns True if the packet was received and decoded.

# Recreates and returns the new test_array instance using the data received from the microcontroller. The method raises 
# an error if it is unable to read the data.
updated_array = tl_class.read_data(test_array)
```

***Note!*** Each call to the `receive_data()` method resets the instance’s reception buffer, discarding any potentially
unprocessed data.

### Discovering Connectable Ports
To help determining which USB ports are available for communication, this library exposes the `axtl-ports` CLI command. 
This command is available from any environment that has the library installed and internally calls the 
`print_available_ports()` standalone function. The command prints all USB ports that can be connected
by the pySerial interface alongside the available ID information. The returned port address can then be provided to the 
TransportLayer class as the 'port' argument to establish the serial communication through the port.

___

## API Documentation

See the [API documentation](https://ataraxis-transport-layer-pc-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for project developers.

### Installing the Project

***Note!*** This installation method requires **mamba version 2.3.2 or above**. Currently, all Sun lab automation 
pipelines require that mamba is installed through the [miniforge3](https://github.com/conda-forge/miniforge) installer.

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. If the downloaded distribution is stored as a compressed archive, unpack it using the appropriate decompression tool.
3. ```cd``` to the root directory of the prepared project distribution.
4. Install the core Sun lab development dependencies into the ***base*** mamba environment via the 
   ```mamba install tox uv tox-uv``` command.
5. Use the ```tox -e create``` command to create the project-specific development environment followed by 
   ```tox -e install``` command to install the project into that environment as a library.

### Additional Dependencies

In addition to installing the project and all user dependencies, install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the developed project. 
   Currently, this library supports the three latest stable versions. It is recommended to use a tool like 
   [pyenv](https://github.com/pyenv/pyenv) to install and manage the required versions.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check the [tox.ini file](tox.ini) for details about the 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All pull requests for this project have to successfully complete the ```tox``` task before being merged. 
To expedite the task’s runtime, use the ```tox --parallel``` command to run some tasks in-parallel.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself may experience runtime failures. In 
most cases, this is related to their caching behavior. If an unintelligible error is encountered with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a CLI command typically solves the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). See the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc/tags) for the available project 
releases.

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
  example and benchmark. Check pySerialTransfer project as a good alternative to this library with a non-overlapping 
  set of features.
- The creators of all other dependencies and projects listed in the [pyproject.toml](pyproject.toml) file.

---
