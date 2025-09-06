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
host-computers (PCs). It provides methods for bidirectionally communicating with microcontrollers running the 
[ataraxis-transport-layer-mc](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc) companion library written in 
C++. The library abstracts the steps necessary for data transmission, such as serializing data into payloads, 
packing the payloads into packets, and transmitting packets as byte-streams. It also abstracts the reverse sequence of 
steps necessary to verify and decode the payload from the packet received as a stream of bytes. The library is 
specifically designed to support time-critical applications, such as scientific experiments, and achieves microsecond
communication speeds for high-end microcontroller-PC configurations.

___

## Features

- Supports Windows, Linux, and macOS.
- Uses Consistent Overhead Byte Stuffing (COBS) to encode payloads.
- Supports Cyclic Redundancy Check (CRC) 8-, 16- and 32-bit polynomials to ensure data integrity during transmission.
- Uses Just-in-Time (JIT) compilation and NumPy to optimize data processing and communication speeds.
- Wraps JIT-compiled methods into pure-python interfaces to improve the user experience.
- Has a [companion](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc) libray written in C++ to simplify 
  PC-MicroController communication.
- GPL 3 License.
- 
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

All library dependencies are installed automatically by all supported installation methods 
(see [Installation](#installation) section).

___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.
Developers should see the [Developers](#Developers) section for more details on installing from source.

1. Download this repository to the local machine using the preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc/releases).
2. Unpack the downloaded zip and note the path to the binary wheel (`.whl`) file contained in the archive.
3. Run ```python -m pip install WHEEL_PATH```, replacing the 'WHEEL_PATH' with the path to the wheel file, to install 
   the wheel into the active python environment.

### pip
Use the following command to install the library using pip: ```pip install ataraxis-transport-layer-pc```.

___

## Usage

### TransportLayer
The TransportLayer class provides a high-level API for bidirectional communication over USB or UART serial interfaces. 
It ensures proper encoding and decoding of data packets using the Consistent Overhead Byte Stuffing (COBS) protocol and 
ensures transmitted packet integrity via Cyclic Redundancy Check (CRC).

#### Packet Anatomy:
The TransportLayer sends and receives data in the form of packets. Each packet adheres to the following general layout:

`[START] [PAYLOAD SIZE] [COBS OVERHEAD] [PAYLOAD (1 to 254 bytes)] [DELIMITER] [CRC CHECKSUM (1 to 4 bytes)]`

To optimize runtime efficiency, the class generates two buffers at initialization time that store encoded and 
decoded payloads. TransportLayer’s write_data() and read_data() methods work with payload data buffers. The rest of 
the packet data is processed exclusively by send_data() and receive_data() methods and is not accessible to users.

***Note!*** Because of this design, end-users can ignore all packet-related information and focus on working with 
transmitted and received payloads.

#### JIT Compilation:
The class uses numba under-the-hood to compile many data processing steps to efficient C-code the first time these
methods are called. Since compilation is expensive, the first call to each numba-compiled method is typically very slow,
but all further calls are considerably faster. For optimal performance, call all TransportLayer methods at least once 
before entering the time-critical portion of the runtime so that it has time to precompile the code.

#### Initialization Delay
Some microcontrollers, such as Arduino AVR boards, reset upon establishing UART connection. If TransportLayer attempts
to transmit the data to a microcontroller undergoing the reset, the data may not reach the microcontroller at all or 
become corrupted. When using a microcontroller with UART interface, delay further code execution by ~2–5 seconds 
after initializing the TransportLayer class to allow the microcontroller to finish its reset sequence.

#### Baudrates
For microcontrollers using the UART serial interface, it is essential to set the baudrate to a value supported 
by the microcontroller’s hardware. Usually, manufactures provide a list of supported baudrates for each 
microcontroller. Additionally, the baudrate values used in the microcontroller code and the PC code have to match. 
If any of these conditions are not satisfied, the connection can become unstable, leading to the corruption of 
exchanged data packets.

#### Quickstart
See the [rx_tx_loop.py](./examples/rx_tx_loop.py) for a minimal example of how to use this library. It is designed to 
be used together with the quickstart example of the 
[companion](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-mc#quickstart) library.

#### Key Methods

##### Sending Data
There are two key methods associated with sending data to the microcontroller:
- The `write_data()` method serializes the input object into bytes and writes the resultant byte sequence into 
  the `_transmission_buffer` starting at the specified `start_index`.
- The `send_data()` method encodes the payload into a packet using COBS, calculates the CRC checksum for the encoded 
  packet, and transmits the packet and the CRC checksum to microcontroller. The method requires that at least one byte 
  of data is written to the staging buffer via the write_data() method before it can be sent to the microcontroller.

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
  property is False, calling receive_data() is unlikely to receive any data.
- The `receive_data()` method reads the encoded packet from the byte-stream stored in pySerial interface buffer, 
  verifies its integrity with CRC, and decodes the payload from the packet using COBS. If the packet is successfully
  received and unpacked, this method returns True. In the current library version, the method also checks the 
  'available' property, making it generally unnecessary to have a separate check before calling the method.
- The `read_data()` method recreates the input object with the data extracted from the received payload. To do so, 
  the method reads the number of bytes necessary to 'fill' the object with data from the payload, starting at the
  `start_index` and uses the object type to recreate the instance with new data. Following this procedure, the new 
  object whose memory matches the read data is returned to the caller. Note, this is different from the C++ 
  library, where referencing modifies the object instance, instead of being recreated.

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

### Discovering Connectable Ports
To help determining which USB ports are available for communication, this library exposes the `axtl-ports` CLI command. 
This command is available from any environment that has the library installed and internally calls the 
`print_available_ports()` standalone function. The command prints all USB ports that can be connected
by the pySerial backend alongside the available ID information. The returned port address can then be provided to the 
TransportLayer class as the 'port' argument to establish the serial communication through the port.

___

## API Documentation

See the [API documentation](https://ataraxis-transport-layer-pc-api-docs.netlify.app/) for the
detailed description of the methods and classes exposed by components of this library.

___

## Developers

This section provides installation, dependency, and build-system instructions for the developers that want to
modify the source code of this library. Additionally, it contains instructions for recreating the conda environments
that were used during development from the included .yml files.

### Installing the library

1. Download this repository to the local machine using the preferred method, such as git-cloning.
2. ```cd``` to the root directory of the project.
3. Install development dependencies. There are multiple ways of satisfying this requirement:
    1. **_Preferred Method:_** Use mamba, uv, or pip to install
       [tox](https://tox.wiki/en/latest/user_guide.html) or use an environment that has it installed and
       call ```tox -e import``` to automatically import the os-specific development environment included with the
       project source code. Alternatively, use ```tox -e create``` to create the environment from scratch and 
       automatically install the necessary dependencies using the pyproject.toml file.
    2. Run ```python -m pip install .'[dev]'``` command to install development dependencies and the library using 
       pip. Some platforms may require a slightly modified version of this command: 
       ```python -m pip install .[dev]```.

### Additional Dependencies

In addition to installing the development environment, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version supported by the project. 
   Currently, this library supports the three latest stable versions. The easiest way to get tox to work as intended is 
   to have separate python distributions. Alternatively, use [pyenv](https://github.com/pyenv/pyenv) to install multiple
   Python versions. This is needed for the 'test' task to work as intended.

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

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself may experience runtime failures. In 
most cases, this is related to their caching behavior. Despite a considerable effort to disable caching behavior known 
to be problematic, in some cases it cannot or should not be eliminated. If an unintelligible error is encountered with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually  
or via a CLI command is very likely to fix the issue.

___

## Versioning

This project uses [semantic versioning](https://semver.org/). For the versions available, see the 
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
