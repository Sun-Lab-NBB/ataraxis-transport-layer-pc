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
- Uses JIT-compilation and NumPy to optimize data processing and communication speeds, where possible.
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

For users, all library dependencies are installed automatically by all supported installation methods.
(see [Installation](#installation) section). 

For developers, see the [Developers](#developers) section for 
information on installing additional development dependencies.
___

## Installation

### Source

Note, installation from source is ***highly discouraged*** for everyone who is not an active project developer.
Developers should see the [Developers](#Developers) section for more details on installing from source. The instructions
below assume you are ***not*** a developer.

1. Download this repository to your local machine using your preferred method, such as Git-cloning. Use one
   of the stable releases from [GitHub](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer-pc/releases).
2. Unpack the downloaded zip and copy the path to the appropriate binary wheel (`.whl`) file.
3. Run ```python -m pip install WHEEL_PATH```, replacing 'WHEEL_PATH' with the path to the wheel file to install the 
   wheel into the active conda environment.

### PIP
Use the following command to install the library using PIP: ```pip install ataraxis-transport-layer-pc```
___

## Usage

### TransportLayer
The TransportLayer class provides an intermediate-level API for bidirectional communication over USB or UART serial 
interfaces. It ensures proper encoding and decoding of data packets using the Consistent Overhead Byte Stuffing (COBS) 
protocol and ensures transmitted packet integrity via Cyclic Redundancy Check (CRC).

#### Packet Anatomy:
This class sends and receives data in the form of packets. Each packet adheres to the following general 
layout:

`[START] [PAYLOAD SIZE] [COBS OVERHEAD] [PAYLOAD (1 to 254 bytes)] [DELIMITER] [CRC CHECKSUM (1 to 4 bytes)]`

To optimize runtime efficiency, the class generates two buffers at compile time that store encoded and decoded payloads.
TransportLayer’s write_data() and read_data() methods work with payload data buffers. The rest of the packet data is 
processed exclusively during send_data() and receive_data() runtime and is not accessible to users. Therefore, users 
can safely ignore all packet-related information and focus on working with transmitted and received serialized payloads.

#### Quickstart
This is a minimal example of how to use this library.

```
# Imports the TransportLayer class.
from ataraxis_transport_layer_pc import TransportLayer

# Imports dataclass to demonstrate struct-like data transmission
from dataclasses import dataclass

# Imports numpy to use for payload generation.
import numpy as np

# Imports sleep function to delay execution after connection cycling
from time import sleep

# Instantiates a new TransportLayer object. Most class parameters are set to value that should scale with any
# microcontroller. However, you do need to provide the USB port name (can be discovered via 'axtl-ports' CLI command)
# and the microcontroller's Serial buffer size (can be obtained from the microcontroller's manufacturer). Check the API
# documentation website if you want to fine-tune other class parameters to better match your use case.
tl_class = TransportLayer(port="/dev/ttyACM2", microcontroller_serial_buffer_size=300)

# Some Arduino boards reset after receiving a connection request. To make this example universal, sleeps for 5 seconds
# to ensure the microcontroller is ready to receive data.
sleep(5)

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

# Waits for the microcontroller to receive the data and respond by sending its data.
while not tl_class.available:
    continue  # If no data is available, the loop blocks until it becomes available.

# If the data is available, carries out the reception procedure (reads the received byte-stream, parses the
# payload, and makes it available for reading).
data_received = tl_class.receive_data()

# If the reception was successful, reads the data, assumed to contain serialized test objects. Note, this
# example is intended to be used together with the example script from the ataraxis-transport-layer-mc library.
if data_received:
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
print("Test completed successfully!")
print(f"test_scalar = {test_scalar}")
print(f"test_array = {test_array}")
print(f"test_struct = {test_struct}")
```
#### Key Methods

##### Sending Data
There are two key methods associated with sending data to the PC:
- The `write_data()` method serializes the input object into bytes and writes the resultant byte sequence into 
  the `_transmission_buffer` payload region starting at the specified `start_index`.
- The `send_data()` method encodes the payload into a packet using COBS, calculates the CRC checksum for the encoded 
  packet, and transmits the packet and the CRC checksum to PC. The method requires that at least one byte of data is 
  written to the staging buffer via the WriteData() method before it can be sent to the PC.

The example below showcases the sequence of steps necessary to send the data to the PC and assumes TransportLayer 
'tl_class' was initialized following the steps in the [Quickstart](#quickstart) example:
```
// Generates the test array to simulate the payload.
uint8_t test_array[10] = {1, 2, 3, 0, 0, 6, 0, 8, 0, 0};

// Writes the data into the _transmission_buffer.
tl_class.WriteData(test_array, 0);

// Sends the payload to the Stream buffer. If all steps of this process succeed, the method returns 'true' and the data
// is handed off to the 
bool sent_status = tl_class.SendData();
```

#### Receiving Data
There are three key methods associated with receiving data from the PC:
- The `available` property checks if the serial interface has received enough bytes to justify parsing the data. If this
  method returns False, calling ReceiveData() will likely fail.
- The `receive_data()` method reads the encoded packet from the byte-stream stored in Serial interface buffer, verifies 
  its integrity with CRC, and decodes the payload from the packet using COBS. If the packet was successfully received 
  and unpacked, this method returns True.
- The `read_data()` method overwrites the memory (data) of the input object with the data extracted from the received 
  payload. To do so, the method reads the number of bytes necessary to 'fill' the object with data from the payload, 
  starting at the `start_index`. Following this procedure, the object will have new value(s) that match the read 
  data.

The example below showcases the sequence of steps necessary to receive data from the PC and assumes TransportLayer
'tl_class' was initialized following the steps in the [Quickstart](#quickstart) example: 
```
// Packages and sends the contents of the transmission buffer that were written above to the PC.
tl_class.SendData();  //

if (tl_class.Available())
{
    tl_class.ReceiveData();
}
uint16_t value    = 44321;
uint8_t array[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

struct MyStruct
{
        uint8_t a  = 60;
        uint16_t b = 12345;
        uint32_t c = 1234567890;
} test_structure;

// Overwrites the test objects with the data stored inside the buffer
uint16_t next_index = tl_class.ReadData(value);  // ReadData defaults to start_index 0 if it is not provided
uint16_t next_index = tl_class.ReadData(array, next_index);
uint16_t next_index = tl_class.ReadData(test_structure, next_index);
```
___
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

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` to the root directory of the project using your command line interface of choice.
3. Install development dependencies. You have multiple options of satisfying this requirement:
    1. **_Preferred Method:_** Use conda or pip to install [tox](https://tox.wiki/en/latest/user_guide.html) and call
       ```tox -e import``` to automatically import the os-specific development environment included with the source 
       code in your local conda distribution. Alternatively, you can use ```tox -e create``` to create the environment 
       from scratch and automatically install the necessary dependencies using pyproject.toml file. See 
       [environments](#environments) section for other environment installation methods.
    2. Run ```python -m pip install .'[dev]'``` command to install development dependencies and the library using 
       pip. On some systems, you may need to use a slightly modified version of this command: 
       ```python -m pip install .[dev]```.
    3. As long as you have an environment with [tox](https://tox.wiki/en/latest/user_guide.html) installed
       and do not intend to run any code outside the predefined project automation pipelines, tox will automatically
       install all required dependencies for each task. In this case, skip installing additional development 
       dependencies.

### Additional Dependencies

In addition to installing the required python packages, separately install the following dependencies:

1. [Python](https://www.python.org/downloads/) distributions, one for each version that you intend to support. 
   The easiest way to get tox to work as intended is to have separate python distributions, but using 
   [pyenv](https://github.com/pyenv/pyenv) is a good alternative. This is needed for the 'test' task to work as 
   intended.

### Development Automation

This project comes with a fully configured set of automation pipelines implemented using 
[tox](https://tox.wiki/en/latest/user_guide.html). Check [tox.ini file](tox.ini) for details about 
available pipelines and their implementation. Alternatively, call ```tox list``` from the root directory of the project
to see the list of available tasks.

**Note!** All commits to this project have to successfully complete the ```tox``` task before being pushed to GitHub. 
To minimize the runtime for this task, use ```tox --parallel```.

For more information, you can also see the 'Usage' section of the 
[ataraxis-automation project](https://github.com/Sun-Lab-NBB/ataraxis-automation) documentation.

### Environments

All environments used during development are exported as .yml files and as spec.txt files to the [envs](envs) folder.
The environment snapshots were taken on each of the three explicitly supported OS families: Windows 11, OSx (M1) 14.5
and Linux Ubuntu 22.04 LTS.

**Note!** Since the OSx environment was built for an M1 (Apple Silicon) platform, it may not work on Intel-based 
Apple devices.

To install the development environment for your OS:

1. Download this repository to your local machine using your preferred method, such as git-cloning.
2. ```cd``` into the [envs](envs) folder.
3. Use one of the installation methods below:
    1. **_Preferred Method_**: Install [tox](https://tox.wiki/en/latest/user_guide.html) or use another
       environment with already installed tox and call ```tox -e import```.
    2. **_Alternative Method_**: Run ```conda env create -f ENVNAME.yml``` or ```mamba env create -f ENVNAME.yml```. 
       Replace 'ENVNAME.yml' with the name of the environment you want to install (axbu_dev_osx for OSx, 
       axbu_dev_win for Windows, and axbu_dev_lin for Linux).

**Hint:** while only the platforms mentioned above were explicitly evaluated, this project is likely to work on any 
common OS, but may require additional configurations steps.

Since the release of [ataraxis-automation](https://github.com/Sun-Lab-NBB/ataraxis-automation) version 2.0.0 you can 
also create the development environment from scratch via pyproject.toml dependencies. To do this, use 
```tox -e create``` from project root directory.

### Automation Troubleshooting

Many packages used in 'tox' automation pipelines (uv, mypy, ruff) and 'tox' itself are prone to various failures. In 
most cases, this is related to their caching behavior. Despite a considerable effort to disable caching behavior known 
to be problematic, in some cases it cannot or should not be eliminated. If you run into an unintelligible error with 
any of the automation components, deleting the corresponding .cache (.tox, .ruff_cache, .mypy_cache, etc.) manually 
or via a cli command is very likely to fix the issue.
___

## Versioning

We use [semantic versioning](https://semver.org/) for this project. For the versions available, see the 
[tags on this repository](https://github.com/Sun-Lab-NBB/ataraxis-transport-layer/tags).

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
- [numpy](https://github.com/numpy/numpy) project for providing low-level functionality for many of the 
  classes exposed through this library.
- [numba]() project for providing 
- The creators of all other projects used in our development automation pipelines [see pyproject.toml](pyproject.toml).

---
