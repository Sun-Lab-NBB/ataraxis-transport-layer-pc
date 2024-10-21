from .transport_layer import SerialTransportLayer
from dataclasses import dataclass, field
import numpy as np
from typing import Any
from enum import Enum
from numpy.typing import NDArray
import struct


@dataclass
class ParameterMessage:
    module_type: np.uint8
    module_id: np.uint8
    return_code: np.uint8
    object: Any  # This will not be packed; assumed to be handled separately
    packed_data: np.ndarray = field(init=False)

    def __post_init__(self):
        # Use struct to pack fixed-size fields
        packed = struct.pack(
            "BBB",  # Format: 3 uint8 fields
            self.module_type,
            self.module_id,
            self.return_code,
        )
        # Converts packed data (which is a bytes' object) to a numpy array
        fixed_fields_array = np.frombuffer(packed, dtype=np.uint8)

        protocol = np.array([Protocols.PARAMETERS.value], dtype=np.uint8)

        # Initializes packed_data as the result of packing all fields
        self.packed_data = np.concatenate([protocol, fixed_fields_array])


@dataclass
class DataMessage:
    module_type: np.uint8
    module_id: np.uint8
    command: np.uint8
    event: np.uint8
    object_size: np.uint8
    object: Any = None  # Object is handled separately or provided later

    @classmethod
    def from_array(cls, data: NDArray[np.uint8]):
        # Returns an instance of DataMessage with the decoded fields
        return cls(
            module_type=np.uint8(data[0]).copy(),
            module_id=np.uint8(data[1]).copy(),
            command=np.uint8(data[2]).copy(),
            event=np.uint8(data[3]).copy(),
            object_size=np.uint8(data[4]).copy(),
            object=None,  # Object can be processed separately based on object_size
        )


@dataclass
class ServiceMessage:
    service_code: np.uint8


class Protocols(Enum):
    COMMAND = np.uint8(1)
    PARAMETERS = np.uint8(2)
    DATA = np.uint8(3)
    RECEPTION = np.uint8(4)
    IDENTIFICATION = np.uint8(5)


class SerialCommunication:
    """Add later"""

    def __init__(
        self,
        usb_port: str,
        baudrate: int = 115200,
        maximum_transmitted_payload_size: int = 254,
    ) -> None:

        # Specializes the TransportLayer to mostly match a similar specialization carried out by the microcontroller
        # Communication class.
        self.transport_layer = SerialTransportLayer(
            port=usb_port,
            baudrate=baudrate,
            polynomial=0x1021,
            initial_crc_value=0xFFFF,
            final_crc_xor_value=0x0000,
            maximum_transmitted_payload_size=maximum_transmitted_payload_size,
            minimum_received_payload_size=7,  # Data message header size (5) + protocol (1) + minimum object size (1)
            start_byte=129,
            delimiter_byte=0,
            timeout=20000,
            test_mode=False,
        )

    def reset_transmission_state(self) -> None:
        """Resets the transmission_buffer and associated tracker variables involved in the data transmission process."""
        self.transport_layer.reset_transmission_buffer()

    def reset_reception_state(self) -> None:
        """Resets the reception_buffer and associated tracker variables involved in the data reception process."""
        self.transport_layer.reset_reception_buffer()

    def send_command_message(
        self,
        module_type: np.uint8,
        module_id: np.uint8,
        return_code: np.uint8,
        command: np.uint8,
        noblock: np.bool,
        cycle: np.bool,
        cycle_delay: np.uint32,
    ) -> None:

        # Ensures the transmission_buffer is cleared
        self.reset_reception_state()

        # Packages the input data into a byte numpy array. Prepends the 'command' protocol code to the packaged data.
        packed_data = np.array(
            [
                Protocols.COMMAND.value,
                module_type,
                module_id,
                return_code,
                command,
                noblock,
                cycle,
                cycle_delay,
            ],
            dtype=np.uint8,
        )

        # Writes the packaged data into the transmission buffer.
        # noinspection PyTypeChecker
        self.transport_layer.write_data(data_object=packed_data)

        # Constructs and sends the data message to the connected system.
        self.transport_layer.send_data()

    def send_parameter_message(
        self,
        module_type: np.uint8,
        module_id: np.uint8,
        return_code: np.uint8,
        parameter_object: Any,
    ) -> None:
        # Ensures the transmission_buffer is cleared
        self.reset_reception_state()

        # Packages the header data for the parameter message and writes it to the transmission buffer
        packed_data = np.array(
            [
                Protocols.PARAMETERS.value,
                module_type,
                module_id,
                return_code,
            ],
            dtype=np.uint8,
        )

        # noinspection PyTypeChecker
        next_index = self.transport_layer.write_data(data_object=packed_data)

        size_index = next_index
        object_index = next_index + 1
        next_index = self.transport_layer.write_data(data_object=parameter_object, start_index=object_index)

        self.transport_layer.write_data(data_object=np.uint8(next_index - object_index), start_index=size_index)

        self.transport_layer.send_data()

    def receive_message(self) -> DataMessage:
        pass
