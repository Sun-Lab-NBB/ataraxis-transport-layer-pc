from tqdm import tqdm
import numpy as np
from src.serial_transfer_protocol import SerializedTransferProtocol
from src.ataraxis_transport_layer.helper_modules import ElapsedTimer

protocol = SerializedTransferProtocol(
    port="COM7",
    baudrate=115200,
    polynomial=np.uint16(0x1021),
    initial_crc_value=np.uint16(0xFFFF),
    final_crc_xor_value=np.uint16(0x0000),
    maximum_transmitted_payload_size=np.uint8(254),
    minimum_received_payload_size=1,
    start_byte=np.uint8(129),
    delimiter_byte=np.uint8(0),
    timeout=np.uint64(20000),
    test_mode=False,
    allow_start_byte_errors=False,
)

test_array = np.array(
    [1, 2, 3, 4, 5, 6, 7],
    dtype=np.uint8,
)
microcontroller_time = np.uint32(0)
received_array = np.zeros(test_array.size, dtype=np.uint8)

timer = ElapsedTimer("us")
deltas_write = []
deltas_send = []
deltas_wait = []
deltas_microcontroller = []
deltas_receive = []
deltas_read = []
deltas_total = []
test_count = 100000

for test_num in tqdm(range(test_count), desc="Running communication cycles", ncols=120):
    # Writes test data to buffer
    timer.reset()
    protocol.write_data(test_array)
    deltas_write.append(timer.elapsed)

    timer.reset()
    protocol.send_data()
    deltas_send.append(timer.elapsed)

    # Waits for the data to come back from the controller
    timer.reset()
    while not protocol.available:
        pass
    deltas_wait.append(timer.elapsed)

    # Receives and parses the data
    timer.reset()
    protocol.receive_data()
    deltas_receive.append(timer.elapsed)

    timer.reset()
    received_array, index = protocol.read_data(test_array, 0)
    microcontroller_time, _ = protocol.read_data(microcontroller_time, index)
    deltas_read.append(timer.elapsed)

    deltas_microcontroller.append(microcontroller_time)

    # Verifies data integrity
    assert np.array_equal(test_array, received_array), f"Error {received_array}"

# Factors out numba compilation (happens at iteration one and calculates the average delays in microseconds for all
# tracker execution steps
average_write = np.around(np.mean(deltas_write[10::]), 3)
stdev_write = np.around(np.std(deltas_write[10::]), 3)
average_send = np.around(np.mean(deltas_send[10:]), 3)
stdev_send = np.around(np.std(deltas_send[10::]), 3)
average_receive = np.around(np.mean(deltas_receive[10:]), 3)
stdev_receive = np.around(np.std(deltas_receive[10::]), 3)
average_read = np.around(np.mean(deltas_read[10:]), 3)
stdev_read = np.around(np.std(deltas_read[10::]), 3)

# Combined PC stats
pc_average = np.around(average_write + average_send + average_receive + average_read, 3)
pc_std = np.around(stdev_write + stdev_send + stdev_receive + stdev_read, 3)

# Microcontroller + Transmission stats
amc_average = np.around(np.mean(deltas_microcontroller[10:]), 3)
amc_std = np.around(np.std(deltas_microcontroller[10::]), 3)

wait_average = np.around(np.mean(np.array(deltas_wait[10:]) - np.array(deltas_microcontroller[10:])), 3)
wait_std = np.around(np.std(np.array(deltas_wait[10:]) - np.array(deltas_microcontroller[10:])), 3)

for test_num in tqdm(range(test_count), desc="Running communication cycles", ncols=120):
    # Writes test data to buffer
    timer.reset()
    protocol.write_data(test_array)
    protocol.send_data()

    # Waits for the data to come back from the controller
    while not protocol.receive_data():
        pass

    received_array, index = protocol.read_data(test_array, 0)
    microcontroller_time, _ = protocol.read_data(microcontroller_time, index)
    deltas_total.append(timer.elapsed)

    # Verifies data integrity
    assert np.array_equal(test_array, received_array), f"Error {received_array}"

# Total cycle (PC + Controller + Transmission) stats
average = np.around(np.mean(deltas_total[10:]), 3)
std = np.around(np.std(deltas_total[10:]), 3)

# Boasts about the results
print(
    f"Passed a total of {test_count} tests. \nTook {average_write} +- {stdev_write} us to write data to staging "
    f"buffer. \nTook {average_send} +- {stdev_send} us to send the data to the microcontroller. \nTook "
    f"{average_receive} +- {stdev_receive} us to parse response data. \nTook {average_read} +- {stdev_read} us to read "
    f"the data. \nThe PC portion of the loop took {pc_average} +- {pc_std} us per cycle. "
    f"\nThe microcontroller portion of the loop took {amc_average} +- {amc_std} us per cycle. "
    f"\nThe transmission portion of the loop took {wait_average} +- {wait_std} us per cycle. "
    f"\nThe total cycle (PC + Controller + Transmission) took {average} +- {std} us."
)
