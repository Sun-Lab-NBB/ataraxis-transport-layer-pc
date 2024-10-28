import numpy as np
from time import sleep
from ataraxis_time import PrecisionTimer

from src.ataraxis_transport_layer.communication import SerialCommunication, IdentificationMessage, ReceptionMessage, \
    DataMessage, CommandMessage

# Connect to the controller
com_class = SerialCommunication(usb_port='/dev/cu.usbmodem142018301')
timer = PrecisionTimer('us')
another_timer = PrecisionTimer('us')
message = CommandMessage(module_type=np.uint8(1), module_id=np.uint8(0), command=np.uint8(4),
                         return_code=np.uint8(187), noblock=np.bool(True), cycle=np.bool(True),
                         cycle_delay=np.uint32(654))

samples = 122
send_arr = np.zeros(samples, np.uint64)
rec_arr = np.zeros(samples, np.uint64)
id_arr = np.zeros(samples, np.uint64)
comm_rec_arr = np.zeros(samples, np.uint64)
comm_id_arr = np.zeros(samples, np.uint64)
for i in range(samples):
    print(i)
    # Sends an identification request
    timer.reset()
    another_timer.reset()
    com_class.send_command_message(message)
    send_arr[i] = timer.elapsed

    while 1:
        timer.reset()
        status, output = com_class.receive_message()  # Receives the reception code
        elapsed = timer.elapsed
        if status:
            comm_rec_arr[i] = another_timer.elapsed
            rec_arr[i] = elapsed
            break

    while 1:
        timer.reset()
        status, output = com_class.receive_message()  # Receives the controller ID
        elapsed = timer.elapsed
        if status:
            comm_id_arr[i] = another_timer.elapsed
            id_arr[i] = elapsed
            break

print('Send Command Time:')
print(f'{round(np.average(send_arr[2:]), 3)} +- {round(np.std(send_arr[2:]), 3)} us')

print('Receive Reception Code Time:')
print(f'{round(np.average(rec_arr[2:]), 3)} +- {round(np.std(rec_arr[2:]), 3)} us')

print('Receive ID Code Time:')
print(f'{round(np.average(id_arr[2:]), 3)} +- {round(np.std(id_arr[2:]), 3)} us')

print('Command to Reception Time:')
print(f'{round(np.average(comm_rec_arr[2:]), 3)} +- {round(np.std(comm_rec_arr[2:]), 3)} us')

print('Command to ID Time:')
print(f'{round(np.average(comm_id_arr[2:]), 3)} +- {round(np.std(comm_id_arr[2:]), 3)} us')
