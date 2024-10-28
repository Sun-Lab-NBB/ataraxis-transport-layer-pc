import json
from typing import Any

import numpy as np
import paho.mqtt.client as mqtt
from ataraxis_data_structures import SharedMemoryArray


class UnityCommunication:
    """Provides methods for communicating to Unity during cylindrical treadmill task.

    Args:
        ip: the ip address that Unity is using to create MQTT channels.
        port: the port that Unity is using to create MQTT channels.
        shm_name: the name of the reward shared memory array. If multiple UnityComm classes are created synchronously, each should have a unique shm_name.

    Attributes:
        reward_shm: a Shared Memory Array with a single value. This value starts as 0 and is set to 1 when the Unity system records that the mouse received a reward. The value of the first index of reward_shm is only set to 1 upon a reward if the connect method has been called.
        _broker: the ip address that Unity is using to create MQTT channels.
        _port: the port that Unity is using to create MQTT channels.
        _lick_topic: the name of the MQTT channel unity uses to communicate licks.
        _move_topic: the name of the MQTT channel unity uses to communicate movement.
        _reward_topic: the name of the MQTT channel unity uses to communicate rewards.
        _send_client: MQTT client subscribed to Unity's move and lick channels.
        _receive_client: MQTT client subscribed to Unity's reward channel. This additional client exclusively subscribes to Unity's reward channel, thus its listener recieves less triggers.
    """

    def __init__(self, ip: str = "127.0.0.1", port: int = 1883, shm_name: str = "reward_shm"):
        prototype = np.array([0], dtype=np.int32)

        self.reward_shm = SharedMemoryArray.create_array(
            name=shm_name,
            prototype=prototype,
        )

        self._broker: str = ip
        self._port: int = port

        self._lick_topic: str = "LickPort/"
        self._move_topic: str = "LinearTreadmill/Data"
        self._reward_topic: str = "Gimbl/Reward/"

        self._send_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # type: ignore

        # This additional client exclusively subscribes to Unity's reward channel, thus it only listens to reward signals. This will lead to less triggers to the on_message function.
        self._receive_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  # type: ignore

    def connect(self) -> None:
        """Connects to MQTT channels and the Shared Memory Array. Sets up a listener to modify reward_shm when a reward is recorded in Unity. Should be called before any calls to send_movement, send_lick, or reward_ocurred. Should be called before accessing reward_shm. Should be followed by a call to disconnect()."""
        self.reward_shm.connect()

        self._send_client.connect(self._broker, self._port)
        self._receive_client.connect(self._broker, self._port)

        self._send_client.subscribe(self._lick_topic)
        self._send_client.subscribe(self._move_topic)
        self._receive_client.subscribe(self._reward_topic)

        def on_message(client: mqtt.Client, userdata: Any, message: mqtt.MQTTMessage) -> None:
            if message.topic == self._reward_topic:
                self.reward_shm.write_data(0, 1)

        self._receive_client.on_message = on_message
        self._receive_client.loop_start()

    def send_movement(self, movement: float) -> None:
        """Sends movement to the Unity mouse object.

        Requires the connect method to have been called.


        Args:
            movement: the amount, in Unity units, to move the Unity mouse object.

        """
        json_string = json.dumps({"movement": movement})
        byte_array = json_string.encode("utf-8")
        self._send_client.publish(self._move_topic, byte_array)

    def send_lick(self) -> None:
        """Triggers a lick by the Unity mouse object.

        Requires the connect method to have been called.
        """
        self._send_client.publish(self._lick_topic)

    def reward_occurred(self) -> bool:
        """Returns True if a reward has occurred since the connect method has been called.

        Alternatively, the reward_shm array can be accessed directly.
        """
        if self.reward_shm.read_data(0) == 1:
            return True
        return False

    def reward_reset(self) -> None:
        """Resets the shared memory array such that reward_occurred will return False again until a new reward occurs."""
        self.reward_shm.write_data(0, 0)

    def disconnect(self) -> None:
        """Disconnects all channels and the Shared Memory Array."""
        self._receive_client.loop_stop()
        self._receive_client.disconnect()
        self._send_client.disconnect()
        self.reward_reset()
        self.reward_shm.disconnect()
