from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig
import time

config = DuckConfig()
hwi = HWI(config, "/dev/ttyAMA0")
hwi.set_kds([0] * len(hwi.joints))
hwi.turn_on()
time.sleep(1)

knees = []

for i, joint in enumerate(hwi.joints.keys()):
    if "knee" in joint:
        knees.append((joint, hwi.joints[joint], i))

for leg, joint_id, i in knees:
    print(f"Testing {leg} ...")
    hwi.io.set_kps([joint_id], [hwi.low_torque_kps[0]])
    current_pos = hwi.get_present_positions()[i]
    time.sleep(0.5)
    hwi.set_position(leg, current_pos + 0.5)
    time.sleep(0.5)

hwi.turn_off()
