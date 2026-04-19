from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.duck_config import DuckConfig
import time
import json
import os

def main():
    default_config = DuckConfig(None)
    hwi = HWI(duck_config=default_config, usb_port="/dev/ttyAMA0")
    with open(f"{os.path.expanduser('~')}/duck_config.json", "r") as f:
        real_config = json.load(f)
    offsets = real_config["joints_offsets"]
    hwi.turn_on()
    print("Motors are now on and locked at position 0.")
    print("Applying offsets from real duck config...")
    hwi.set_position_all(offsets)
    print("Offsets applied. Motors are now locked at real starting positions. Press ctrl+c to unlock and exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Turning off motors...")
        hwi.turn_off()
        print("Motors are now off. Exiting.")

if __name__ == "__main__":
    main()