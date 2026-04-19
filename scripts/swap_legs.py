from pypot.feetech import FeetechSTS3215IO
import time

leg_pairs = [
    (20, 10),
    (21, 11),
    (22, 12),
    (23, 13),
    (24, 14),
]


def change_motor_id(io: FeetechSTS3215IO, current_id: int, new_id: int):
    io.set_lock({current_id: 0})
    io.set_mode({current_id: 0})
    io.set_maximum_acceleration({current_id: 0})
    io.set_acceleration({current_id: 0})
    io.set_P_coefficient({current_id: 32})
    io.set_I_coefficient({current_id: 0})
    io.set_D_coefficient({current_id: 0})
    io.change_id({current_id: new_id})

    current_id = new_id

    time.sleep(1)

    io.set_goal_position({current_id: 0})

    time.sleep(1)

    print("===")
    print("Done configuring motor.")
    print(f"Motor id: {current_id}")
    print(f"P coefficient : {io.get_P_coefficient([current_id])}")
    print(f"I coefficient : {io.get_I_coefficient([current_id])}")
    print(f"D coefficient : {io.get_D_coefficient([current_id])}")
    print(f"acceleration: {io.get_acceleration([current_id])}")
    print(f"max_acceleration: {io.get_maximum_acceleration([current_id])}")
    print(f"mode: {io.get_mode([current_id])}")
    print("===")


if __name__ == "__main__":
    io = FeetechSTS3215IO("/dev/ttyAMA0")
    for left_id, right_id in leg_pairs:
        print(f"Swapping {left_id} and {right_id} ...")
        change_motor_id(io, left_id, 5)
        change_motor_id(io, right_id, left_id)
        change_motor_id(io, 5, right_id)
        print(f"Done swapping {left_id} and {right_id} !")