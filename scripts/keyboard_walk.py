#!/usr/bin/env python3
"""
Simple keyboard-controlled walking without Xbox controller.
Run this script and use keyboard to send velocity commands.
"""
import os
import time
import sys
import termios
import tty
from threading import Thread
from v2_rl_walk_mujoco import RLWalk

HOME_DIR = os.path.expanduser("~")

class KeyboardController:
    def __init__(self, x_step=0.05, y_step=0.05, yaw_step=0.3,
                 x_max=0.15, y_max=0.15, yaw_max=0.6):
        self.x_vel = 0.0
        self.y_vel = 0.0
        self.yaw_vel = 0.0
        self.running = True
        self.x_step = x_step
        self.y_step = y_step
        self.yaw_step = yaw_step
        self.x_max = x_max
        self.y_max = y_max
        self.yaw_max = yaw_max

    def get_key(self):
        """Get a single keypress without waiting for Enter"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def print_status(self):
        """Print current velocity commands"""
        print(f"\rX: {self.x_vel:+.2f} m/s | Y: {self.y_vel:+.2f} m/s | Yaw: {self.yaw_vel:+.2f} rad/s | (q=quit, s=stop)    ", end='', flush=True)

    def keyboard_thread(self):
        """Background thread to read keyboard input"""
        print("\n" + "=" * 70)
        print("KEYBOARD CONTROLS")
        print("=" * 70)
        print("  w/s : Forward/Backward velocity (±0.05 m/s)")
        print("  a/d : Left/Right strafe velocity (±0.05 m/s)")
        print("  j/l : Rotate left/right (±0.3 rad/s)")
        print("  SPACE : Stop all motion (set velocities to 0)")
        print("  q : Quit")
        print("=" * 70)
        print()

        while self.running:
            try:
                key = self.get_key()

                if key == 'q':
                    self.running = False
                    print("\n\nQuitting...")
                    break
                elif key == ' ':  # Space bar
                    self.x_vel = 0.0
                    self.y_vel = 0.0
                    self.yaw_vel = 0.0
                elif key == 'w':
                    self.x_vel = min(self.x_max, self.x_vel + self.x_step)
                elif key == 's':
                    self.x_vel = max(-self.x_max, self.x_vel - self.x_step)
                elif key == 'a':
                    self.y_vel = min(self.y_max, self.y_vel + self.y_step)
                elif key == 'd':
                    self.y_vel = max(-self.y_max, self.y_vel - self.y_step)
                elif key == 'j':
                    self.yaw_vel = min(self.yaw_max, self.yaw_vel + self.yaw_step)
                elif key == 'l':
                    self.yaw_vel = max(-self.yaw_max, self.yaw_vel - self.yaw_step)

                self.print_status()

            except Exception as e:
                print(f"\nError reading keyboard: {e}")
                break

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{HOME_DIR}/BEST_WALK.npz",
                        help="path to the .npz weights file for NumpyInfer")
    parser.add_argument("-p", type=int, default=15, help="kp")
    parser.add_argument("--control_freq", type=int, default=50)
    parser.add_argument("--x_step", type=float, default=0.05)
    parser.add_argument("--y_step", type=float, default=0.05)
    parser.add_argument("--yaw_step", type=float, default=0.3)
    parser.add_argument("--x_max", type=float, default=0.15)
    parser.add_argument("--y_max", type=float, default=0.15)
    parser.add_argument("--yaw_max", type=float, default=0.6)
    args = parser.parse_args()

    print("=" * 70)
    print("KEYBOARD WALKING CONTROL")
    print("=" * 70)
    print()
    print(f"Model: {args.model}")
    print("Initializing robot...")

    rl_walk = RLWalk(
        args.model,
        control_freq=args.control_freq,
        pid=[args.p, 0, 0],
        commands=False,
        pitch_bias=0,
    )

    rl_walk.paused = False

    print("Robot initialized!")

    kb = KeyboardController(
        x_step=args.x_step, y_step=args.y_step, yaw_step=args.yaw_step,
        x_max=args.x_max, y_max=args.y_max, yaw_max=args.yaw_max,
    )

    # Start keyboard input thread
    kb_thread = Thread(target=kb.keyboard_thread, daemon=True)
    kb_thread.start()

    # Start walking control loop in background
    walk_thread = Thread(target=rl_walk.run, daemon=True)
    walk_thread.start()

    # Main loop - update commands from keyboard
    kb.print_status()
    try:
        while kb.running:
            # Update robot commands from keyboard
            rl_walk.last_commands[0] = kb.x_vel
            rl_walk.last_commands[1] = kb.y_vel
            rl_walk.last_commands[2] = kb.yaw_vel

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nInterrupted by Ctrl+C")

    # Stop robot
    print("\nStopping robot...")
    rl_walk.last_commands = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time.sleep(0.5)

    print("Done!")

if __name__ == "__main__":
    main()
