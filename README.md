# Open Duck Mini Runtime

This repo is a fork of [Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime), part of the [Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) project from Antoine Pirrone. I am using this to document all the changes and additions I have made/will make in my version of the robot.

## Current Status

**April 20, 2026**: I have successfully got the walking policy to run on my build of the droid which uses an original Raspberry Pi Zero W as opposed to the Rasperry Pi Zero 2 W recommended for the project. This may sound like a trivial difference, but the key distinctions between the two are that the first Pi Zero does not support 64-bit operating systems and only has a single core while the Zero 2 has 4. This is an issue as the runtime utilizes ONNX, and the runtime does not have a build available for armv6. Furthermore, ONNX tries to uses NEON which is not available on armv6, so building the runtime for this architecture is not that straightforward. Ultimately, I was able to get the policy to run on this hardware by forgoing ONNX altogether and recreating the model and inference code with pure NumPy.


https://github.com/user-attachments/assets/b63a1b17-d0ae-4a2f-890a-13c5ce4b3efb

## Looking Ahead

My immediate next step will be to put together the battery system so that the droid will no longer be leashed to the power supply. I'll use this opportunity to make some of the electrical systems more permenant, as right now everything is just connected with jumper wires. I need to get the rest of the parts for the expression features as well. I am considering possibly adding a second Pi Zero to control this stuff and handle the connection to the external control. Although I did get it to work so far, I am concerned about the feasibility of the Pi Zero as a controller with more responsibilites added to it.

## New Files

Runtime
- `mini_bdx_runtime/mini_bdx_runtime/numpy_infer.py`: NumPy replacement of `onnx_infer.py`.

Data
- `BEST_WALK.npz`: Walk policy `BEST_WALK_ONNX_2.onnx` converted to NumPy.
- `pi_inference.json`: Observation + action pairs using the ONNX runtime for armv6 I compiled. Generated from `scripts/dump_inference_pi.py`.

ONNX Build
- `onnx_for_armv6/Dockerfile.arm6`: Dockerfile I used to cross compile ONNX runtime for armv6.
- `onnx_for_armv6/onnxruntime-1.18.1-cp313-cp313-linux_armv6l.whl`: ONNX runtime wheel for Python 3.13 on armv6. While this will successfully install and run on armv6, there is an error with the values it gets, so this **cannot be used!**.
- `onnx_for_armv6/onnxruntime_mlas.cmake`: MLAS CMake file with the NEON block removed.
- `onnx_for_armv6/mlasi.h`: Modified `onnxruntime/core/mlas/lib/mlasi.h` with NEON reference removed.
- `onnx_for_armv6/qgemm.h`: Modified `onnxruntime/core/mlas/lib/qgem.h` with NEON reference removed.

Scripts
- `scripts/check_legs.py`: Moves both legs so you can see check that the legs are mapped to the left and right sides correctly.
- `scripts/compare_inference_desktop.py`
- `scripts/dump_inference_pi.py`
- `scripts/keyboard_walk.py`: Allows control of the bot like `scripts/v2_rl_walk_mujoco.py` but using key presses instead of an Xbox controller.
- `scripts/lock_motors.py`: Sets the bot to the starting position and then locks the motors.
- `scripts/swap_legs.py`: Swaps the motors mapped to the "left" and "right" motor IDs.s
