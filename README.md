## 🤖 G1HighlevelArm: Unitree G1 Dual-Arm + Hand High-Level Controller
This repository implements a **high-level controller for the Unitree G1 robot**, enabling coordinated arm and hand manipulation with trajectory recording, replay, and vision QR-code based triggering.

### 📆 Overview

| Module | Description |
|--------|-------------|
| `g1_high_level_controller.py` | Main controller: dual-arm control, hand gestures, remote logic, trajectory handling |
| `g1_highlevel_hand.py`        | Dex3 hand controller with gesture switching and torque control |
| `g1_arm_IK.py`                | IK solver based on Pinocchio |
| `g1_head_image.py`            | Real-time QR detection and depth estimation (RealSense + QRDet) |

---

### 🛠️ Installation

```bash
conda create -n g1_highlevel python=3.8
conda activate g1_highlevel
Please refer to the official RL setup from unitree: 
pip install numpy opencv-python scipy PyYAML pinocchio casadi pyrealsense2 torch

# Install Unitree SDK2 Python bindings manually
# Install QRDet (https://github.com/NVlabs/qrdet)
```

---

### 🚀 Launch Controller

```bash
python deploy_real/g1_high_level_controller.py
```

On startup:
- Initializes DDS communication
- Sets zero-torque mode → damping mode → move to default pose
- Starts control loop and remote listener

---

### 🎮 Gamepad Controls (Logitech F710 or similar)

| Button | Function |
|--------|----------|
| `L1`   | Play sequence A (grasp + return) |
| `L2`   | Enter visual detection mode, auto-trigger sequence B (this can be change to button control) |
| `R1`   | Hand GRIP gesture |
| `R2`   | Hand RELEASE gesture |
| `X`    | Move to default pose |
| `Y`    | Zero torque mode (floating) |
| `A`    | Start trajectory recording (zero-torque for the arm)|
| `B`    | Stop & save recording (Arm will hold the current position)|
| `↑`  | Prepare selected trajectory from motion bank |
| `↓`  | Play selected trajectory |
| `→`  | Switch hand to DEFAULT gesture |
| `Select` | Safe shutdown |
The recorded traj will be saved to `records`, you can type the name of the traj file and press ENTER to select the file. Then use `↑` to prepare and use `↓` to play.
---

### 🤖 Arm + Hand Features

#### 🖐 Dex3 Gesture Control
- Predefined gestures: `DEFAULT`, `GRIP`, `RELEASE` (These position can be customized)
- Zero torque / damping / motion control mode
- High-frequency DDS state update and command publish

#### ⚖️ Inverse Kinematics (IK)
- Based on `casadi` + `pinocchio`
- Adds `L_ee`, `R_ee` frames for left/right wrist targets
- Regularized optimization with smoothing and constraints

#### 📷 Vision-Triggered Actions ()
- Detect QR codes via RealSense + QRDet
- Estimate distance + angle
- Trigger sequence B only if within spatial & angular bounds for N seconds
- Writes `l2_trigger_state.txt` to record event

---

### 🔹 Trajectory Record & Replay

#### Format
- `.npz` file with: `traj`, `dt`, `note`
- Frame = `[q(12), pos_L(3), quat_L(4), pos_R(3), quat_R(4)]`

#### Modes
- `joint`: replay joint positions
- `workspace`: replay hand poses using IK

#### Example Usage
```python
from g1_high_level_controller import G1HighlevelArmController
ctrl = G1HighlevelArmController()
ctrl.start()
ctrl.prepare_replay("records/traj_example.npz", speed=1.0, mode="workspace")
ctrl.do_replay()
```

---

### 📂 File-based Triggering

- Path: `l2_trigger_state.txt`
- Log-based triggering with entries like `L2_pressed`
- Avoids re-triggering if already logged
- Useful for ROS or external process integration

---

### 📹 Debug: QRCode Visualizer

```bash
python g1_head_image.py
```
- Displays live camera feed
- Visualizes detection, orientation, distance
- Helps tune thresholds and test `VisionQRDetector`

---

## 🔗 References

- [PBHC (KungfuBot)](https://github.com/shi-soul/KungfuBot)
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)
- [QRDet](https://github.com/NVlabs/qrdet)

---

## 📧 Contact

Feel free to raise issues or email the authors. Feedback and suggestions are welcome!
