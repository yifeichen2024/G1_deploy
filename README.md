# 🔊 Unitree G1 deployment Overview

This repository provides a complete **high-level and low-level control framework for the Unitree G1 humanoid robot**, supporting both precise **dual-arm manipulation** and **whole-body motion deployment** with reinforcement learning policies.

### 🔧 Key Features
- 🦾 **Dual-Arm + Dex3 Hand Control**: With trajectory recording, replay, and QR-code triggered responses
- 🤖 **Whole-Body Motion Deployment**: Execute learned locomotion and behavior policies from simulation (e.g. PBHC) on the real G1 robot
- 🎮 **Gamepad-Driven Operation**: Seamless integration with Logitech controllers for safe and interactive testing
- ⚙️ **ONNX Inference Engine**: Supports both single and multi-policy deployment with safety protection

---
## 🤖 G1HighlevelArm: Unitree G1 Dual-Arm + Hand High-Level Controller
This repository implements a **high-level controller for the Unitree G1 robot**, enabling coordinated arm and hand manipulation with trajectory recording, replay, and vision QR-code based triggering. This will be used for teleop framework.
(This don't need to go into the debug mode.)
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

### 🎮 Gamepad Controls (Unitree remote controller or similar xbox controller)

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

## 🚀 Real-World Deployment of Whole-Body Motion Policies

Here is a deployment pipeline I used for whole-body motion policies trained in simulation (e.g., PBHC) and executed on the **Unitree G1 humanoid robot** in the real world. 
This need to go into the debug mode. This function is keep updating, there are still some problems.

Supports:
- ✅ Deploying a **single ONNX policy** (e.g., walking, stance)
- 🔁 **Multi-policy switching** at runtime with controller buttons
- ⚖️ Safe control transition: zero-torque → default pose → policy execution
- ⚠️ Built-in **action/tau/speed limit checks** for hardware safety
---

### 📁 Scripts

| File | Description |
|------|-------------|
| `deploy_real_onnx.py` | Main single-policy deployment |
| `deploy_real_onnx_multi.py` | Multi-policy version with runtime switching (e.g., walk ↔ balance) |

---

### 📝 Requirements

- ONNX or pytorch model file exported from IsaacGym training (e.g., PBHC policy) 
- Real Unitree G1 robot with **Unitree SDK2** DDS interface
- Config `.yaml` file under `deploy_real/configs/`
- Network interface (e.g., `enp2s0`, `eth0`) connected to the robot

---

### 🔧 Launching Deployment

#### ✅ Single ONNX Model
```bash
python deploy_real_onnx.py <net> <config.yaml>
# Example:
python deploy_real_onnx.py enp2s0 g1_29dof_PBHC.yaml
```

#### ⇆ Multi-Policy Mode
```bash
python deploy_real_onnx_multi.py <net> <config.yaml>
```
YAML config must define:
```yaml
policy_paths:
  - path/to/stance.onnx
  - path/to/walk.onnx
  - path/to/humanlike.onnx
motion_lens: [10.0, 1.5, 2.5]   # seconds per motion
```

---

### 🎮 Remote Controller Mapping (Logitech F710)

| Button | Function |
|--------|---------|
| `START` | Proceed from zero torque to next phase |
| `A`     | Execute default pose alignment |
| `SELECT`| Exit safely |
| `X`     | Reset to default pose |
| `L1`    | Switch to previous policy (multi-mode only) |
| `R1`    | Switch to next policy (multi-mode only) |

---

### 🔄 Runtime Flow

1. **Zero Torque State**  (floating, safe)
2. **Move to Default Pose**
3. **Wait for A → Apply Default Pose Command**
4. **Enter Execution Loop**:
   - Reads joint + IMU state
   - Builds observation buffer
   - Feeds ONNX model → gets action
   - Applies action to actuators with KP/KD
   - Performs safety checks (velocity, torque, NaN)
   - Switch policy on L1/R1 (multi only)

---

### 🌀 Observation Vector

Input to policy includes:
- Angular velocity (scaled)
- DOF position / velocity (scaled)
- Projected gravity (IMU)
- Action history, velocity history, gravity history
- Phase encoding for periodic motion (e.g. gait cycle)

---

### ⚠️ Safety Features

- Action range warning: if |action| > `threshold`, log warning
- Torque limit check: abort if `tau_est` > max
- Joint speed check: abort if `dqj` > limit
- Fallback to **damping command** on error or `SELECT`
- Multi-mode controller plays motions and returns to `stance` after completion.

---

### 🔍 Debug Tips

- Logs are written to `deploy_log_YYYYMMDD_HHMMSS.txt`
- Use `X` to reset pose mid-execution
- Use `print()` inside `run()` to monitor action values

---

## 🔗 References

- [PBHC (KungfuBot)](https://github.com/shi-soul/KungfuBot)
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)
- [QRDet](https://github.com/NVlabs/qrdet)

---

## 📧 Contact

Feel free to raise issues or email the authors. Feedback and suggestions are welcome!
