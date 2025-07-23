import time, sys, json, pathlib
import numpy as np
import yaml, pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp
from enum import Enum, auto
from collections import deque
import os
import select
import threading 

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC

from common.remote_controller import RemoteController, KeyMap
from g1_arm_IK import G1_29_ArmIK
from g1_highlevel_hand import Dex3GestureController, HandGesture, _RIS_Mode, _getch
from vision_detector import VisionQRDetector

# -----------------------------------------------------------------------------
# G1 Joint Index
# -----------------------------------------------------------------------------
class G1JointIndex:
    # only arm joints
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    RightElbow = 25
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    kNotUsedJoint = 29

# -----------------------------------------------------------------------------
# Load config for action_joints, control_dt, default_angles
# -----------------------------------------------------------------------------
class Config: pass

def load_cfg(path="deploy_real/configs/config_high_level.yaml"):
    with open(path, 'r') as f:
        d = yaml.safe_load(f)
    cfg = Config()
    for k, v in d.items():
        setattr(cfg, k, np.array(v) if isinstance(v, list) else v)
    cfg.kps_record = cfg.kps_play * 0
    cfg.kds_record = cfg.kds_play * 0
    cfg.replay_transition_duration = 1.5
    return cfg

cfg = load_cfg()

class Mode(Enum):
    IDLE      = auto()
    HOLD      = auto()          # Hold current position
    IK_STREAM = auto()          # online IK， for testing
    PLAY      = auto()          # Play the record trajectory
    WAIT_SEQ_B = auto()
# -----------------------------------------------------------------------------
# G1 Arm controller
# -----------------------------------------------------------------------------
class G1HighlevelArmController:
    def __init__(self, record_dir="records", history_len=3):
        # --- com and the kinematics ---
        self.low_state   : LowState_ | None = None
        self.first_state = False
        self.low_cmd     = unitree_hg_msg_dds__LowCmd_()
        self.crc         = CRC()
        self.remote      = RemoteController()
        self.prev_buttons = np.zeros_like(self.remote.button, dtype=int)
        self.ik          = G1_29_ArmIK(Unit_Test=False, Visualization=False)

        # DDS
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_);  self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self._cb, 10)
        print("[DDS] Arm Publisher & Subscriber ready.")

        # Hand controller
        self.dex3 = Dex3GestureController(fps=30.0)
        self.dex3.switch_gesture(HandGesture.DEFAULT)  # the default state is close hand.
        
        # --- control param ---
        self.mode         = Mode.IDLE
        self.target_q     = cfg.default_angles.copy()
        self.kps = np.zeros_like(cfg.kps_play)
        self.kds = np.ones_like(cfg.kds_play)

        self.history      = deque(maxlen=history_len)   # joint history.
        self.thread       = None

        # --- traj record/play ---
        self._recording     = False
        self._record_buffer = []
        self.record_dir     = pathlib.Path(record_dir); self.record_dir.mkdir(exist_ok=True)
        self._play_traj     = None      # np.ndarray(step, dof)
        self._play_idx      = 0

        # play buffer
        self._replay_traj     = None   # dict: {t, q, Mf_L, Mf_R}
        self._replay_idx      = 0
        self._replay_speed    = 1.0
        self._replay_mode     = "joint"   # "joint" or "workspace"
        self._replay_ready    = False

        # motion bank
        self._build_motion_bank()      # build motion bank for future usage
        self._sel_motion_idx = 0       # the index of current selected motion.
        
        # —— visual detection for QR code 
        self.vision = VisionQRDetector(model_size='s')
        self.detection_active     = False
        self.detect_start_time    = None
        # stable detection time 
        self.seq_hold_time = 3  # when the detection distance is in the range for 3 seconds. activate the hand out sequence.

        # —— L2 pressed state for com with the voice assistant —— 
        self.flag_path = pathlib.Path("l2_trigger_state.txt")
        self.flag_path.write_text("None\n")
        # if file not exist, create and write in None.
        if not self.flag_path.exists():
            self.flag_path.write_text("None\n")
            print(f"[INIT] create file {self.flag_path.name}, write in None")

        self.ready2placebill = False 
    
    def _build_motion_bank(self):
        """scan the motion bank, list all the *.npz file name and put into a list."""
        files = sorted(pathlib.Path(self.record_dir).glob("*.npz"))
        self.motion_names = [f.stem for f in files]          
        self.motion_files = [str(f) for f in files]          
        print("[MOTION BANK] loaded:", ", ".join(self.motion_names))

    def _cb(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True
        self.remote.set(msg.wireless_remote) # remote controller connection 

    def _send_joint(self, q, kps=None, kds=None):
        '''This part in charge of sending all the ctrl value to the lowlevel motor for the arm.'''

        kps = np.asarray(kps if kps is not None else self.kps)
        kds = np.asarray(kds if kds is not None else self.kds)

        for i, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q  = float(q[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(kps[i])
            self.low_cmd.motor_cmd[m].kd = float(kds[i])
            self.low_cmd.motor_cmd[m].tau= 0.0

        # fixed joint value. 
        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q  = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau= 0.0

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


    def current_q(self):
        '''
        read the current motor's state q, from lowlevel. 
        Fixed joint not included.
        '''
        return np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])

    def FK(self, q):
        '''
        forward kinematics
        param:  q: arm motor joint state (n-joints)
        return: tuple: tuple(left_hand_f, right_hand_f) of (pin.SE3)
        '''
        pin.forwardKinematics(self.ik.reduced_robot.model, self.ik.reduced_robot.data, q)
        pin.updateFramePlacements(self.ik.reduced_robot.model, self.ik.reduced_robot.data)
        d = self.ik.reduced_robot.data
        return d.oMf[self.ik.L_hand_id], d.oMf[self.ik.R_hand_id]

    def IK(self, poseL: pin.SE3, poseR: pin.SE3):
        '''
        inverse kinematics
        param:  poseL: left  hand end-effector pose
                poseR: right hand end-effector pose
        return: q   : arm motor joint state (n-joints)
        '''
        q_now = self.current_q()
        q_cmd, _ = self.ik.solve_ik(poseL.homogeneous, poseR.homogeneous, current_lr_arm_motor_q=q_now)
        return q_cmd

    def move_to(self, poseL: pin.SE3, poseR: pin.SE3,
            ws_steps=50, jnt_steps=20):
        """
        move the two hand from the current pos to the given pos
        params: poseL: the pose of the L hand target pos (pin.SE3)
                poseR: the pose of the R hand target pos (pin.SE3)
               ws_steps: control step of pos when moving to the target. (int, dafault: 50)
               jnt_steps: control step of joint when moving to the target. (int, dafault: 20)
        """
        # 1. WS
        cur_L, cur_R = self.FK(self.current_q())
        for T_L, T_R in zip(self._interpolate_pose(cur_L, poseL, ws_steps),
                    self._interpolate_pose(cur_R, poseR, ws_steps)):
            self.target_q = self.IK(T_L, T_R)
            # self._send_joint(self.IK(T_L, T_R))
            time.sleep(cfg.control_dt)

        # 2. Joint fine‐tune
        q_goal = self.IK(poseL, poseR)
        q0     = self.current_q()
        for a in np.linspace(0, 1, jnt_steps):
            # self._send_joint((1-a)*q0 + a*q_goal)
            self.target_q = (1-a)*q0 + a*q_goal
            time.sleep(cfg.control_dt)

        self.target_q = q_goal


    def move_to_default(self, duration=3.0):
        '''
        Move the two arm to default position.
        param:  duration: float
        '''
        pos_L = np.array([ 0.10571,  0.18578, -0.10308], dtype=float)
        rot_L = np.array([
            [ 0.4695 ,  0.384  ,  0.79506],
            [ 0.0372 ,  0.89107, -0.45234],
            [-0.88215,  0.24195,  0.40407],
        ], dtype=float)

        pos_R = np.array([ 0.12127, -0.20089, -0.08074], dtype=float)
        rot_R = np.array([
            [ 0.62136, -0.28825,  0.72857],
            [-0.13909,  0.87452,  0.46462],
            [-0.77108, -0.39004,  0.5033 ],
        ], dtype=float)
        poseL_def = pin.SE3(rot_L, pos_L)
        poseR_def = pin.SE3(rot_R, pos_R)
        
        # poseL_def, poseR_def = self.FK(cfg.default_angles)  

        self.kps = cfg.kps_play.copy() * 0.5 # when moving to default, use a more smaller kp for safty.
        self.kds = cfg.kds_play.copy() 
        
        self.move_to(poseL_def, poseR_def,
                    ws_steps=int(duration/2/cfg.control_dt),
                    jnt_steps=int(duration/2/cfg.control_dt))
        print(f"[HOLD] Move to default.")


    def _interpolate_pose(self, T0: pin.SE3, T1: pin.SE3, n: int):
        """interpolate between one pose to another. return the interpolate pose matrix."""
        # 旋转：Slerp 
        R0, R1 = Rotation.from_matrix(T0.rotation), Rotation.from_matrix(T1.rotation)
        key_times = [0, 1]

        rot_seq = Rotation.from_matrix([R0.as_matrix(), R1.as_matrix()])
        slerp = Slerp(key_times, rot_seq)
        # 平移：线性
        for a in np.linspace(0, 1, n):
            rot = slerp(a).as_matrix()
            pos = (1 - a) * T0.translation + a * T1.translation
            # if a in (0,1):
            #     print(f"[SLERP] a={a:.1f}, pos={pos}, rot00={rot[0,0]:.3f}")
            yield pin.SE3(rot, pos)

    def zero_torque_mode(self):
        '''Create zero torque for the motors.'''
        self.mode = Mode.HOLD
        self.target_q = self.current_q()
        self.kps = np.zeros_like(cfg.kps_play)
        self.kds = np.zeros_like(cfg.kds_play)

    def damping_mode(self, kd=2):
        '''create damping ctrl for the motors'''
        self.mode = Mode.HOLD
        self.target_q = self.current_q()
        self.kps = np.zeros_like(cfg.kps_play)
        self.kds = np.ones_like(cfg.kds_play)*kd
    
     # --------------- RECORD -----------------
    def start_record(self):
        '''Start recording part, set the kps and kds to close to zero stiffness'''
        self._recording = True
        self.target_q = self.current_q()

        self.kps = (cfg.kps_play*cfg.stiffness_factor).copy()
        self.kds = (cfg.kds_play*cfg.stiffness_factor).copy()

        self._record_buffer.clear()
        print("[REC] start.")

    def stop_record(self, save=True, name=None):
        '''Stop the recording and save all the traj information. And set the target position to be the current position to hold.'''
        self._recording = False

        self.target_q = self.current_q()
        # for idx, joint in enumerate(cfg.action_joints):
        self.kps = (cfg.kps_play).copy()
        self.kds = (cfg.kds_play).copy()

        traj = np.vstack(self._record_buffer)
        if save:
            fn = self.record_dir / (name or f"traj_{int(time.time())}.npz")
            np.savez_compressed(fn, traj=traj,
                                dt=cfg.control_dt,
                                note="cols=[q(12)|pL(3)|qL(4)|pR(3)|qR(4)]") 
            print(f"[REC] save {fn}, shape {traj.shape}")
        return traj
    
    def _pack_frame(self):
        '''the record data and format. '''
        q   = self.current_q()
        TL, TR = self.FK(q)
        quatL = Rotation.from_matrix(TL.rotation).as_quat()   
        quatR = Rotation.from_matrix(TR.rotation).as_quat()    
        return np.hstack([q,
                        TL.translation, quatL,
                        TR.translation, quatR])

    def _frame_dim(self): 
        '''Record traj dim'''
        return len(cfg.action_joints) + 7 + 7  #  quat
    
    # --------------- PLAY -----------------
    def _load_traj(self, fp: str):
        """
        load the recorded *.npz file
        """
        data = np.load(fp)
        if "traj" in data:          
            arr = data["traj"]
            q_dof = len(cfg.action_joints)
            t = np.arange(len(arr))*cfg.control_dt
            q   = arr[:, :q_dof]
            pL  = arr[:, q_dof:q_dof+3]
            qL  = arr[:, q_dof+3:q_dof+7]
            pR  = arr[:, q_dof+7:q_dof+10]
            qR  = arr[:, q_dof+10:q_dof+14]
            Mf_L = [pin.SE3(pin.Quaternion(qL[i]), pL[i]) for i in range(len(arr))]
            Mf_R = [pin.SE3(pin.Quaternion(qR[i]), pR[i]) for i in range(len(arr))]
        else:                       # WorkspaceRecorder generate.
            t, q, Mf_L, Mf_R = data["t"], data["q"], data["Mf_L"], data["Mf_R"]
        return dict(t=t, q=q, Mf_L=Mf_L, Mf_R=Mf_R)
    
    def prepare_replay(self, file_path: str, speed=1.0, mode="workspace"):
        """prepare to replay the traj. transit to the first frame before start the replay. """
        self._replay_traj  = self._load_traj(file_path)
        self._replay_idx   = 0
        self._replay_speed = speed
        self._replay_mode  = mode
        self._replay_ready = False
        self._replay_t_arr   = self._replay_traj["t"] / self._replay_speed
        self._replay_start_t = time.time()

        # current state to the first state of the traj.
        cur_q = self.current_q()
        cur_L, cur_R   = self.FK(cur_q)
        tgt_L_h, tgt_R_h = self._replay_traj["Mf_L"][0], self._replay_traj["Mf_R"][0]
        steps = int(cfg.replay_transition_duration / cfg.control_dt)
        for T_L, T_R in zip(self._interpolate_pose(cur_L, tgt_L_h, steps),
                            self._interpolate_pose(cur_R, tgt_R_h, steps)): 
            self.target_q = self.IK(T_L, T_R)
            time.sleep(cfg.control_dt)
        self._replay_ready = True    # mark the the prepare is finished.
        print(f"[REPLAY] Transition complete, ready to play {file_path}.")
        self.mode = Mode.HOLD 

    def do_replay(self):
        '''Start to replay. It can only play after it is prepared to ensure the motion is smooth.'''
        if not self._replay_ready:
            print("[REPLAY] Please do prepare_replay() first")
            return
        
        self.mode = Mode.PLAY
        self._replay_idx = 0                # replay from the begining 
        self._replay_start_t = time.time()   # replay time start from now 
        print(f"[PLAY] start, mode={self._replay_mode}, len={len(self._replay_traj['t'])}")
        
    def example_lift_hands(self, dz=0.05, steps=50):
        """example to use the IK. it will lift two hands."""
        self.mode = Mode.IK_STREAM
        q0 = self.current_q()
        poseL0, poseR0 = self.FK(q0)

        self.kps = cfg.kps_play.copy()
        self.kds = cfg.kds_play.copy()

        for a in np.linspace(0, 1, steps):
            poseL = pin.SE3(poseL0.rotation, poseL0.translation + np.array([0.05, 0, a*dz]))
            poseR = pin.SE3(poseR0.rotation, poseR0.translation + np.array([0.05, 0, a*dz]))
            self.target_q = self.IK(poseL, poseR)   
            time.sleep(cfg.control_dt)
        self.mode = Mode.HOLD

    def start(self):
        while not self.first_state:
            print("[INIT] waiting lowstate…")
            time.sleep(0.1)

        # start the low level control loop.
        self.thread = RecurrentThread(interval=cfg.control_dt,
                                      target=self._control_loop,
                                      name="hl_arm_loop")
        self.thread.Start()

        print("[INIT] control thread started.")

        self.zero_torque_mode() 
        print("[HL] Ender zero torque mode.")
        time.sleep(0.5)
        self.damping_mode(kd=1.0)
        print("[HL] Enter damping mode.")
        time.sleep(0.5)
        self.dex3.switch_gesture(HandGesture.DEFAULT)
        self.move_to_default(duration=3)
        print("[HL] Move to default.")

    def stop(self):
        print("[HL] stopping controller…")
        self.damping_mode(kd=2)
        self.dex3.zero_torque()
        self.dex3.running = False

        if self.thread and self.thread.IsRunning():
            self.thread.Stop()
            self.thread.Join()         

    def _control_loop(self):
        if self.low_state is None:
            return 

        # 1) Giving orders based on the state.
        if self.mode == Mode.HOLD:
            self._send_joint(self.target_q, kps=self.kps, kds=self.kds)  # target_q kps kds are write based on the external functions.
        elif self.mode == Mode.IK_STREAM:
            self._send_joint(self.target_q, kps=self.kps, kds=self.kds)  
        elif self.mode == Mode.PLAY and self._replay_traj is not None:
                self.kps = cfg.kps_play.copy()
                self.kds = cfg.kds_play.copy()

                if self._replay_idx >=  len(self._replay_t_arr):
                    # finish playing 
                    self.mode = Mode.HOLD

                    # DEBUG
                    # self._replay_traj = None
                    self._replay_ready = False   # replay needed 

                    print("[PLAY] Finished.")
                    self.target_q = self.current_q()
                    Mf_L, Mf_R = self.FK(self.target_q)
                    
                    print(f"[PLAY] finished. Left: {Mf_L.translation}, Right: {Mf_R.translation}")
                    return
                
                elapsed = time.time() - self._replay_start_t
                while (self._replay_idx < len(self._replay_t_arr)
                    and self._replay_t_arr[self._replay_idx] <= elapsed):
                    self._replay_idx += 1

                # ensure it is in the boudary.
                if self._replay_idx >= len(self._replay_t_arr):
                    return                      
                
                frame_i = self._replay_idx
                if self._replay_mode == "joint":
                    q_cmd = self._replay_traj["q"][frame_i]
                else:   # workspace
                    MfL_h = self._replay_traj["Mf_L"][frame_i]
                    MfR_h = self._replay_traj["Mf_R"][frame_i]
                    q_cmd = self.IK(MfL_h, MfR_h)
                self._send_joint(q_cmd, cfg.kps_play, cfg.kds_play)
                # self._replay_idx += 1

        # in recording mode, store the data in to the buffer.
        if self._recording:
            self._record_buffer.append(self._pack_frame())

    def play_sequence_a(self):
        '''Play a customized sequence for grasp a bill book. you can also see this as a example to plan other movements.'''
        if not self._replay_ready:
            try:
                self.prepare_replay("records/traj_17_1.npz", speed=1.0, mode="workspace")
            except Exception as e:
                print("[A-SEQUENCE] traj_17_1 load failed:", e)
                return
        self.do_replay()
        while self.mode == Mode.PLAY:
            time.sleep(0.01)

        # switch gesture.
        self.dex3.switch_gesture(HandGesture.RELEASE)
        print("[A-SEQUENCE] Release, wait 3s")
        time.sleep(3.0)
        print("[A-SEQUENCE] Grip")
        self.dex3.switch_gesture(HandGesture.GRIP)
        time.sleep(0.5)
        
        try:
            self.prepare_replay("records/traj_17_2.npz", speed=1.0, mode="workspace")
            self.do_replay()
        except Exception as e:
            print("[A-SEQUENCE] traj_17_2 load failed:", e)
            return
        return 

    def play_sequence_b(self):
        '''
        Play a customized sequence for hand out and drop a bill book. 
        you can also see this as a example to plan other movements.
        '''
        try:
            self.prepare_replay("records/traj_17_3.npz", speed=1.0, mode="workspace")
            self.do_replay()
        except Exception as e:
            print("[B-SEQUENCE] traj_17_3 load failed:", e)
            return
        while self.mode == Mode.PLAY:
            time.sleep(0.01)

        self.dex3.switch_gesture(HandGesture.RELEASE)
        print("[B-SEQUENCE] Release. ")
        time.sleep(1)
        # self.dex3.switch_gesture(HandGesture.DEFAULT)
        
        try:
            self.prepare_replay("records/traj_17_4.npz", speed=1.0, mode="workspace")
            self.dex3.switch_gesture(HandGesture.DEFAULT)
            self.do_replay()
        except Exception as e:
            print("[B-SEQUENCE] traj_17_4 load failed:", e)
            return
        while self.mode == Mode.PLAY:
            time.sleep(0.01)
        
        self.move_to_default(3.0)

        return 

    def remote_poll(self):
        """
        Call this function in the main thread(loop). it constantly read the remote controller buttons.
        You can also automate this using function call
        """
        r = self.remote.button

        if r[KeyMap.L1] == 1:     
            print("[SEQUENCE A] started.")
            # threading.Thread(target=self.play_sequence_a, daemon=True).start()
            self.play_sequence_a()

            # === Printing current state used in testing and debugging ===
            # poseL, poseR = self.FK(self.current_q())
            # print("[L1] EE pos L:", poseL.translation, "R:", poseR.translation)
            # print("[L1] EE ori L:", poseL.rotation, "R:", poseR.rotation)
            # print(f"[STATE] L: {np.round(self.dex3.left_state,3)} | R: {np.round(self.dex3.right_state,3)}", end="\r")
            # time.sleep(0.1)

        # vision detection 
        # —— start vision detection —— 
        if self.detection_active:
            z, angle = self.vision.get_pose()
            now = time.time()
            print(f"[DEBUG] {z}, {angle}") 
            # the QR code detection range can be customized
            ok = (z is not None) and (0.57 <= z <= 0.61) and (-8 <= angle <= 8) # and (-166 <= x_px <= -110) and (-50 <= y_px <= 50)
            if ok:
                
                # existing = None
                # if self.flag_path.exists():
                #     existing = self.flag_path.read_text().strip()
                # # 只有当内容不是 "L2_pressed" 时才写入
                # if existing != "L2_pressed":
                #     self.flag_path.write_text("L2_pressed")
                #     print(f"[FILE] 写入 '{self.flag_path.name}': L2_pressed")
                # else:
                #     print(f"[FILE] 内容已是 'L2_pressed'，跳过写入")

                if self.detect_start_time is None:
                    self.detect_start_time = now
                elif now - self.detect_start_time >= self.seq_hold_time:
                    print("[VISION] Distance stable. Start to play sequence B.")
                    print("[INPUT] L2 pressed, write in state.")
                    self.ready2placebill = True
                    
                    # read the current file.
                    # === write txt version ===
                    last = None
                    # read only the last line.
                    with self.flag_path.open('r') as f:
                        lines = f.read().splitlines()
                        if lines:
                            last = lines[-1].strip()

                    # if the last line is not pressed. than switch to another line.如果最后一行不是已按下状态，就追加一行
                    if last != "L2_pressed":
                        with self.flag_path.open('a') as f:
                            f.write("L2_pressed\n")
                        print(f"[FILE] Log update: L2_pressed")
                    else:
                        print(f"[FILE] State is already L2_pressed.")

                    # start to play the sequence_b 
                    self.play_sequence_b()
                    # self.ready2placebill = False 
                    # 
                    self.detection_active  = False
                    self.detect_start_time = None
            else:
                # if break the condition 
                if self.detect_start_time is not None:
                    print("[VISION] if break the condition. Restart the detection loop.")
                self.detect_start_time = None

            # TODO Currently, only the selet can interrupte the detection. You can also add new condition if needed.
            if r[KeyMap.select] == 1:   # safe mode 
                self.stop()
                raise SystemExit
            
            return
            
        if r[KeyMap.L2] == 1:     
            # === for directly button control, if using vision comment following lines ===
            # === write txt version ===
            last = None
            with self.flag_path.open('r') as f:
                lines = f.read().splitlines()
                if lines:
                    last = lines[-1].strip()

            if last != "L2_pressed":
                with self.flag_path.open('a') as f:
                    f.write("L2_pressed\n")
                print(f"[FILE] Log update: L2_pressed")
            else:
                print(f"[FILE] State is already L2_pressed.")
                
            self.ready2placebill = True 
            self.play_sequence_b()
            self.ready2placebill = False 
            # ======
            
            # === for vision control, if using vision uncomment following lines ===
            # print("[INPUT] L2 按下,进入视觉等待模式(0.6-0.65m & ±10 deg)")
            # self.detection_active  = True
            # self.detect_start_time = None

            # self.mode = Mode.WAIT_SEQ_B
            # =======

        
            # === Lift arm testing ====
            # self.example_lift_hands(dz=0.05, steps=40)
            # ======
        
        # R1- hand grasp motion 
        if r[KeyMap.R1] == 1:
            self.dex3.switch_gesture(HandGesture.GRIP)
        # L1- hand release motion
        if r[KeyMap.R2] == 1:
            self.dex3.switch_gesture(HandGesture.RELEASE)

        # # play one motion sequence    
        # if r[KeyMap.start] == 1:  # 播放最近一次录制
        #     fn = sorted(self.record_dir.glob("traj_*.npz"))[-1]
        #     self.play_trajectory(fn)

        # up - prepare the selected traj. 
        if r[KeyMap.up] == 1:
            try:
                # self.dex3.switch_gesture(HandGesture.DEFAULT)
                # prepare 选中的动作
                # motion files
                fp = self.motion_files[self._sel_motion_idx]
                self.prepare_replay(fp, speed=1.0, mode="workspace")
            except IndexError:
                print("[WARN] no traj file to replay")

        # down - prepare the selected traj 
        if r[KeyMap.down] == 1:
            self.do_replay()

        # A - start the recording.
        if r[KeyMap.A] == 1 and not self._recording:  
            self.start_record()

        # B - Stop the recording.
        if r[KeyMap.B] == 1 and self._recording:     
            self.stop_record()
        
        # select - Safe terminate mode
        if r[KeyMap.select] == 1:   # safe mode 
            self.stop()
            raise SystemExit
        
        # X - move to default position.
        if r[KeyMap.X] == 1:
            self.dex3.switch_gesture(HandGesture.DEFAULT)
            self.move_to_default(3.0)
        # right - hand back to default.
        if r[KeyMap.right] == 1:
            self.dex3.switch_gesture(HandGesture.DEFAULT)

        # Y damping mode. this one is safer.
        if r[KeyMap.Y] ==1:      
            self.dex3.damping()
            self.zero_torque_mode()
            print(f"[HOLD] Move to zero torque.")

        # update the prev button.
        self.prev_buttons[:] = r


    def remote_poll_audio(self):
        '''This one is for the intergation with the audio.'''
        print("Ensure no obstacle. Press ENTER...")
        # ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)
        try:
            while True:
                self.remote_poll()   
                # time.sleep(0.02)
        except KeyboardInterrupt:
            self.stop()
            print("User exit, stopping thread…")


def main():
    print("Ensure no obstacle. Press ENTER...")
    input()
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)

    ctrl = G1HighlevelArmController()
    ctrl.start()                
    try:
        while True:
            # ------- Terminal command -----------
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip()

                if cmd:                           
                    try:                           # Support enter the number.
                        idx = int(cmd)
                        ctrl._sel_motion_idx = idx % len(ctrl.motion_names)
                    except ValueError:             # Support enter the file names.
                        if cmd in ctrl.motion_names:
                            ctrl._sel_motion_idx = ctrl.motion_names.index(cmd)
                        else:
                            print("[CMD] unknown motion:", cmd); continue
                    print(f"[CMD] selected #{ctrl._sel_motion_idx} {ctrl.motion_names[ctrl._sel_motion_idx]}")
            ctrl.remote_poll() 
            # time.sleep(0.02)
    except KeyboardInterrupt:
        ctrl.stop()
        print("User exit, stopping thread…")

if __name__ == "__main__":
    main()


    
