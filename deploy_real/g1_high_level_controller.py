import time, sys, json, pathlib
import numpy as np
import yaml, pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp
from enum import Enum, auto
from collections import deque
import os

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.utils.crc import CRC

from common.remote_controller import RemoteController, KeyMap
from g1_arm_IK import G1_29_ArmIK

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
    cfg.kds_record = cfg.kds_play * 0.5
    return cfg

cfg = load_cfg()

class Mode(Enum):
    IDLE      = auto()
    HOLD      = auto()          # 保持当前位置
    IK_STREAM = auto()          # 在线 IK “一边算一边发”
    PLAY      = auto()          # 播放离线轨迹

# -----------------------------------------------------------------------------
# G1 Arm controller
# -----------------------------------------------------------------------------
class G1HighlevelArmController:
    def __init__(self, record_dir="records", history_len=3):
        # --- 通信 & 运动学 ---
        self.low_state   : LowState_ | None = None
        self.first_state = False
        self.low_cmd     = unitree_hg_msg_dds__LowCmd_()
        self.crc         = CRC()
        self.remote      = RemoteController()

        self.ik          = G1_29_ArmIK(Unit_Test=False, Visualization=False)

        # DDS
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_);  self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self._cb, 10)
        print("[DDS] Publisher & Subscriber ready.")

        # --- 控制变量 ---
        self.mode         = Mode.IDLE
        self.target_q     = cfg.default_angles.copy()
        self.kps = np.zeros_like(cfg.kps_play)
        self.kds = np.ones_like(cfg.kds_play)

        self.history      = deque(maxlen=history_len)   # 关节历史
        self.thread       = None

        # --- 轨迹录制 / 播放 ---
        self._recording     = False
        self._record_buffer = []
        self.record_dir     = pathlib.Path(record_dir); self.record_dir.mkdir(exist_ok=True)
        self._play_traj     = None      # np.ndarray(step, dof)
        self._play_idx      = 0

    def _cb(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True
        self.remote.set(msg.wireless_remote)

    # def set_target(self, q, kps=cfg.kps_play, kds=cfg.kds):
    #     with self._cmd_lock:
    #         self._cmd["q"]   = np.asarray(q, dtype=float)
    #         self._cmd["kps"] = np.asarray(kps if kps is not None else cfg.kps_play)
    #         self._cmd["kds"] = np.asarray(kds if kds is not None else cfg.kds_play)

    def _send_joint(self, q, kps=None, kds=None):
        kps = np.asarray(kps if kps is not None else self.kps)
        kds = np.asarray(kds if kds is not None else self.kds)
        for i, m in enumerate(cfg.action_joints):
            self.low_cmd.motor_cmd[m].q  = float(q[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(kps[i])
            self.low_cmd.motor_cmd[m].kd = float(kds[i])
            self.low_cmd.motor_cmd[m].tau= 0.0

        # 固定关节
        for i, m in enumerate(cfg.fixed_joints):
            self.low_cmd.motor_cmd[m].q  = float(cfg.fixed_target[i])
            self.low_cmd.motor_cmd[m].dq = 0.0
            self.low_cmd.motor_cmd[m].kp = float(cfg.fixed_kps[i])
            self.low_cmd.motor_cmd[m].kd = float(cfg.fixed_kds[i])
            self.low_cmd.motor_cmd[m].tau= 0.0

        self.low_cmd.motor_cmd[cfg.kNotUsedJoint].q = 1
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    
    # 访问器 --------------------------------------------------
    def current_q(self):
        return np.array([self.low_state.motor_state[m].q for m in cfg.action_joints])

    def FK(self, q):
        pin.forwardKinematics(self.ik.reduced_robot.model, self.ik.reduced_robot.data, q)
        pin.updateFramePlacements(self.ik.reduced_robot.model, self.ik.reduced_robot.data)
        d = self.ik.reduced_robot.data
        return d.oMf[self.ik.L_hand_id], d.oMf[self.ik.R_hand_id]

    def IK(self, poseL: pin.SE3, poseR: pin.SE3):
        q_now = self.current_q()
        q_cmd, _ = self.ik.solve_ik(poseL.homogeneous, poseR.homogeneous, current_lr_arm_motor_q=q_now)
        return q_cmd

    # TODO Test
    def move_to(self, poseL: pin.SE3, poseR: pin.SE3,
            ws_steps=50, jnt_steps=20):
        """1. 工作空间插值 2. 关节微调"""
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

    # TODO Test
    def move_to_default(self, duration=3.0):
        poseL_def, poseR_def = self.FK(cfg.default_angles)  # 近似
        self.move_to(poseL_def, poseR_def,
                    ws_steps=int(duration/2/cfg.control_dt),
                    jnt_steps=int(duration/2/cfg.control_dt))

    # TODO test
    def _interpolate_pose(self, T0: pin.SE3, T1: pin.SE3, n: int):
        """返回包含起末端的 n 个 pin.SE3"""
        # 旋转：Slerp 
        R0, R1 = Rotation.from_matrix(T0.rotation), Rotation.from_matrix(T1.rotation)
        key_times = [0, 1]

        # TODO slerp usage check. checked.
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
        self.mode = Mode.HOLD
        self.target_q = self.current_q()
        self.kps = np.zeros_like(cfg.kps_play)
        self.kds = np.zeros_like(cfg.kds_play)

        # self._send_joint(self.current_q(), kps=np.zeros_like(cfg.kps_play), kds=np.zeros_like(cfg.kds_play))

    def damping_mode(self, kd=2):
        self.mode = Mode.HOLD
        self.target_q = self.current_q()
        self.kps = np.zeros_like(cfg.kps_play)
        self.kds = np.ones_like(cfg.kds_play)*kd

        # self._send_joint(self.current_q(),
        #                  kps=np.zeros_like(cfg.kps_play),
        #                  kds=np.ones_like(cfg.kds_play)*kd)
    
     # --------------- RECORD -----------------
    def start_record(self):
        self._recording = True
        self.target_q = self.current_q()
        # kps: list[float] = []
        # kds: list[float] = []
        # for idx, joint in enumerate(cfg.action_joints):
        self.kps = (cfg.kps_play*cfg.stiffness_factor).copy()
        self.kds = (cfg.kds_play*cfg.stiffness_factor).copy()

        self._record_buffer.clear()
        print("[REC] start.")

    def stop_record(self, save=True, name=None):
        self._recording = False

        # kps: list[float] = []
        # kds: list[float] = []
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
    
    # TODO Test
    def _pack_frame(self):
        q   = self.current_q()
        TL, TR = self.FK(q)
        quatL = pin.Quaternion(TL.rotation) # Rotation.from_matrix(TL.rotation).as_quat()   # (x,y,z,w)
        quatR = pin.Quaternion(TR.rotation) # Rotation.from_matrix(TR.rotation).as_quat()
        return np.hstack([q,
                        TL.translation, quatL,
                        TR.translation, quatR])

    def _frame_dim(self): 
        return len(cfg.action_joints) + 7 + 7  #  四元数
    # --------------- PLAY -----------------
    # TODO Test 
    def play_trajectory(self, path_or_arr: np.ndarray | str, speed=1.0, mode="joint"):
        """
        traj : ndarray shape (T, dof) 或 文件路径
        speed: 1.0 = 实时 (dt = cfg.control_dt/speed)
        """
        data = np.load(path_or_arr)["traj"] if isinstance(path_or_arr, (str, pathlib.Path)) else path_or_arr
        self._play_traj, self._play_idx, self.mode = data, 0, Mode.PLAY
        self._play_dt   = cfg.control_dt / speed
        self._play_mode = mode  # "joint" or "workspace"
        print(f"[PLAY] {mode} len={len(data)}")
        
    def example_lift_hands(self, dz=0.05, steps=50):
        """示范：同时抬双手 dz 米，实时 IK 发关节"""
        self.mode = Mode.IK_STREAM
        q0 = self.current_q()
        poseL0, poseR0 = self.FK(q0)

        for a in np.linspace(0, 1, steps):
            poseL = pin.SE3(poseL0.rotation, poseL0.translation + np.array([0.01, 0, a*dz]))
            poseR = pin.SE3(poseR0.rotation, poseR0.translation + np.array([0.01, 0, a*dz]))
            self.target_q = self.IK(poseL, poseR)   # 设置为线程循环读取
            time.sleep(cfg.control_dt)
        self.mode = Mode.HOLD

    def start(self):
        while not self.first_state:
            print("[INIT] waiting lowstate…")
            time.sleep(0.1)

        # 控制线程
        self.thread = RecurrentThread(interval=cfg.control_dt,
                                      target=self._control_loop,
                                      name="hl_arm_loop")
        self.thread.Start()
        # create cmd

        print("[INIT] control thread started.")
        # TODO Test
        # Damping mode first
        # zero torque mode to loose everything.
        # go to default state
        self.zero_torque_mode() 
        print("[HL] Ender zero torque mode.")
        time.sleep(0.5)
        self.damping_mode(kd=1.0)
        print("[HL] Enter damping mode.")
        time.sleep(0.5)
        self.move_to_default(duration=3)
        print("[HL] Move to default.")

    def stop(self):
        print("[HL] stopping controller…")
        self.damping_mode(kd=2)
        if self.thread and self.thread.IsRunning():
            self.thread.Stop()
            self.thread.Join()          # 等循环线程真正结束
                  # 或 zero_torque_mode

    def _control_loop(self):
        if self.low_state is None:
            return 
        # TODO I think for the joint control and position control. Here is the logic:
        # when recording, record both the joint and the workspace info.(the format of the workspace position and orientation data should be careful aglined with the data for the IK required.)
        # when playing, Use joint control or the position control. Set a param that can choose between this two mode. for different usage decide by the user.
        # when go back to default. 
        # [Move to DEFAULT and other transisition or move to ]when doing transition between two different target use poisition control first to go that target. then check the error between the target joint value. if error is larger then 0.01, use joint control to go to that target.


        # else: 
        #     for idx, joint in enumerate(cfg.action_joints):
        #         kps.append(cfg.kps_play[idx])
        #         kds.append(cfg.kds_play[idx])

        # 1) 根据模式下发命令
        if self.mode == Mode.HOLD:
            self._send_joint(self.target_q, kps=self.kps, kds=self.kds)  # target_q kps kds由外部函数实时刷写
        elif self.mode == Mode.IK_STREAM:
            self._send_joint(self.target_q, kps=self.kps, kds=self.kds)  # target_q kps kds由外部函数实时刷写

        elif self.mode == Mode.PLAY and self._play_traj is not None:
                self.kps = cfg.kps_play.copy()
                self.kds = cfg.kds_play.copy()

                if self._play_idx < len(self._play_traj):
                    q_dof = len(cfg.action_joints)
                    idx_pL = slice(q_dof, q_dof+3)          # Left translation
                    idx_qL = slice(q_dof+3, q_dof+7)        # Left rotation
                    idx_pR = slice(q_dof+7, q_dof+10)        # Right translation
                    idx_qR = slice(q_dof+10, q_dof+14)       # Right rotation 
                    frame = self._play_traj[self._play_idx]; 
                    self._play_idx += 1

                    if self._play_mode=="joint":
                        self._send_joint(frame[:q_dof], self.kps, self.kds)
                    else:  # workspace
                        poseL = pin.SE3(pin.Quaternion(frame[idx_qL]),
                                        frame[idx_pL])
                        poseR = pin.SE3(pin.Quaternion(frame[idx_qR]),
                                        frame[idx_pR])
                        self._send_joint(self.IK(poseL, poseR), self.kps, self.kds)
    
                    time.sleep(self._play_dt)
                else:
                    self.mode = Mode.HOLD
                    Mf_L, Mf_R = self.FK(self.current_q())
                    self.target_q = self.current_q()
                    print(f"[PLAY] finished. Left: {Mf_L.translation}, Right: {Mf_R.translation}")

        # 轨迹录制缓存
        if self._recording:
            self._record_buffer.append(self._pack_frame())


    def remote_poll(self):
        """在主线程里循环调用，非阻塞读取遥控器事件"""
        r = self.remote.button
        if r[KeyMap.L1] == 1:     # 打印 FK
            poseL, poseR = self.FK(self.current_q())
            print("[L1] EE pos L:", poseL.translation, "R:", poseR.translation)
            time.sleep(0.3)

        if r[KeyMap.L2] == 1:     # 抬手测试
            self.example_lift_hands(dz=0.05, steps=40)

        if r[KeyMap.start] == 1:  # 播放最近一次录制
            fn = sorted(self.record_dir.glob("traj_*.npz"))[-1]
            self.play_trajectory(fn)

        if r[KeyMap.A] == 1 and not self._recording:  # 例：A键开始录
            self.start_record()

        if r[KeyMap.B] == 1 and self._recording:      # B键结束录
            self.stop_record()
        
        if r[KeyMap.select] == 1:   # safe mode 
            self.stop()
            raise SystemExit
        
        if r[KeyMap.X] == 1:
            # TODO move to default default state ensure the workspace and joint positions.
            self.move_to_default(3.0)
        
        if r[KeyMap.Y]==1:      
            self.zero_torque_mode()

def main():
    print("Ensure no obstacle. Press ENTER...")
    input()
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)

    ctrl = G1HighlevelArmController()
    ctrl.start()                 # 启动线程

    try:
        while True:
            ctrl.remote_poll()   # 主线程里刷遥控器
            time.sleep(0.02)
    except KeyboardInterrupt:
        ctrl.stop()
        print("User exit, stopping thread…")

    


    
