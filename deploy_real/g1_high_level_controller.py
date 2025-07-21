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
    HOLD      = auto()          # 保持当前位置
    IK_STREAM = auto()          # 在线 IK “一边算一边发”
    PLAY      = auto()          # 播放离线轨迹
    WAIT_SEQ_B = auto()
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
        self.prev_buttons = np.zeros_like(self.remote.button, dtype=int)
        self.ik          = G1_29_ArmIK(Unit_Test=False, Visualization=False)

        # DDS
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_);  self.pub.Init()
        self.sub = ChannelSubscriber("rt/lowstate", LowState_); self.sub.Init(self._cb, 10)
        print("[DDS] Arm Publisher & Subscriber ready.")

        # 手部控制器
        self.dex3 = Dex3GestureController(fps=30.0)
        self.dex3.switch_gesture(HandGesture.DEFAULT)  # 启动时默认握拳
        
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

        # 播放相关缓存
        self._replay_traj     = None   # dict: {t, q, Mf_L, Mf_R}
        self._replay_idx      = 0
        self._replay_speed    = 1.0
        self._replay_mode     = "joint"   # "joint" or "workspace"
        self._replay_ready    = False

        # motion bank
        self._build_motion_bank()      # ← 新增
        self._sel_motion_idx = 0       # 当前选中的动作下标
        
        # —— 新增：视觉检测器
        self.vision = VisionQRDetector(model_size='s')
        self.detection_active     = False
        self.detect_start_time    = None
        # 等待 sequence_b 用的状态
        self.seq_hold_time = 3  # 连续满足条件的秒数阈值

        # —— 新增：L2 触发状态文件 —— 
        self.flag_path = pathlib.Path("l2_trigger_state.txt")
        self.flag_path.write_text("None\n")
        # 如果文件不存在，创建并写入初始状态 L2_pressed=false
        if not self.flag_path.exists():
            self.flag_path.write_text("None\n")
            print(f"[INIT] 创建状态文件 {self.flag_path.name}，内容为 None")

        self.ready2placebill = False 
    
    def _build_motion_bank(self):
        """扫描目录，把所有 .npz 按文件名自然排序后存进列表"""
        files = sorted(pathlib.Path(self.record_dir).glob("*.npz"))
        self.motion_names = [f.stem for f in files]          
        self.motion_files = [str(f) for f in files]          
        print("[MOTION BANK] loaded:", ", ".join(self.motion_names))

    def _cb(self, msg: LowState_):
        self.low_state = msg
        if not self.first_state:
            self.first_state = True
        self.remote.set(msg.wireless_remote)

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

        self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
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
        
        # poseL_def, poseR_def = self.FK(cfg.default_angles)  # 近似

        self.kps = cfg.kps_play.copy() * 0.5 # when moving to default, use a more smaller kp
        self.kds = cfg.kds_play.copy() 
        
        self.move_to(poseL_def, poseR_def,
                    ws_steps=int(duration/2/cfg.control_dt),
                    jnt_steps=int(duration/2/cfg.control_dt))
        print(f"[HOLD] Move to default.")
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
        quatL = Rotation.from_matrix(TL.rotation).as_quat()   # (x,y,z,w) # pin.Quaternion(TL.rotation).as_quat(canonical=True) 
        quatR = Rotation.from_matrix(TR.rotation).as_quat()   # pin.Quaternion(TR.rotation).as_quat(canonical=True) 
        return np.hstack([q,
                        TL.translation, quatL,
                        TR.translation, quatR])

    def _frame_dim(self): 
        return len(cfg.action_joints) + 7 + 7  #  四元数
    
    # --------------- PLAY -----------------
    def _load_traj(self, fp: str):
        """
        支持两种文件结构：
        ① 旧版 npz: traj=(T, 12+3+4+3+4)
        ② 新版 npz: 直接包含 t, q, Mf_L, Mf_R
        """
        data = np.load(fp)
        if "traj" in data:          # 旧格式
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
        else:                       # 新格式（WorkspaceRecorder 生成）
            t, q, Mf_L, Mf_R = data["t"], data["q"], data["Mf_L"], data["Mf_R"]
        return dict(t=t, q=q, Mf_L=Mf_L, Mf_R=Mf_R)
    
    def prepare_replay(self, file_path: str, speed=1.0, mode="workspace"):
        """加载轨迹 & 平滑过渡到首帧"""
        self._replay_traj  = self._load_traj(file_path)
        self._replay_idx   = 0
        self._replay_speed = speed
        self._replay_mode  = mode
        self._replay_ready = False
        self._replay_t_arr   = self._replay_traj["t"] / self._replay_speed
        self._replay_start_t = time.time()

        # 当前 EE 位姿 → 轨迹首帧
        cur_q = self.current_q()
        cur_L, cur_R   = self.FK(cur_q)
        tgt_L_h, tgt_R_h = self._replay_traj["Mf_L"][0], self._replay_traj["Mf_R"][0]
        steps = int(cfg.replay_transition_duration / cfg.control_dt)
        for T_L, T_R in zip(self._interpolate_pose(cur_L, tgt_L_h, steps),
                            self._interpolate_pose(cur_R, tgt_R_h, steps)): 
            self.target_q = self.IK(T_L, T_R)
            time.sleep(cfg.control_dt)
        self._replay_ready = True    # 标记准备完成
        print(f"[REPLAY] Transition complete, ready to play {file_path}.")
        self.mode = Mode.HOLD 

    def do_replay(self):
        if not self._replay_ready:
            print("[REPLAY] 请先调用 prepare_replay()")
            return
        
        self.mode = Mode.PLAY
        self._replay_idx = 0                # 重播从头开始
        self._replay_start_t = time.time()   # 播放计时从现在算
        print(f"[PLAY] start, mode={self._replay_mode}, len={len(self._replay_traj['t'])}")
        
    def example_lift_hands(self, dz=0.05, steps=50):
        """示范：同时抬双手 dz 米，实时 IK 发关节"""
        self.mode = Mode.IK_STREAM
        q0 = self.current_q()
        poseL0, poseR0 = self.FK(q0)

        self.kps = cfg.kps_play.copy()
        self.kds = cfg.kds_play.copy()

        for a in np.linspace(0, 1, steps):
            poseL = pin.SE3(poseL0.rotation, poseL0.translation + np.array([0.05, 0, a*dz]))
            poseR = pin.SE3(poseR0.rotation, poseR0.translation + np.array([0.05, 0, a*dz]))
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
            self.thread.Join()          # 等循环线程真正结束
                  # 或 zero_torque_mode

    def _control_loop(self):
        if self.low_state is None:
            return 

        # # —— 新增：WAIT_SEQ_B 模式下做视觉判断——
        # if self.mode == Mode.WAIT_SEQ_B:
        #     z, angle = self.vision.get_pose()
        #     print(f"[DEBUG] {z:.3f}, {angle:.2f}")
        #     # 条件范围
        #     ok = (z is not None) and (0.61 <= z <= 0.65) and (-5 <= angle <= 5)
            
        #     now = time.time()
        #     if ok:
        #         # 第一次满足时记录起点
        #         if self.wait_seq_start is None:
        #             self.wait_seq_start = now
                    
        #         # 若满足时长，执行 sequence_b 并退出
        #         elif now - self.wait_seq_start >= self.seq_hold_time:
        #             print("[VISION] 条件持续满足，开始 sequence B")
        #             self.vision.stop()
        #             time.sleep(2)
        #             self.play_sequence_b() # sequence_b 不能放在control_loop中执行
        #             self.mode = Mode.HOLD
        #             self.wait_seq_start = None
        #     else:
        #         # 一旦跳出区间，就重置
        #         if self.wait_seq_start is not None:
        #             print("[VISION] 条件中断，重置计时")
        #         self.wait_seq_start = None
        #     return  # WAIT_SEQ_B 时仅判断视觉，不下发其它指令
        

        # 1) 根据模式下发命令
        if self.mode == Mode.HOLD:
            self._send_joint(self.target_q, kps=self.kps, kds=self.kds)  # target_q kps kds由外部函数实时刷写
        elif self.mode == Mode.IK_STREAM:
            self._send_joint(self.target_q, kps=self.kps, kds=self.kds)  # target_q kps kds由外部函数实时刷写

        elif self.mode == Mode.PLAY and self._replay_traj is not None:
                self.kps = cfg.kps_play.copy()
                self.kds = cfg.kds_play.copy()

                if self._replay_idx >=  len(self._replay_t_arr):
                    # 播放结束
                    self.mode = Mode.HOLD

                    # DEBUG
                    # self._replay_traj = None
                    self._replay_ready = False   # 需要重新 prepare

                    print("[PLAY] Finished.")
                    self.target_q = self.current_q()
                    Mf_L, Mf_R = self.FK(self.target_q)
                    
                    print(f"[PLAY] finished. Left: {Mf_L.translation}, Right: {Mf_R.translation}")
                    return
                
                elapsed = time.time() - self._replay_start_t
                while (self._replay_idx < len(self._replay_t_arr)
                    and self._replay_t_arr[self._replay_idx] <= elapsed):
                    self._replay_idx += 1

                # 越界检查（刚好走完时 while 可能多 ++）
                if self._replay_idx >= len(self._replay_t_arr):
                    return                      # 下一轮会进入结束分支
                
                frame_i = self._replay_idx
                if self._replay_mode == "joint":
                    q_cmd = self._replay_traj["q"][frame_i]
                else:   # workspace
                    MfL_h = self._replay_traj["Mf_L"][frame_i]
                    MfR_h = self._replay_traj["Mf_R"][frame_i]
                    q_cmd = self.IK(MfL_h, MfR_h)
                self._send_joint(q_cmd, cfg.kps_play, cfg.kds_play)
                # self._replay_idx += 1

        # 轨迹录制缓存
        if self._recording:
            self._record_buffer.append(self._pack_frame())

    def play_sequence_a(self):
        if not self._replay_ready:
            try:
                self.prepare_replay("records/traj_17_1.npz", speed=1.0, mode="workspace")
            except Exception as e:
                print("[A-SEQUENCE] traj_17_1 load failed:", e)
                return
        self.do_replay()
        while self.mode == Mode.PLAY:
            time.sleep(0.01)

        # 手部切换
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
        """在主线程里循环调用，非阻塞读取遥控器事件"""
        r = self.remote.button
        # press = (self.prev_buttons == 0) & (r == 1)
        # pressed_L1 = press[KeyMap.L1]
        # pressed_L2 = press[KeyMap.L2]
        # pressed_R1 = press[KeyMap.R1]
        # pressed_R2 = press[KeyMap.R2]
        # pressed_up = press[KeyMap.up]
        # pressed_down = press[KeyMap.down]
        # pressed_A = press[KeyMap.A]
        # pressed_B = press[KeyMap.B]
        # pressed_X = press[KeyMap.X]
        # pressed_Y = press[KeyMap.Y]
        # pressed_select = press[KeyMap.select]

        if r[KeyMap.L1] == 1:     
            print("[SEQUENCE A] started.")
            # threading.Thread(target=self.play_sequence_a, daemon=True).start()
            self.play_sequence_a()
            # === Printing used in testing ===
            # poseL, poseR = self.FK(self.current_q())
            # print("[L1] EE pos L:", poseL.translation, "R:", poseR.translation)
            # print("[L1] EE ori L:", poseL.rotation, "R:", poseR.rotation)
            # print(f"[STATE] L: {np.round(self.dex3.left_state,3)} | R: {np.round(self.dex3.right_state,3)}", end="\r")
            # time.sleep(0.1)

        # —— 1. 如果处于“检测模式”，先做视觉判断 —— 
        if self.detection_active:
            z, angle = self.vision.get_pose()
            now = time.time()
            print(f"[DEBUG] {z}, {angle}")
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
                    print("[VISION] 条件持续满足，开始执行 sequence B")
                    print("[INPUT] L2 按下，尝试写入状态文件")
                    self.ready2placebill = True
                    
                    # 读取已有内容（如果文件存在）
                    # === write txt version ===
                    # last = None
                    # # 只读最后一行
                    # with self.flag_path.open('r') as f:
                    #     lines = f.read().splitlines()
                    #     if lines:
                    #         last = lines[-1].strip()

                    # # 如果最后一行不是已按下状态，就追加一行
                    # if last != "L2_pressed":
                    #     with self.flag_path.open('a') as f:
                    #         f.write("L2_pressed\n")
                    #     print(f"[FILE] 追加日志: L2_pressed")
                    # else:
                    #     print(f"[FILE] 日志最后一行已是 'L2_pressed'，跳过追加")

                    self.play_sequence_b()
                    self.ready2placebill = False 
                    # 恢复正常
                    self.detection_active  = False
                    self.detect_start_time = None
            else:
                # 一旦中断，重置计时
                if self.detect_start_time is not None:
                    print("[VISION] 条件中断，重置计时")
                self.detect_start_time = None

            if r[KeyMap.select] == 1:   # safe mode 
                self.stop()
                raise SystemExit
            # 检测模式下不处理其它按键

            return
        if r[KeyMap.L2] == 1:     
            # for button control 
                # existing = None
            # if self.flag_path.exists():
            #     existing = self.flag_path.read_text().strip()
            # # 只有当内容不是 "L2_pressed" 时才写入
            # if existing != "L2_pressed":
            #     self.flag_path.write_text("L2_pressed")
            #     print(f"[FILE] 写入 '{self.flag_path.name}': L2_pressed")
            # else:
            #     print(f"[FILE] 内容已是 'L2_pressed'，跳过写入")

            # print("[INPUT] L2 按下，尝试写入状态文件")
            # # 读取已有内容（如果文件存在）
            # last = None
            # # 只读最后一行
            # with self.flag_path.open('r') as f:
            #     lines = f.read().splitlines()
            #     if lines:
            #         last = lines[-1].strip()

            # # 如果最后一行不是已按下状态，就追加一行
            # if last != "L2_pressed":
            #     with self.flag_path.open('a') as f:
            #         f.write("L2_pressed\n")
            #     print(f"[FILE] 追加日志: L2_pressed")
            # else:
            #     print(f"[FILE] 日志最后一行已是 'L2_pressed'，跳过追加")

            # self.play_sequence_b()

            # for vision control.
            print("[INPUT] L2 按下,进入视觉等待模式(0.6-0.65m & ±10 deg)")
            self.detection_active  = True
            self.detect_start_time = None

            # self.mode = Mode.WAIT_SEQ_B

            # === Lift arm testing ====
            # self.example_lift_hands(dz=0.05, steps=40)
        
        if r[KeyMap.R1] == 1:
            self.dex3.switch_gesture(HandGesture.GRIP)
        if r[KeyMap.R2] == 1:
            self.dex3.switch_gesture(HandGesture.RELEASE)

        # # play one motion sequence    
        # if r[KeyMap.start] == 1:  # 播放最近一次录制
        #     fn = sorted(self.record_dir.glob("traj_*.npz"))[-1]
        #     self.play_trajectory(fn)

        if r[KeyMap.up] == 1:
            try:
                # self.dex3.switch_gesture(HandGesture.DEFAULT)
                # prepare 选中的动作
                # motion files
                fp = self.motion_files[self._sel_motion_idx]
                self.prepare_replay(fp, speed=1.0, mode="workspace")
            except IndexError:
                print("[WARN] no traj file to replay")

        if r[KeyMap.down] == 1:
            self.do_replay()

        if r[KeyMap.A] == 1 and not self._recording:  # A键开始录
            self.start_record()

        if r[KeyMap.B] == 1 and self._recording:      # B键结束录
            self.stop_record()
        
        if r[KeyMap.select] == 1:   # safe mode 
            self.stop()
            raise SystemExit
        
        if r[KeyMap.X] == 1:
            # TODO move to default default state ensure the workspace and joint positions.
            self.dex3.switch_gesture(HandGesture.DEFAULT)
            self.move_to_default(3.0)
        
        if r[KeyMap.right] == 1:
            self.dex3.switch_gesture(HandGesture.DEFAULT)

        if r[KeyMap.Y] ==1:      
            self.dex3.damping()
            self.zero_torque_mode()
            print(f"[HOLD] Move to zero torque.")

        # update the prev button.
        self.prev_buttons[:] = r


    def remote_poll_audio(self):
        print("Ensure no obstacle. Press ENTER...")
        input()
        # ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)
        try:
            while True:
                self.remote_poll()   # 主线程里刷遥控器
                # time.sleep(0.02)
        except KeyboardInterrupt:
            self.stop()
            print("User exit, stopping thread…")


def main():
    print("Ensure no obstacle. Press ENTER...")
    input()
    ChannelFactoryInitialize(0, sys.argv[1] if len(sys.argv) > 1 else None)

    ctrl = G1HighlevelArmController()
    ctrl.start()                 # 启动线程
    try:
        while True:
            # ------- Terminal command -----------
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip()
                if cmd:                            # 不为空
                    try:                           # ① 数字编号
                        idx = int(cmd)
                        ctrl._sel_motion_idx = idx % len(ctrl.motion_names)
                    except ValueError:             # ② 名字
                        if cmd in ctrl.motion_names:
                            ctrl._sel_motion_idx = ctrl.motion_names.index(cmd)
                        else:
                            print("[CMD] unknown motion:", cmd); continue
                    print(f"[CMD] selected #{ctrl._sel_motion_idx} {ctrl.motion_names[ctrl._sel_motion_idx]}")
            ctrl.remote_poll()   # 主线程里刷遥控器
            # time.sleep(0.02)
    except KeyboardInterrupt:
        ctrl.stop()
        print("User exit, stopping thread…")

if __name__ == "__main__":
    main()


    
