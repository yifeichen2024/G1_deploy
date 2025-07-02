import os, pickle
import logging

import torch

from .torch_utils import quat_diff, quat_to_exp_map, slerp
from tqdm import tqdm
logger = logging.getLogger(__name__)

def smooth(x, box_pts, device):
    box = torch.ones(box_pts, device=device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding='same'
    )
    return smoothed.squeeze(0).T


class MotionLib:
    def __init__(self, motion_file, device):
        self._device = device
        self._load_motions(motion_file)
        
    def _load_motions(self, motion_file):
        self._motion_names = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_files = []
        
        self._motion_root_pos = []
        self._motion_root_rot = []
        self._motion_root_vel = []
        self._motion_root_ang_vel = []
        self._motion_dof_pos = []
        self._motion_dof_vel = []
        
        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        
        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            curr_file = motion_files[i]
            self._motion_names.append(os.path.basename(curr_file))
            try:
                with open(curr_file, "rb") as f:
                    motion_data = pickle.load(f)
                    
                    fps = motion_data["fps"]
                    curr_weight = motion_weights[i]
                    dt = 1.0 / fps
                    
                    root_pos = torch.tensor(motion_data["root_pos"], dtype=torch.float, device=self._device)
                    root_rot = torch.tensor(motion_data["root_rot"], dtype=torch.float, device=self._device)
                    dof_pos = torch.tensor(motion_data["dof_pos"], dtype=torch.float, device=self._device)
                    
                    num_frames = root_pos.shape[0]
                    curr_len = dt * (num_frames - 1)
                    
                    root_vel = torch.zeros_like(root_pos) # (num_frames, 3)
                    root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
                    root_vel[-1, :] = root_vel[-2, :]
                    root_vel = smooth(root_vel, 19, device=self._device)
                    
                    root_ang_vel = torch.zeros_like(root_pos) # (num_frames, 3)
                    root_drot = quat_diff(root_rot[:-1], root_rot[1:])
                    root_ang_vel[:-1, :] = fps * quat_to_exp_map(root_drot)
                    root_ang_vel[-1, :] = root_ang_vel[-2, :]
                    root_ang_vel = smooth(root_ang_vel, 19, device=self._device)
                    
                    dof_vel = torch.zeros_like(dof_pos) # (num_frames, num_dof)
                    dof_vel[:-1, :] = fps * (dof_pos[1:, :] - dof_pos[:-1, :])
                    dof_vel[-1, :] = dof_vel[-2, :]
                    dof_vel = smooth(dof_vel, 19, device=self._device)
                    
                    self._motion_weights.append(curr_weight)
                    self._motion_fps.append(fps)
                    self._motion_dt.append(dt)
                    self._motion_num_frames.append(num_frames)
                    self._motion_lengths.append(curr_len)
                    self._motion_files.append(curr_file)
                    
                    self._motion_root_pos.append(root_pos)
                    self._motion_root_rot.append(root_rot)
                    self._motion_root_vel.append(root_vel)
                    self._motion_root_ang_vel.append(root_ang_vel)
                    self._motion_dof_pos.append(dof_pos)
                    self._motion_dof_vel.append(dof_vel)
            except Exception as e:
                logger.error(f"Error loading motion file {curr_file}: {e}")
                continue
        
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights /= torch.sum(self._motion_weights)
        
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float, device=self._device)
        
        self._motion_root_pos = torch.cat(self._motion_root_pos, dim=0)
        self._motion_root_rot = torch.cat(self._motion_root_rot, dim=0)
        self._motion_root_vel = torch.cat(self._motion_root_vel, dim=0)
        self._motion_root_ang_vel = torch.cat(self._motion_root_ang_vel, dim=0)
        self._motion_dof_pos = torch.cat(self._motion_dof_pos, dim=0)
        self._motion_dof_vel = torch.cat(self._motion_dof_vel, dim=0)
        
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        
        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
        
    def num_motions(self):
        return self._motion_weights.shape[0]
    
    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()
                
    def _fetch_motion_files(self, motion_file: str):
        motion_files = [motion_file]
        motion_weights = [1.0]
        
        return motion_files, motion_weights
    
    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]
        
        phase = times / self._motion_lengths[motion_ids]
        phase = torch.clip(phase, 0.0, 1.0)
        
        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0.float()
        
        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx0 += frame_start_idx
        frame_idx1 += frame_start_idx
        
        return frame_idx0, frame_idx1, blend
        
    def calc_motion_frame(self, motion_ids, motion_times):
        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times -= motion_loop_num * self._motion_lengths[motion_ids]
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)
        
        root_pos0 = self._motion_root_pos[frame_idx0]
        root_pos1 = self._motion_root_pos[frame_idx1]
        
        root_rot0 = self._motion_root_rot[frame_idx0]
        root_rot1 = self._motion_root_rot[frame_idx1]
        
        root_vel = self._motion_root_vel[frame_idx0]
        root_ang_vel = self._motion_root_ang_vel[frame_idx0]
        
        dof_pos0 = self._motion_dof_pos[frame_idx0]
        dof_pos1 = self._motion_dof_pos[frame_idx1]
        
        dof_vel = self._motion_dof_vel[frame_idx0]
        
        blend_unsqueeze = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
        root_rot = slerp(root_rot0, root_rot1, blend)
        
        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1
        
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel
        