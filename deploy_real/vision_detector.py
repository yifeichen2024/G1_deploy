# vision_detector.py
import pyrealsense2 as rs
import numpy as np
from qrdet import QRDetector
import time

class VisionQRDetector:
    def __init__(self, model_size='s'):
        # 初始化二维码检测器
        self.detector = QRDetector(model_size=model_size)
        # 初始化 Realsense 管线
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.align = rs.align(rs.stream.color)
        self.pipeline.start(cfg)
        self.intrinsics = None

    def get_pose(self):
        """抓一帧，返回 (z, angle) 或 (None, None)"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None

        img = np.asanyarray(color_frame.get_data())
        dets = self.detector.detect(image=img, is_bgr=True)
        if not dets:
            return None, None

        # 取第一组检测
        det = dets[0]
        cx, cy = det['cxcy']
        z = depth_frame.get_distance(int(cx), int(cy))
        if not (0.1 < z < 5.0):
            return None, None

        quad = det['quad_xy']
        v = quad[1] - quad[0]
        angle = np.degrees(np.arctan2(v[1], v[0]))  # -180..180

        h, w = img.shape[:2]
        img_cx, img_cy = w // 2, h // 2

        # 3. 计算像素偏移
        dx_pix = cx - img_cx    # 正值：二维码中心在画面右侧
        dy_pix = cy - img_cy    # 正值：二维码中心在画面下方

        # 缓存内参
        if self.intrinsics is None:
            self.intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        return z, angle, dx_pix, dy_pix

    def stop(self):
        """退出时调用，释放管线"""
        self.pipeline.stop()
