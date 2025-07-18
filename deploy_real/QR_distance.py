import pyrealsense2 as rs
import cv2
import numpy as np
import torch
if not isinstance(torch.__version__, str):  # TorchVersion → str
    torch.__version__ = str(torch.__version__)
from qrdet import QRDetector

# 初始化 QR 检测器
detector = QRDetector(model_size='s')

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
pipeline.start(config)

def median_depth_from_bbox(depth_frame, bbox_xyxy):
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    x1 = max(0, min(x1, depth_frame.get_width() - 1))
    x2 = max(0, min(x2, depth_frame.get_width() - 1))
    y1 = max(0, min(y1, depth_frame.get_height() - 1))
    y2 = max(0, min(y2, depth_frame.get_height() - 1))
    zs = [depth_frame.get_distance(x, y)
          for y in range(y1, y2)
          for x in range(x1, x2)]
    zs = [z for z in zs if 0.1 < z < 5.0]
    return float(np.median(zs)) if zs else None

try:
    while True:
        # 采集帧
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_img = np.asanyarray(color_frame.get_data())
        detections = detector.detect(image=color_img, is_bgr=True)

        last_values = []  # 用来存放本帧所有 det 的 (dx, dy, z, θ)

        for det in detections:  
            # 1. 计算二维码 bbox 和中心点
            x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 2. 计算画面中心
            h, w = color_img.shape[:2]
            img_cx, img_cy = w // 2, h // 2

            # 3. 计算像素偏移
            dx_pix = cx - img_cx    # 正值：二维码中心在画面右侧
            dy_pix = cy - img_cy    # 正值：二维码中心在画面下方

            # 4. 保持之前的 Z 测距和角度计算
            z = median_depth_from_bbox(depth_frame, det['bbox_xyxy'])
            dx_str = f" dx={dx_pix:+d}px"
            dy_str = f" dy={dy_pix:+d}px"

            # 你原来的倾斜角度
            dx_box = x2 - x1
            dy_box = y2 - y1
            angle_deg = np.degrees(np.arctan2(dy_box, dx_box))
            theta_str = f" theta={angle_deg:.1f}"
            last_values.append((dx_pix, dy_pix, z, angle_deg))
            # 5. 绘制与标签
            cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['confidence']:.2f} | Z={z:.2f}m" if z else f"{det['confidence']:.2f} | Z=?"
            label += dx_str + dy_str + theta_str

            cv2.putText(color_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)




        cv2.imshow("QR + Depth", color_img)
        key = cv2.waitKey(1)
            # 如果按下回车 (Enter)
        if key == 13 or key == 10:
            for idx, (dx, dy, z, theta) in enumerate(last_values, 1):
                z_str = "?" if z is None else f"{z:.2f}"
                print(f"[Det {idx}] dx={dx:+d}px, dy={dy:+d}px, Z={z_str}m, θ={theta:.1f}°")
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
