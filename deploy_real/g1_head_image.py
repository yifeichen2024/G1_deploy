import pyrealsense2 as rs
import cv2
import numpy as np
import torch
if not isinstance(torch.__version__, str):
    torch.__version__ = str(torch.__version__)
from qrdet import QRDetector

# 初始化
detector = QRDetector(model_size='s')

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)
pipeline.start(cfg)

def median_depth_from_bbox(depth_frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    x1 = np.clip(x1, 0, depth_frame.get_width()-1)
    x2 = np.clip(x2, 0, depth_frame.get_width()-1)
    y1 = np.clip(y1, 0, depth_frame.get_height()-1)
    y2 = np.clip(y2, 0, depth_frame.get_height()-1)
    zs = [depth_frame.get_distance(x, y)
          for y in range(y1, y2) for x in range(x1, x2)]
    zs = [z for z in zs if 0.1 < z < 5.0]
    return float(np.median(zs)) if zs else None

try:
    while True:
        frames     = pipeline.wait_for_frames()
        aligned    = align.process(frames)
        color_frame= aligned.get_color_frame()
        depth_frame= aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        dets = detector.detect(image=img, is_bgr=True)

        # 取一次内参用于反投影
        intr = depth_frame.profile.as_video_stream_profile().intrinsics

        for det in dets:
            # —— 1. 四角 quad_xy —— 
            quad = det['quad_xy']           # shape (4,2), np.float32
            quad_int = quad.astype(int)

            # —— 2. 中心 cx, cy —— 
            cx, cy = det['cxcy']            # float, float

            # —— 3. 深度 Z —— 
            z = depth_frame.get_distance(int(cx), int(cy))
            z = z if 0.1 < z < 5.0 else None

            # —— 4. 位置定位 —— 
            pos_str = ""
            if z is not None:
                X, Y, Z = rs.rs2_deproject_pixel_to_point(
                    intr, [cx, cy], z)
                pos_str = f" | X={X:.2f} Y={Y:.2f}"

            # —— 5. 平面朝向 θ —— 
            # 用 quad[0]→quad[1] 边作为“水平”参考
            v = quad[1] - quad[0]           # vec from corner0 to corner1
            angle = np.degrees(np.arctan2(v[1], v[0]))  # -180..180
            theta_str = f" | a={angle:.1f}deg"

            # —— 6. 绘制可视化 —— 
            # 画多边形
            cv2.polylines(img, [quad_int], True, (0,255,0), 2)
            # 标记角点编号（可选）
            for i, (x, y) in enumerate(quad_int):
                cv2.putText(img, str(i), (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # —— 7. 标签文本 —— 
            conf = det['confidence']
            z_str = f"{z:.2f}m" if z is not None else "Z=?"
            label = f"{conf:.2f} | {z_str}" + pos_str + theta_str
            # 在 quad[0] 角点旁显示
            x0, y0 = quad_int[0]
            cv2.putText(img, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            print(f"Z={z:.2f}m | X={X:.2f} Y={Y:.2f} | a={angle:.1f} deg")
        cv2.imshow("QR + Depth + Pose", img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
