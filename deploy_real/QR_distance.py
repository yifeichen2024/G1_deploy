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

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox_xyxy'])
            conf = det['confidence']
            z = median_depth_from_bbox(depth_frame, det['bbox_xyxy'])

            # 画框
            cv2.rectangle(color_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{conf:.2f} | Z={z:.3f}m" if z else f"{conf:.2f} | Z=?"
            cv2.putText(color_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("QR + Depth", color_img)
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
