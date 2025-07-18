import pyrealsense2 as rs
import numpy as np
import cv2

# 创建一个pipeline
pipeline = rs.pipeline()

# 创建一个配置对象
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)     # 彩色图像
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)      # 深度图像
config.enable_stream(rs.stream.accel)                                   # 加速度计
config.enable_stream(rs.stream.gyro)                                    # 陀螺仪

# 启动pipeline
pipeline.start(config)

try:
    while True:
        # 等待一帧帧数据
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if not color_frame or not depth_frame:
            continue

        # 转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 显示彩色图像
        cv2.imshow('Color Image', color_image)

        # 显示深度图像（使用 colormap 映射增强可视化）
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Depth Image', depth_colormap)

        # 读取 IMU 数据
        if accel_frame:
            accel = accel_frame.as_motion_frame().get_motion_data()
            print(f"Accel: x={accel.x:.4f}, y={accel.y:.4f}, z={accel.z:.4f}")
        if gyro_frame:
            gyro = gyro_frame.as_motion_frame().get_motion_data()
            print(f"Gyro: x={gyro.x:.4f}, y={gyro.y:.4f}, z={gyro.z:.4f}")

        # 按下 ESC 键退出
        key = cv2.waitKey(1)
        if key == 27:
            break

finally:
    # 停止pipeline
    pipeline.stop()
    cv2.destroyAllWindows()