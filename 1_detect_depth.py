# 2024-08-19
# author: zhuangaoooo
# 1_detect_depth.py
# This program uses YOLOv8 to detect objects(e.g. person) and robustly get the depth of objects with the RANSAC algorithm.

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor

# initialize the pipeline of RealSense
pipeline = rs.pipeline()
config = rs.config()

# enable the color and depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# start pipeline
profile = pipeline.start(config)

# get the depth_scale of depth sensor
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# align color and depth
align = rs.align(rs.stream.color)

model = YOLO('yolov8n.pt')  # make sure the direction is right

def estimate_depth_with_ransac(depth_data, num_samples=50):
    # randomly select samples
    sampled_depths = np.random.choice(depth_data.flatten(), num_samples, replace=False)
    # estimate depth using RANSAC
    ransac = RANSACRegressor()
    X = np.arange(len(sampled_depths)).reshape(-1, 1)
    ransac.fit(X, sampled_depths)
    inlier_mask = ransac.inlier_mask_
    estimated_depth = np.mean(sampled_depths[inlier_mask])
    return estimated_depth

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        results = model(color_image)

        for result in results:
            for bbox in result.boxes:
                class_id = result.names[int(bbox.cls[0])]
                confidence = bbox.conf[0]
                if confidence > 0.7 and class_id == 'person':
                    x1, y1, x2, y2 = bbox.xyxy[0].int().tolist()
                    bbox_depth_data = depth_image[((y1+y2)//2-np.abs(y1-y2)//4):((y1+y2)//2+np.abs(y1-y2)//4), ((x1+x2)//2-np.abs(x1-x2)//4):((x1+x2)//2+np.abs(x1-x2)//4)] # choose central part of image to sample depth
                    estimated_depth = estimate_depth_with_ransac(bbox_depth_data) * depth_scale
                    
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, f'Person: {confidence:.2f}, Depth: {estimated_depth:.2f}m', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('RealSense with YOLOv8 Detection', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
