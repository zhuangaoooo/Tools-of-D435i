# 2024-08-24
# author: zhuangaoooo
# 2_rosbag2TUM.py
# This program offers an all-in-one method to change your rosbag which may be recorded using realsense SDK into a TUM form.
# Usage: python 2_rosbag2TUM.py --bag_path /path/to/bagfile.bag --output_dir /path/to/output_directory
# Part of the code is referenced from https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools

import argparse
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import rosbag

def extract_images_from_bag(bag_path, output_dir):
    """
    Extracts RGB and Depth images from a ROS bag file and saves them along with timestamp files.

    Args:
        bag_path (str): Path to the ROS bag file.
        output_dir (str): Directory to save extracted images and timestamp files.

    Returns:
        rgb_txt_path (str): Path to the generated rgb.txt file.
        depth_txt_path (str): Path to the generated depth.txt file.
    """

    # Initialize directories
    rgb_dir = os.path.join(output_dir, "rgb")
    depth_dir = os.path.join(output_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    rgb_txt_path = os.path.join(output_dir, "rgb.txt")
    depth_txt_path = os.path.join(output_dir, "depth.txt")

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    # Start pipeline
    profile = pipeline.start(config)

    # Get playback device and set real-time to false for faster processing
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # Align depth to color
    align = rs.align(rs.stream.color)

    # Open timestamp files
    with open(rgb_txt_path, 'w') as rgb_file, open(depth_txt_path, 'w') as depth_file:
        frame_number = 0
        try:
            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                # Get timestamps
                timestamp = frames.get_timestamp() / 1000.0  # Convert to seconds
                timestamp_str = "%.6f" % timestamp

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())

                # Save images
                color_image_path = os.path.join(rgb_dir, f"{timestamp_str}.png")
                depth_image_path = os.path.join(depth_dir, f"{timestamp_str}.png")

                cv2.imwrite(color_image_path, color_image)
                cv2.imwrite(depth_image_path, depth_image)

                # Write to txt files
                rgb_file.write(f"{timestamp_str} rgb/{timestamp_str}.png\n")
                depth_file.write(f"{timestamp_str} depth/{timestamp_str}.png\n")

                frame_number += 1
                print(f"Saved frame {frame_number} at timestamp {timestamp_str}s")

        except RuntimeError:
            # No more frames
            print("Finished extracting all frames from the bag file.")
        finally:
            pipeline.stop()

    return rgb_txt_path, depth_txt_path

def read_file_list(filename):
    """
    Reads a file and returns a dictionary with timestamps as keys and file paths as values.

    Args:
        filename (str): Path to the file.

    Returns:
        dict: Dictionary with timestamps and corresponding data.
    """
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            if line.strip() == "" or line.startswith("#"):
                continue
            parts = line.strip().split()
            timestamp = float(parts[0])
            data[timestamp] = parts[1:]
    return data

def associate(rgb_list, depth_list, offset=0.0, max_difference=0.02):
    """
    Associates rgb and depth images based on timestamps.

    Args:
        rgb_list (dict): Dictionary of rgb timestamps and paths.
        depth_list (dict): Dictionary of depth timestamps and paths.
        offset (float): Time offset between rgb and depth timestamps.
        max_difference (float): Maximum allowed difference between timestamps.

    Returns:
        list: List of matched tuples (rgb_timestamp, rgb_data, depth_timestamp, depth_data).
    """
    matches = []
    rgb_keys = sorted(rgb_list.keys())
    depth_keys = sorted(depth_list.keys())

    depth_idx = 0
    for rgb_ts in rgb_keys:
        while depth_idx < len(depth_keys) and depth_keys[depth_idx] + offset < rgb_ts - max_difference:
            depth_idx += 1
        if depth_idx < len(depth_keys):
            depth_ts = depth_keys[depth_idx]
            time_diff = abs(rgb_ts - (depth_ts + offset))
            if time_diff < max_difference:
                matches.append((rgb_ts, rgb_list[rgb_ts], depth_ts, depth_list[depth_ts]))

    return matches

def save_associations(matches, output_path):
    """
    Saves the associated matches to a file.

    Args:
        matches (list): List of matched tuples.
        output_path (str): Path to save the associations.
    """
    with open(output_path, 'w') as file:
        for match in matches:
            rgb_ts, rgb_data, depth_ts, depth_data = match
            line = f"{rgb_ts:.6f} {' '.join(rgb_data)} {depth_ts:.6f} {' '.join(depth_data)}\n"
            file.write(line)
    print(f"Associations saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract and associate RGB and Depth images from ROS bag.")
    parser.add_argument('--bag_path', required=True, help="Path to the ROS bag file.")
    parser.add_argument('--output_dir', required=True, help="Directory to save extracted images and files.")
    parser.add_argument('--offset', type=float, default=0.0, help="Time offset between rgb and depth timestamps.")
    parser.add_argument('--max_difference', type=float, default=0.02, help="Maximum allowed time difference for matching.")
    args = parser.parse_args()

    # Step 1: Extract images and timestamps
    print("Extracting images and timestamps from bag file...")
    rgb_txt_path, depth_txt_path = extract_images_from_bag(args.bag_path, args.output_dir)

    # Step 2: Associate timestamps
    print("Associating timestamps...")
    rgb_list = read_file_list(rgb_txt_path)
    depth_list = read_file_list(depth_txt_path)
    matches = associate(rgb_list, depth_list, args.offset, args.max_difference)

    # Save associations
    associate_txt_path = os.path.join(args.output_dir, "associate.txt")
    save_associations(matches, associate_txt_path)

if __name__ == "__main__":
    main()

