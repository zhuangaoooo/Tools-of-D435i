# Tools-of-D435i
This repository includes some useful tools of D435i. Hope these tools can give you a help.

1. 1_detect_depth.py: This program uses YOLOv8 to detect objects(e.g. person) and robustly get the depth of objects with the RANSAC algorithm. You can run `python3 1_detect_depth.py` with your device(e.g. D435i) connecting.

2. 2_rosbag2TUM.py: This program offers an all-in-one method to change your rosbag which may be recorded using realsense SDK into a TUM form. Part of the code is referenced from _https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools_. Usage: `python3 2_rosbag2TUM.py --bag_path /path/to/bagfile.bag --output_dir /path/to/output_directory`.


# D435i的工具库
本仓库包含一些D435i的实用工具。希望这些工具能帮到你。

1. 1_detect_depth.py: 该程序使用YOLOv8来检测目标(e.g. person)，并用RANSAC算法鲁棒地获取目标的深度。你可以连接你的设备(e.g. D435i)并运行`python3 1_detect_depth.py`。

2. 2_rosbag2TUM.py: 该程序提供了一站式的方法来将rosbag转化成TUM格式（rosbag可以通过realsense SDK录制）。部分代码参考了_https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools_。使用方法：`python3 2_rosbag2TUM.py --bag_path /path/to/bagfile.bag --output_dir /path/to/output_directory`
