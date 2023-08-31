#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File    :   cv2_inference.py
@Time    :   2023/08/31 21:23:55
@Author  :   Chen Daoyuan
@Contact :   chendymaodai@163.com
@Desc    :   cv2 streaming for-loop
'''
import cv2
import time
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Script")

    # Specify the path to the model weights file
    parser.add_argument('--weight_path',
                        default='yolov8m.pt',
                        help="Path to the model weights file.")
    # Specify the target fps for playback
    parser.add_argument('--target_fps',
                        type=int,
                        default=25,
                        help="Target FPS for playback.")
    # Specify the starting UNIX timestamp (in milliseconds) or None
    parser.add_argument(
        '--start_ts',
        default=None,
        type=int,
        help="Starting UNIX timestamp in milliseconds or None.")
    # Specify the video path
    parser.add_argument(
        '--video_path',
        default='/home/cdy/Data_HDD/Data/Project_SPC/test_data/test1_p1.mp4',
        help="Path to the video file.")

    args = parser.parse_args()
    return vars(args)


def main():
    args = parse_args()

    weight_path = args["weight_path"]
    target_fps = args["target_fps"]
    start_ts = args["start_ts"]
    video_path = args["video_path"]

    if start_ts is None:
        start_ts = int(round(time.time() * 1000))

    # Load model
    model = YOLO(weight_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    assert target_fps <= video_fps, "Error: Target FPS must be less than or equal to the video FPS."

    # Calculate frame interval based on target_fps
    frame_interval = video_fps // target_fps

    # Calculate actual_fps based on frame_interval
    actual_fps = video_fps // frame_interval

    # Calculate ts_interval based on video_fps
    ts_interval = int(1000 / video_fps)

    # Initialize current_ts with start_ts
    current_ts = start_ts

    print(f"Original Width: {width}")
    print(f"Original Height: {height}")
    print(f"Video FPS: {video_fps}")
    print(f"Target FPS: {target_fps}")
    print(f"Actual FPS: {actual_fps}")
    print(f"Frame Interval: {video_fps // target_fps}")
    print(f"Total Frames: {total_frames}")

    pause = False  # Variable to check if video is paused
    current_frame_idx = 0  # Counter for the current frame index

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            if current_frame_idx % frame_interval == 0:
                print("-" * 50)
                print(f"Frame: {current_frame_idx + 1}/{total_frames}")
                print(f"Timestamp: {current_ts}ms")

                # Run YOLOv8 inference on the frame
                results = model(frame, retina_masks=True)

                # Visualize the results on the frame
                annotated_frame = results[0].plot()

                # Display the annotated frame
                cv2.imshow("YOLOv8 Inference", annotated_frame)

                key = cv2.waitKey(1) & 0xFF

                # Break the loop if 'q' is pressed
                if key == ord('q'):
                    break
                # Pause/Play loop if 'p' is pressed
                elif key == ord('p'):
                    pause = not pause

                # If the video is paused, loop until 'p' is pressed again
                while pause:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('p'):
                        pause = not pause
                        break

            # Update the current timestamp
            current_ts += ts_interval

            # Increment the frame counter
            current_frame_idx += 1
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
