import threading
import os
import cv2
from pathlib import Path
from ultralytics import YOLO


def run_tracker_in_thread(filename, model):
    print("-" * 50)
    file_path = Path(filename)
    file_stem = file_path.stem
    file_parent = file_path.parent
    print(file_stem)
    print(file_parent)
    save_path = file_parent / (file_stem + '_tracked.mp4')
    print(save_path)
    video = cv2.VideoCapture(filename)
    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), 25,
                             (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                              int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in range(frames):
        ret, frame = video.read()
        if ret:
            results = model.track(source=frame, persist=True)
            res_plotted = results[0].plot()
            print(file_stem)
            writer.write(res_plotted)
            # cv2.imshow(filename, res_plotted)
            # if cv2.waitKey(1) == ord('q'):
            #     break
    video.release()
    writer.release()


# Load the models
model1 = YOLO('yolov8n.pt')
model2 = YOLO('yolov8n-seg.pt')

# Define the video files for the trackers
video_file1 = '/home/cdy/Data_HDD/Data/Project_SPC/test_data/test1_p1.mp4'
video_file2 = '/home/cdy/Data_HDD/Data/Project_SPC/test_data/test1_p2.mp4'

# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file1, model1),
                                   daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread,
                                   args=(video_file2, model2),
                                   daemon=True)

# Start the tracker threads
tracker_thread1.start()
tracker_thread2.start()

# Wait for the tracker threads to finish
tracker_thread1.join()
tracker_thread2.join()

# Clean up and close windows
cv2.destroyAllWindows()
