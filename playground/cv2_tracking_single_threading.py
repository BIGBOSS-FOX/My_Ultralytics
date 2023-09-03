import cv2
import torch
from pathlib import Path
from ultralytics.trackers import BOTSORT
from ultralytics.utils import IterableSimpleNamespace
from ultralytics import YOLO

video_filepaths = [
    '/home/cdy/Data_HDD/Data/Project_SPC/test_data/test1_p1.mp4',
    '/home/cdy/Data_HDD/Data/Project_SPC/test_data/test1_p2.mp4',
]

weight_path = 'yolov8m.pt'

start_frames = [0, 0]
end_frames = [-1, -1]
target_fpss = [5, 5]

tracker_cfg = IterableSimpleNamespace(**{
    'tracker_type': 'botsort',
    'track_high_thresh': 0.5,
    'track_low_thresh': 0.1,
    'new_track_thresh': 0.6,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'gmc_method': 'sparseOptFlow',
    'proximity_thresh': 0.5,
    'appearance_thresh': 0.25,
    'with_reid': False
})

caps = [cv2.VideoCapture(vfp) for vfp in video_filepaths]
cap_fpss = [cap.get(cv2.CAP_PROP_FPS) for cap in caps]
intervals = [int(cap_fps / target_fps) for cap_fps, target_fps in zip(cap_fpss, target_fpss)]

for i, cap in enumerate(caps):
    if start_frames[i] > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frames[i])
    if end_frames[i] <= 0:
        end_frames[i] = cap.get(cv2.CAP_PROP_FRAME_COUNT)

save_filepaths_p = [Path(vfp).parent / f"{Path(vfp).stem}_inference.mp4" for vfp in video_filepaths]
writers = [cv2.VideoWriter(str(sfp), cv2.VideoWriter_fourcc(*'mp4v'), target_fpss[i], (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))) for i, (cap, sfp) in enumerate(zip(caps, save_filepaths_p))]

release_cap_count = 0

trackers = [BOTSORT(tracker_cfg) for _ in caps]

model = YOLO(weight_path)

while release_cap_count < len(caps):
    for i, cap in enumerate(caps):
        if cap.isOpened():
            ret, frame = cap.read()
            frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if ret:
                if frame_pos % intervals[i] == 0:
                    # Do inference
                    results = model(frame)
                    det = results[0].boxes.cpu().numpy()
                    if len(det) == 0:
                        continue
                    tracks = trackers[i].update(det, frame)
                    if len(tracks) == 0:
                        continue
                    idx = tracks[:, -1].astype(int)
                    results[0] = results[0][idx]
                    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

                    # Visualize the results on the frame
                    annotated_frame = results[0].plot()
                    # print(results)
                    writers[i].write(annotated_frame)
            elif frame_pos >= end_frames[i]:
                cap.release()
                release_cap_count += 1
            else:
                cap.release()
                release_cap_count += 1
        else:
            release_cap_count += 1