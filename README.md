# My Ultralytics

## Updates

2023/8/31:

- Test cli commands
- Write `playground/cv2_inference.py`

2023/9/1:

- Create `myultralytics`
- Implement tracking in `playground/cv2_inference.py`
- Test `playground/cv2_tracking_multi_threading.py`. Comfirm diffent trackers share the same track id pool.

2023/9/3:

- Implement `playground/cv2_tracking_single_threading.py`
- Test `BaseTracker` in `playground/tracker_exp1.py`

2023/9/7:

- Change track id pool from global to per tracker in `playground/my_trackers.py`
- Add option to change number of loops per video playback in `cv2_tracking_single_threading.py`
