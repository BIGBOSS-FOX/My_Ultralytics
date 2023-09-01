import os
import sys

# Add the path to the myultralytics package to the module search path
myultralytics_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, myultralytics_path)

# Now you can import myultralytics as usual
from myultralytics import YOLO

model = YOLO("/home/cdy/Data_HDD/Weights/ultralytics/yolov8m.pt")
print()