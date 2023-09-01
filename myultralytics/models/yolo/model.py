from myultralytics.engine.model import Model


class YOLO(Model):
    """
    YOLO (You Only Look Once) object detection model.
    """

    @property
    def task_map(self):
        raise NotImplementedError
