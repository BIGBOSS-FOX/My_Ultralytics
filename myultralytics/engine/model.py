from pathlib import Path
from typing import Union

from myultralytics.nn.tasks import attempt_load_one_weight


class Model:
    """
    A base model class to unify apis for all the models.

    Args:
        model (str, Path): Path to the model file to load or create.
        task (Any, optional): Task type for the YOLO model. Defaults to None.

    Attributes:
        predictor (Any): The predictor object.
        model (Any): The model object.
        trainer (Any): The trainer object.
        task (str): The type of model task.
        ckpt (Any): The checkpoint object if the model loaded from *.pt file.
        cfg (str): The model configuration if loaded from *.yaml file.
        ckpt_path (str): The checkpoint file path.
        overrides (dict): Overrides for the trainer object.
        metrics (Any): The data for metrics.

    Methods:
        __call__(source=None, stream=False, **kwargs):
            Alias for the predict method.
        _new(cfg:str, verbose:bool=True) -> None:
            Initializes a new model and infers the task type from the model definitions.
        _load(weights:str, task:str='') -> None:
            Initializes a new model and infers the task type from the model head.
        _check_is_pytorch_model() -> None:
            Raises TypeError if the model is not a PyTorch model.
        reset() -> None:
            Resets the model modules.
        info(verbose:bool=False) -> None:
            Logs the model info.
        fuse() -> None:
            Fuses the model for faster inference.
        predict(source=None, stream=False, **kwargs) -> List[ultralytics.engine.results.Results]:
            Performs prediction using the YOLO model.

    Returns:
        list(ultralytics.engine.results.Results): The prediction results.
    """

    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        # self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        # self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        # self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        # self.overrides = {}  # overrides for trainer object
        # self.metrics = None  # validation/training metrics
        # self.session = None  # HUB session
        self.task = task  # task type
        model = str(model).strip()
        # print(model)

        # Load or create new YOLO model
        suffix = Path(model).suffix
        if suffix in ('.yaml', '.yml'):
            self._new(model, task)
        else:
            self._load(model, task)
    
    def _new(self, cfg: str, task=None, model=None, verbose=True):
        raise NotImplementedError
    
    def _load(self, weights: str, task=None):
        """
        Initializes a new model and infers the task type from the model head.

        Args:
            weights (str): model checkpoint to be loaded
            task (str | None): model task
        """
        suffix = Path(weights).suffix
        if suffix == '.pt':
            self.model, self.ckpt = attempt_load_one_weight(weights)
            # print(self.model)
            # print(self.ckpt)