import detectron2.data.transforms as T
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, instantiate
from detectron2.data import MetadataCatalog
from omegaconf import OmegaConf

class DefaultPredictor_Lazy:
    def __init__(self, cfg, device="cuda"):
        self.device = torch.device(device)

        if isinstance(cfg, CfgNode):
            self.cfg = cfg.clone()
            self.model = build_model(self.cfg)  # noqa: F821
            test_dataset = cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else None

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)

            self.aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
            )
            self.input_format = cfg.INPUT.FORMAT
        else:
            self.cfg = cfg
            self.model = instantiate(cfg.model)
            test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
            if isinstance(test_dataset, (list, tuple)):
                test_dataset = test_dataset[0]

            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

            mapper = instantiate(cfg.dataloader.test.mapper)
            self.aug = mapper.augmentations
            self.input_format = mapper.image_format

        self.model.eval().to(self.device)

        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(self.device)  # <--- move input too
            inputs = {"image": image, "height": height, "width": width}
            return self.model([inputs])[0]
