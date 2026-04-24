from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from register_project11_dataset import (
    TRAIN_DATASET_NAME,
    VAL_DATASET_NAME,
    register_datasets,
)


PROJECT_ROOT = Path.home() / "project_12_detectron2_training"
OUTPUT_DIR = PROJECT_ROOT / "output" / "faster_rcnn_project11"


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = str(OUTPUT_DIR / "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


def setup_cfg():
    cfg = get_cfg()

    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))

    cfg.DATASETS.TRAIN = (TRAIN_DATASET_NAME,)
    cfg.DATASETS.TEST = (VAL_DATASET_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.CHECKPOINT_PERIOD = 100

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.TEST.EVAL_PERIOD = 100

    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    return cfg


def main():
    register_datasets()

    train_records = DatasetCatalog.get(TRAIN_DATASET_NAME)
    metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)

    print("Loaded train records:", len(train_records))
    print("Training classes:", metadata.thing_classes)

    cfg = setup_cfg()

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()