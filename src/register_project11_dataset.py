from pathlib import Path

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog


PROJECT_ROOT = Path.home() / "project_12_detectron2_training"
DATASET_ROOT = PROJECT_ROOT / "data" / "project11_detection"

TRAIN_JSON = DATASET_ROOT / "annotations" / "instances_train.json"
VAL_JSON = DATASET_ROOT / "annotations" / "instances_val.json"
IMAGES_DIR = DATASET_ROOT / "images"

TRAIN_DATASET_NAME = "project11_detection_train"
VAL_DATASET_NAME = "project11_detection_val"


def register_datasets():
    register_coco_instances(
        TRAIN_DATASET_NAME,
        {},
        str(TRAIN_JSON),
        str(IMAGES_DIR),
    )

    register_coco_instances(
        VAL_DATASET_NAME,
        {},
        str(VAL_JSON),
        str(IMAGES_DIR),
    )


if __name__ == "__main__":
    register_datasets()

    train_records = DatasetCatalog.get(TRAIN_DATASET_NAME)
    metadata = MetadataCatalog.get(TRAIN_DATASET_NAME)

    print("Loaded train records:", len(train_records))
    print("Classes:", metadata.thing_classes)