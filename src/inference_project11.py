from pathlib import Path
import cv2

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from register_project11_dataset import (
    VAL_DATASET_NAME,
    register_datasets,
)

PROJECT_ROOT = Path.home() / "project_12_detectron2_training"
DATASET_ROOT = PROJECT_ROOT / "data" / "project11_detection"
IMAGES_DIR = DATASET_ROOT / "images"

OUTPUT_DIR = PROJECT_ROOT / "output" / "faster_rcnn_project11"
MODEL_WEIGHTS = OUTPUT_DIR / "model_final.pth"
VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def setup_cfg():
    cfg = get_cfg()
    config_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))

    cfg.MODEL.WEIGHTS = str(MODEL_WEIGHTS)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.DATASETS.TEST = (VAL_DATASET_NAME,)
    cfg.MODEL.DEVICE = "cuda"

    return cfg


def main():
    register_datasets()
    _ = DatasetCatalog.get(VAL_DATASET_NAME)
    metadata = MetadataCatalog.get(VAL_DATASET_NAME)

    predictor = DefaultPredictor(setup_cfg())
    records = DatasetCatalog.get(VAL_DATASET_NAME)

    print(f"Running inference on {len(records)} validation images.")

    for record in records:
        image_path = IMAGES_DIR / Path(record["file_name"]).name
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        outputs = predictor(image)

        vis = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
        out = vis.draw_instance_predictions(outputs["instances"].to("cpu"))

        save_path = VIS_DIR / Path(record["file_name"]).name
        cv2.imwrite(str(save_path), out.get_image()[:, :, ::-1])
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()