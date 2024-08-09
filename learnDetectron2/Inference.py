import os
import random
import cv2
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from learnDetectron2.register import get_balloon_dicts
from learnDetectron2.train import cfg
from detectron2.utils.visualizer import ColorMode, Visualizer


if __name__ == "__main__":
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    cfg.MODEL.WEIGHTS = "D:/detectron2-main/run/output/learn_model_final.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_balloon_dicts("E:\\train_img\\balloon_img\\balloon\\val")
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model
        # -output-format
        balloon_metadata = MetadataCatalog.get("balloon_train")
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("D:\\detectron2-main\\run\\2.jpg", out.get_image()[:, :, ::-1])
