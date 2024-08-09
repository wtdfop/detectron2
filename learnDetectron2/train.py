import os
import torch
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.model_zoo import model_zoo
from detectron2.modeling import build_model
from detectron2.solver import build_optimizer
from learnDetectron2 import register
import time


cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file("D:/detectron2-main/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("balloon_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00005  # pick a good LR
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (balloon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

if __name__ == "__main__":
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # 使用SimpleTrainer
    train_loader = build_detection_train_loader(cfg)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    trainer = SimpleTrainer(model, train_loader, optimizer)
    start = time.time()
    trainer.train(0, 300)
    end = time.time()
    print("消耗时间：", end - start)

    # 使用DefaultTrainer
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)
    # trainer.train()

    # torch.save(trainer.state_dict(), "D:/detectron2-main/run/output/learn_model_final.pth")
    state = trainer.model.state_dict()
    torch.save(state, "D:/detectron2-main/run/output/learn_model_final.pth")