import logging
import torch
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.solver import build_optimizer
from detectron2.modeling import build_model
from myCode.register_data import register_deepfashion
import time
# from data_augmentation import build_train_loader


def main():
    # # 禁用日志记录
    logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().disabled = True
    # logging.getLogger().disabled = True

    # 从配置文件获取配置
    cfg = get_cfg()

    # 确保文件路径使用原始字符串
    config_path = r"D:\detectron2-main\myCode\cfg.yaml"

    cfg.merge_from_file(config_path)

    # 使用SimpleTrainer
    train_loader = build_detection_train_loader(cfg)  # 数据加载器
    # test_loader = build_detection_test_loader(cfg, "deepfashion_val")  # 假设你使用的是“deepfashion_val”作为验证集
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    trainer = SimpleTrainer(model, train_loader, optimizer)

    # 使用DefaultTrainer
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)

    # 训练模型
    start_time = time.time()
    trainer.train(0, 300)
    end_time = time.time()
    print("消耗时间为：", end_time - start_time)

    # 保存模型
    # trainer.model.save(cfg.OUTPUT_DIR + "/model_final.pth")
    state_dicts = trainer.model.state_dict()
    torch.save(state_dicts, cfg.OUTPUT_DIR + "/model_final.pth")


if __name__ == '__main__':
    main()
