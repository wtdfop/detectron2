import logging
import torch
import os
import pickle
import detectron2.utils.logger as logger
from detectron2.engine import DefaultTrainer, SimpleTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.solver import build_optimizer
from detectron2.modeling import build_model


def main(sampled_dataset_dicts=None):
    # # 禁用日志记录
    logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().disabled = True
    # logging.getLogger().disabled = True

    # 从配置文件获取配置
    cfg = get_cfg()

    # 确保文件路径使用原始字符串
    config_path = r"D:\detectron2-main\MiniProject\config.yaml"

    # 确保文件存在
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' does not exist!")

    # 使用原始字符串和正确的文件编码读取配置文件
    with open(config_path, 'r', encoding='utf-8'):
        cfg.merge_from_file(config_path)

    # 数据加载器
    train_loader = build_detection_train_loader(cfg)
    # test_loader = build_detection_test_loader(cfg, "deepfashion_val")  # 假设你使用的是“deepfashion_val”作为验证集

    # 构建模型
    model = build_model(cfg)

    # 构建优化器
    optimizer = build_optimizer(cfg, model)

    # 构建训练器
    trainer = SimpleTrainer(model, train_loader, optimizer)
    # trainer = DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=False)

    # 训练模型
    trainer.train(0, 500)

    # 保存模型
    # trainer.model.save(cfg.OUTPUT_DIR + "/mini_model_final.pth")
    torch.save(model.state_dict(), cfg.OUTPUT_DIR + "/mini_model_final.pth")


if __name__ == '__main__':
    file_path = "D:\\detectron2-main\\MiniProject\\data.pkl"
    with open(file_path, 'rb') as f:
        var = pickle.load(f)
    dataset_dicts = var["var1"]
    for d in ["train", "val"]:
        DatasetCatalog.register("mini_deepfashion_"+d, lambda: dataset_dicts[d])
        MetadataCatalog.get("mini_deepfashion_"+d).set(
            thing_classes=["Anorak", "Blazer", "Blouse", "Bomber", "Button-Down", "Cardigan",
                           "Flannel", "Halter", "Henley", "Hoodie", "Jacket", "Jersey",
                           "Parka", "Peacoat", "Poncho", "Sweater", "Tank", "Tee",
                           "Top", "Turtleneck", "Capris", "Chinos", "Culottes", "Cutoffs",
                           "Gauchos", "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
                           "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts", "Trunks",
                           "Caftan", "Cape", "Coat", "Coverup", "Dress", "Jumpsuit",
                           "Kaftan", "Kimono", "Nightdress", "Onesie", "Robe", "Romper",
                           "Shirtdress", "Sundress"])
    main()
