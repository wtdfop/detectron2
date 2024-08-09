from detectron2.data import build_detection_train_loader
from detectron2.config import get_cfg


def main():
    cfg = get_cfg()

    config_path = r"D:\detectron2-main\myCode\config\con.yaml"

    cfg.merge_from_file(config_path)

    train_loader = build_detection_train_loader(cfg)
    for data in train_loader:
        print(data)
        break


if __name__ == '__main__':
    main()
