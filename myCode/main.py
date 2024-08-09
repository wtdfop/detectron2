import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from myCode.testCode import cfg

# 加载配置和模型
def load_model(cfg_path, model_path):
    config = get_cfg()
    config.merge_from_file(cfg_path)
    config.MODEL.WEIGHTS = model_path
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # 设置测试的阈值
    config.MODEL.DEVICE = "cuda"
    model = DefaultPredictor(config)
    return model


# 主函数
def main():
    # 配置文件和模型权重文件路径
    cfg_path = "D:\\detectron2-main\\myCode\\cfg.yaml"
    model_path = "D:\\detectron2-main\\run\\output\\model_final.pth"

    # 加载模型
    model = load_model(cfg_path, model_path)

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while True:
        # 读取摄像头的帧
        ret, frame = cap.read()
        if not ret:
            break

        # 使用模型进行预测
        outputs = model(frame)

        # 将预测结果可视化
        v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # 显示图像
        cv2.imshow("frame", out.get_image()[:, :, ::-1])

        # 按'q'退出
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放摄像头
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
