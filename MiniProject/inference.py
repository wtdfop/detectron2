import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


def load_model(cfg_path, model_path):
    config = get_cfg()
    config.merge_from_file(cfg_path)
    config.MODEL.WEIGHTS = model_path
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试的阈值
    config.MODEL.DEVICE = "cuda"
    model = DefaultPredictor(config)
    return model


if __name__ == "__main__":
    cfg_path = "D:\\detectron2-main\\myCode\\cfg.yaml"
    model_path = "D:\\detectron2-main\\run\\mini_output\\model_final.pth"
    img_path = "D:\\detectron2-main\\myImg\\img_00000005.jpg"

    model = load_model(cfg_path, model_path)

    image = cv2.imread(img_path)

    outputs = model(image)
    print(outputs)

    # 将预测结果可视化
    v = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get("deepfashion_train"), scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # 显示图像
    cv2.imshow("Predicted Image", out.get_image()[:, :, ::-1])

    # 保存结果图像
    output_image_path = "D:\\detectron2-main\\run\\t3.jpg"
    cv2.imwrite(output_image_path, out.get_image()[:, :, ::-1])
