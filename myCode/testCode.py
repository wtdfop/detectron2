import cv2
import os
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# 配置模型
cfg = get_cfg()
cfg.merge_from_file("D:/detectron2-main/configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml")
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # 模型权重文件路径
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置检测阈值
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 使用 GPU 或 CPU
print(torch.cuda.is_available())
# 创建预测器
predictor = DefaultPredictor(cfg)

# 读取图像
image_path = "D:\\detectron2-main\\myImg\\xie.jpeg"  # 替换为你的图像路径
image = cv2.imread(image_path)

if image is None:
    print(f"Failed to read image from path: {image_path}")
else:
    print("Image successfully read from path")

    # 进行推断
    outputs = predictor(image)
    print("Inference outputs:", outputs)

    # 检查推断结果是否包含实例
    if "instances" in outputs:
        instances = outputs["instances"]
        print(f"Number of detected instances: {len(instances)}")
    else:
        print("No instances detected.")

    # 可视化结果
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_image = v.get_image()[:, :, ::-1]

    # 确保输出目录存在
    output_dir = "D:\\detectron2-main\\run"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存结果图像
    output_path = os.path.join(output_dir, "output.jpg")
    success = cv2.imwrite(output_path, result_image)
    if success:
        print(f"Result image successfully saved to {output_path}")
    else:
        print(f"Failed to save result image to {output_path}")
