import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
import register_data

# 获取数据集和元数据
dataset_dicts = DatasetCatalog.get("deepfashion_train")
metadata = MetadataCatalog.get("deepfashion_train")

for d in random.sample(dataset_dicts, 1):  # 随机抽取3张图像进行可视化
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("Dataset Visualization", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)  # 按键继续
cv2.destroyAllWindows()
