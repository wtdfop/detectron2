import pickle
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from myCode import register_data

# 获取数据集和元数据
# register_data.register_deepfashion()
file_path = "D:\\detectron2-main\\MiniProject\\data.pkl"
with open(file_path, 'rb') as f:
    var = pickle.load(f)
dataset_dicts = var["var1"]
for d in ["train", "val"]:
    DatasetCatalog.register("mini_deepfashion_" + d, lambda: dataset_dicts[d])
    MetadataCatalog.get("mini_deepfashion_" + d).set(
        thing_classes=["Anorak", "Blazer", "Blouse", "Bomber", "Button-Down", "Cardigan",
                       "Flannel", "Halter", "Henley", "Hoodie", "Jacket", "Jersey",
                       "Parka", "Peacoat", "Poncho", "Sweater", "Tank", "Tee",
                       "Top", "Turtleneck", "Capris", "Chinos", "Culottes", "Cutoffs",
                       "Gauchos", "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
                       "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts", "Trunks",
                       "Caftan", "Cape", "Coat", "Coverup", "Dress", "Jumpsuit",
                       "Kaftan", "Kimono", "Nightdress", "Onesie", "Robe", "Romper",
                       "Shirtdress", "Sundress"])
dicts = DatasetCatalog.get("mini_deepfashion_train")
metadata = MetadataCatalog.get("mini_deepfashion_train")

for d in random.sample(dicts, 3):  # 随机抽取3张图像进行可视化
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow("Dataset Visualization", vis.get_image()[:, :, ::-1])
    cv2.waitKey(0)  # 按键继续
cv2.destroyAllWindows()
