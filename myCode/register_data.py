import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import pickle
from detectron2.structures import BoxMode


def register_deepfashion():
    # img_dir = f"E:/train_img/003"
    # anno_dir = f"E:/train_img/003/Category and Attribute Prediction Benchmark/Anno_coarse"
    # img_type = get_train_test_val()
    file = "D:/detectron2-main/MyCode/FashionData.pkl"
    with open(file, 'rb') as f:
        dataset_dicts = pickle.load(f)
    for d in ["train", "val", "test"]:
        DatasetCatalog.register("deepfashion_" + d, lambda: dataset_dicts[d])
        MetadataCatalog.get("deepfashion_" + d).set(
            thing_classes=["Anorak", "Blazer", "Blouse", "Bomber", "Button-Down", "Cardigan",
                           "Flannel", "Halter", "Henley", "Hoodie", "Jacket", "Jersey",
                           "Parka", "Peacoat", "Poncho", "Sweater", "Tank", "Tee",
                           "Top", "Turtleneck", "Capris", "Chinos", "Culottes", "Cutoffs",
                           "Gauchos", "Jeans", "Jeggings", "Jodhpurs", "Joggers", "Leggings",
                           "Sarong", "Shorts", "Skirt", "Sweatpants", "Sweatshorts", "Trunks",
                           "Caftan", "Cape", "Coat", "Coverup", "Dress", "Jumpsuit",
                           "Kaftan", "Kimono", "Nightdress", "Onesie", "Robe", "Romper",
                           "Shirtdress", "Sundress"])


register_deepfashion()
