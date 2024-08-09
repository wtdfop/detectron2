import os
import cv2
import pickle

from detectron2.structures import BoxMode


def get_train_test_val():
    filepath = "E:/train_img/003/Category and Attribute Prediction Benchmark/Eval/list_eval_partition.txt"
    img_type = {}
    with open(filepath) as f:
        lines = f.readlines()
        lines = lines[2:]  # 跳过前两行
        for line in lines:
            p1 = line.find("jpg") + 3  # 找到编号
            img_num = line[:p1]  # 分离出图像编号，命名为img_num
            if line.find("train") != -1:
                img_type.update({img_num: "train"})
            elif line.find("test") != -1:
                img_type.update({img_num: "test"})
            elif line.find("val") != -1:
                img_type.update({img_num: "val"})
    return img_type


def get_deepfashion_dicts(img_type, img_dir, anno_dir):
    dataset_dicts = {"train": [], "val": [], "test": []}
    img_category = {
        "Anorak": 0,
        "Blazer": 1,
        "Blouse": 2,
        "Bomber": 3,
        "Button-Down": 4,
        "Cardigan": 5,
        "Flannel": 6,
        "Halter": 7,
        "Henley": 8,
        "Hoodie": 9,
        "Jacket": 10,
        "Jersey": 11,
        "Parka": 12,
        "Peacoat": 13,
        "Poncho": 14,
        "Sweater": 15,
        "Tank": 16,
        "Tee": 17,
        "Top": 18,
        "Turtleneck": 19,
        "Capris": 20,
        "Chinos": 21,
        "Culottes": 22,
        "Cutoffs": 23,
        "Gauchos": 24,
        "Jeans": 25,
        "Jeggings": 26,
        "Jodhpurs": 27,
        "Joggers": 28,
        "Leggings": 29,
        "Sarong": 30,
        "Shorts": 31,
        "Skirt": 32,
        "Sweatpants": 33,
        "Sweatshorts": 34,
        "Trunks": 35,
        "Caftan": 36,
        "Cape": 37,
        "Coat": 38,
        "Coverup": 39,
        "Dress": 40,
        "Jumpsuit": 41,
        "Kaftan": 42,
        "Kimono": 43,
        "Nightdress": 44,
        "Onesie": 45,
        "Robe": 46,
        "Romper": 47,
        "Shirtdress": 48,
        "Sundress": 49
    }
    for txt_file in os.listdir(anno_dir):
        if txt_file == "list_bbox.txt":
            path_file = os.path.join(anno_dir, txt_file)
            with open(path_file) as f:
                lines = f.readlines()
                lines = lines[2:]
                for idx, line in enumerate(lines):
                    line = line.strip().split()  # 去除空格，并将一行读入转换为元素量为5的列表
                    img_name = line[0]
                    img_path = os.path.join(img_dir, img_name)
                    bbox = [float(x) for x in line[1:5]]
                    ends = img_name.find("/img_")  # ends保存"/img_"的位置，即类别字符的最后一位的后一位
                    L = img_name[:ends]  # 这里L保存"/img_"字符串之前的字符
                    last_underscore_index = L.rfind('_')
                    img = L[last_underscore_index + 1:]

                    category_id = img_category[img]

                    # 每个图像的信息
                    record = {
                        "file_name": img_path,
                        "image_id": idx,
                        "height": 0,  # Placeholder, will be populated below
                        "width": 0,  # Placeholder, will be populated below
                        "annotations": [{
                            "bbox": bbox,
                            "bbox_mode": BoxMode.XYWH_ABS,
                            "category_id": category_id
                        }]
                    }

                    # 获取图像的实际大小
                    height, width = cv2.imread(img_path).shape[:2]
                    record["height"] = height
                    record["width"] = width

                    dataset_dicts[img_type[img_name]].append(record)

    return dataset_dicts


if __name__ == "__main__":
    file = "D:/detectron2-main/MyCode/FashionData.pkl"
    img_dir = f"E:/train_img/003"
    anno_dir = f"E:/train_img/003/Category and Attribute Prediction Benchmark/Anno_coarse"
    img_type = get_train_test_val()

    data_dicts = get_deepfashion_dicts(img_type, img_dir, anno_dir)

    with open(file, 'wb') as f:
        pickle.dump(data_dicts, f)  # 存储数据
