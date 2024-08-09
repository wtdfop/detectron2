import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import socket
import numpy as np


def load_predictor(cfg_path, model_path):
    config = get_cfg()
    config.merge_from_file(cfg_path)
    config.MODEL.WEIGHTS = model_path
    config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 设置测试的阈值
    config.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(config)
    return predictor


def get_result(image):
    cfg_path = "D:\\detectron2-main\\myCode\\cfg.yaml"
    model_path = "D:\\detectron2-main\\run\\output\\model_final.pth"
    predictor = load_predictor(cfg_path, model_path)
    output = predictor(image)
    return output


def get_user_data():
    tcp_socket = socket.socket()
    tcp_socket.bind(("192.168.0.104", 8080))
    while True:
        tcp_socket.listen(5)
        conn, addr = tcp_socket.accept()
        while True:
            length = int.from_bytes(conn.recv(4), 'big')
            if length == 0:
                break
            data = b''
            data += conn.recv(length)
            nparray = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparray, cv2.IMREAD_COLOR)
            outputs = get_result(img)
            response = str(outputs['instances'].pred_classes)
            conn.send(response.encode(encoding="utf-8"))
        conn.close()


if __name__ == "__main__":
    get_user_data()
