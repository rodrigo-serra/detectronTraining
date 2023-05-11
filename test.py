#!/usr/bin/env python3

from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import os, pickle, json, random, cv2
import matplotlib.pyplot as plt


CFG_PATH = os.getcwd() + "/model/IS_cfg.pickle"
with open(CFG_PATH, "rb") as f:
    cfg = pickle.load(f)

METADATA_PATH = os.getcwd() + "/model/dataset_metadata.json"
with open(METADATA_PATH, 'r') as openfile:
    jsonObj = json.load(openfile)

MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).evaluator_type = jsonObj["evaluator_type"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).image_root = jsonObj["image_root"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).json_file = jsonObj["json_file"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = jsonObj["thing_classes"]
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_dataset_id_to_contiguous_id = jsonObj["thing_dataset_id_to_contiguous_id"]

DATASET_METADATA = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
predictor = DefaultPredictor(cfg)

# TESTING
valid_dataset_dir = jsonObj["image_root"].replace("/train", "/valid")

img_list = []
for path in os.listdir(valid_dataset_dir):
    if not ('json' in path):
        img_list.append(valid_dataset_dir + '/' + path)


IMG_NUMBER = 10
for img_path in random.sample(img_list, IMG_NUMBER):
    img = cv2.imread(img_path)
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], metadata = DATASET_METADATA, scale = 0.5, instance_mode = ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(v.get_image())
    plt.show()

