#!/usr/bin/env python3

# COMMON LIBRARIES
import os
import cv2
import pickle
import json
import shutil

from datetime import datetime
from roboflow import Roboflow
import matplotlib.pyplot as plt

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor

# TRAINING
from detectron2.engine import DefaultTrainer

# Delete Model Folder
MODEL_DIR = os.getcwd() + "/model"
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)
    print("Deleting Model Folder!")


# Download dataset from Roboflow
rf = Roboflow(api_key="QO1iBTAWSmIJ28ZyKyVr")
project = rf.workspace("socrob").project("door-handles")
dataset = project.version(7).download("coco-segmentation")

DATA_SET_NAME = dataset.name.replace(" ", "-")
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"

# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "train", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TRAIN_DATA_SET_NAME, 
    metadata={}, 
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH, 
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# TEST SET
TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "test")
TEST_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "test", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TEST_DATA_SET_NAME, 
    metadata={}, 
    json_file=TEST_DATA_SET_ANN_FILE_PATH, 
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)

# VALID SET
VALID_DATA_SET_NAME = f"{DATA_SET_NAME}-valid"
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "valid")
VALID_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "valid", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=VALID_DATA_SET_NAME, 
    metadata={}, 
    json_file=VALID_DATA_SET_ANN_FILE_PATH, 
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)


# Dataset Upload Validation
# print([ data_set for data_set in MetadataCatalog.list() if data_set.startswith(DATA_SET_NAME)])

metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)
dataset_train = DatasetCatalog.get(TRAIN_DATA_SET_NAME)

dataset_entry = dataset_train[0]
image = cv2.imread(dataset_entry["file_name"])

visualizer = Visualizer(
    image[:, :, ::-1],
    metadata=metadata, 
    scale=0.8, 
    instance_mode=ColorMode.IMAGE_BW
)

out = visualizer.draw_dataset_dict(dataset_entry)
plt.figure(figsize = (15, 20))
plt.imshow(out.get_image())
plt.show()


# SAVE METADATA
if not os.path.exists(MODEL_DIR):
    print("Creating Model Directory!")
    os.makedirs(MODEL_DIR)

METADATA_DIR = MODEL_DIR + "/dataset_metadata.json"
dataset_metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)

dictionary = {
    "evaluator_type": dataset_metadata.evaluator_type,
    "image_root": dataset_metadata.image_root,
    "json_file": dataset_metadata.json_file,
    "name": dataset_metadata.name,
    "thing_classes": dataset_metadata.thing_classes,
    "thing_dataset_id_to_contiguous_id": dataset_metadata.thing_dataset_id_to_contiguous_id
}
# Serializing json
json_object = json.dumps(dictionary, indent=4)

# Writing to sample.json
with open(METADATA_DIR, "w") as outfile:
    outfile.write(json_object)


# HYPERPARAMETERS
ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
MAX_ITER = 1000
EVAL_PERIOD = 200
BASE_LR = 0.001
NUM_CLASSES = 3
CFG_PATH = os.getcwd() + "/model/IS_cfg.pickle"

# OUTPUT DIR
# OUTPUT_DIR_PATH = os.path.join("model", DATA_SET_NAME, ARCHITECTURE, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
OUTPUT_DIR_PATH = "model/output/instance_segmentation"
os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)
cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME,)
cfg.DATASETS.TEST = (TEST_DATA_SET_NAME,)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.TEST.EVAL_PERIOD = EVAL_PERIOD
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
cfg.OUTPUT_DIR = OUTPUT_DIR_PATH

# SAVE CONFIGS IN PICKLE FILE
with open(CFG_PATH, "wb") as f:
    pickle.dump(cfg, f, protocol = pickle.HIGHEST_PROTOCOL)


# TRAIN MODEL
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



