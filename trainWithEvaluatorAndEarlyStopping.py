#!/usr/bin/env python3

# COMMON LIBRARIES
import os
import cv2
import pickle
import json
import shutil
import torch

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
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

# TRAINING
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader


from utils import savetrainInfo, downloadDataset



class HybridEarlyStoppingTrainer(DefaultTrainer):
    def __init__(self, cfg, patience=5, min_delta=0.001, loss_patience=300, loss_min_delta=0.005):
        super().__init__(cfg)
        self.patience = patience
        self.min_delta = min_delta
        self.best_metric = None
        self.metric_counter = 0

        self.loss_patience = loss_patience
        self.loss_min_delta = loss_min_delta
        self.best_loss = None
        self.loss_counter = 0

    def build_evaluator(self, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        
        if not hasattr(self, '_data_loader_iter'):
            self._data_loader_iter = iter(self.data_loader)
        
        data = next(self._data_loader_iter)
        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        
        end.record()
        torch.cuda.synchronize()
        
        self.storage.put_scalar("time", start.elapsed_time(end) / 1000.0)
        self.storage.put_scalars(**loss_dict)
        
        if self.iter % self.cfg.TEST.EVAL_PERIOD == 0 and self.iter != 0:
            results = self.test(self.cfg, self.model)
            metric = results['segm']['AP']  # Using AP for segmentation as an example

            if self.best_metric is None or metric > self.best_metric + self.min_delta:
                self.best_metric = metric
                self.metric_counter = 0
            else:
                self.metric_counter += 1
                if self.metric_counter >= self.patience:
                    print("Early stopping triggered at iteration due to validation metric: ", self.iter)
                    raise StopIteration

        # Secondary check using training loss
        current_loss = losses.item()
        if self.best_loss is None or current_loss < self.best_loss - self.loss_min_delta:
            self.best_loss = current_loss
            self.loss_counter = 0
        else:
            self.loss_counter += 1
            if self.loss_counter >= self.loss_patience:
                print("Early stopping triggered at iteration due to training loss: ", self.iter)
                raise StopIteration

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        if evaluators is None:
            evaluators = [cls.build_evaluator(cfg, name) for name in cfg.DATASETS.TEST]
        data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        return inference_on_dataset(model, data_loader, evaluators)



# PARAMETERS & HYPERPARAMETERS CONFIG
f = open("config.json")
train_config = json.load(f)

RESUME_TRAINING = True
if train_config["RESUME_TRAINING"] == "False":
    RESUME_TRAINING = False

SHOW_IMAGE = True
if train_config["SHOW_IMAGE"] == "False":
    SHOW_IMAGE = False

MODEL_DIR = os.getcwd() + "/model"
ARCHITECTURE = train_config["ARCHITECTURE"]
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
NUM_WORKERS = train_config["NUM_WORKERS"]
WARMUP_ITERS = train_config["WARMUP_ITERS"]
WARMUP_FACTOR = 1.0 / 1000
IMS_PER_BATCH = train_config["IMS_PER_BATCH"]
BASE_LR = train_config["BASE_LR"]
MAX_ITER = train_config["MAX_ITER"]
STEPS = (train_config["STEPS"][0], train_config["STEPS"][1])
GAMMA = train_config["GAMMA"]
CHECKPOINT_PERIOD = train_config["CHECKPOINT_PERIOD"]

AMP_ENABLED = True
if train_config["AMP_ENABLED"] == "False":
    AMP_ENABLED = False

BATCH_SIZE_PER_IMAGE = train_config["BATCH_SIZE_PER_IMAGE"]
EVAL_PERIOD = train_config["EVAL_PERIOD"]
MASK_FORMAT = train_config["MASK_FORMAT"]
CFG_PATH = os.getcwd() + "/model/IS_cfg.pickle"
OUTPUT_DIR_PATH = "model/output/instance_segmentation"
# OUTPUT_DIR_PATH = os.path.join("model", DATA_SET_NAME, ARCHITECTURE, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))


# Delete Model Folder
if not RESUME_TRAINING:
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
        print("Deleting Model Folder!")



# DOWNLOAD DATASET
dataset = downloadDataset()

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


# DATASET UPLOAD VALIDATION (SHOW IMAGE)
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
if SHOW_IMAGE:
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

# Specifying number of classes
NUM_CLASSES = len(dictionary["thing_classes"])

os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
cfg.OUTPUT_DIR = OUTPUT_DIR_PATH


if RESUME_TRAINING:
    # Assuming the path to the last checkpoint file
    last_checkpoint = os.path.join(cfg.OUTPUT_DIR, "last_checkpoint")
    print(last_checkpoint)
    # Read the path of the last checkpoint
    if os.path.exists(last_checkpoint):
        with open(last_checkpoint, "r") as f:
            last_checkpoint_path = f.read().strip()

    # Print the checkpoint path for verification
    print(f"Resuming from checkpoint: {last_checkpoint_path}")

    # Update the config to load the last checkpoint
    cfg.MODEL.WEIGHTS = last_checkpoint_path
else:
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(CONFIG_FILE_PATH)



cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME,)
cfg.DATASETS.TEST = (VALID_DATA_SET_NAME,)

cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS

cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
cfg.SOLVER.WARMUP_FACTOR = WARMUP_FACTOR
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.STEPS = STEPS
cfg.SOLVER.GAMMA = GAMMA
cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
cfg.SOLVER.AMP.ENABLED = AMP_ENABLED

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = BATCH_SIZE_PER_IMAGE
cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES

cfg.TEST.EVAL_PERIOD = EVAL_PERIOD

cfg.INPUT.MASK_FORMAT= MASK_FORMAT

# SAVE CONFIGS IN PICKLE FILE
with open(CFG_PATH, "wb") as f:
    pickle.dump(cfg, f, protocol = pickle.HIGHEST_PROTOCOL)


# SAVE TRAINING INFO
# savetrainInfo(filename="trained_model_info.txt",
#               dictionary=dictionary,
#               cfg=cfg,
#               data_set_name=DATA_SET_NAME,
#               model_dir=MODEL_DIR,
#               train_data_set_name=TRAIN_DATA_SET_NAME,
#               train_data_set_images_dir_path=TRAIN_DATA_SET_IMAGES_DIR_PATH,
#               train_data_set_ann_file_path=TRAIN_DATA_SET_ANN_FILE_PATH,
#               test_data_set_name=TEST_DATA_SET_NAME,
#               test_data_set_images_dir_path=TEST_DATA_SET_IMAGES_DIR_PATH,
#               test_data_set_ann_file_path=TEST_DATA_SET_ANN_FILE_PATH,
#               valid_data_set_name=VALID_DATA_SET_NAME,
#               valid_data_set_images_dir_path=VALID_DATA_SET_IMAGES_DIR_PATH,
#               valid_data_set_ann_file_path=VALID_DATA_SET_ANN_FILE_PATH,
#               metadata_dir=METADATA_DIR,
#               config_file_path=CONFIG_FILE_PATH,
#               output_dir_path=OUTPUT_DIR_PATH,
#               cfg_path=CFG_PATH)



# The rest of your training script remains unchanged, except for using HybridEarlyStoppingTrainer
trainer = HybridEarlyStoppingTrainer(cfg, patience=5, min_delta=0.001, loss_patience=300, loss_min_delta=0.005)
trainer.resume_or_load(resume=RESUME_TRAINING   )

try:
    trainer.train()
except StopIteration:
    print("Training stopped early due to early stopping criteria.")


