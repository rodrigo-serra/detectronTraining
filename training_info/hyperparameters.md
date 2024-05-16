## 1. Hyperparameters and Optimized Training

```
cfg.DATALOADER.NUM_WORKERS
```

This sets the number of data loading workers. More workers can speed up data loading but also increase memory usage.

```
cfg.SOLVER.IMS_PER_BATCH
```

This sets the number of images per batch across all GPUs. If you're using multiple GPUs, this value is split across them.

```
cfg.SOLVER.BASE_LR
```

This sets the base learning rate for the optimizer. Learning rate controls how much to change the model in response to the estimated error each time the model weights are updated. 


```
cfg.SOLVER.MAX_ITER
```

This is the maximum number of iterations (batches) the training process will run. Each iteration processes one batch of images.

```
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
```

Number of RoIs to sample per image during training for the RoI heads. This controls the number of positive/negative samples used for training the classifier and regressor.


```
cfg.MODEL.ROI_HEADS.NUM_CLASSES
```

Number of classes for the dataset you are training on. This should match the number of classes in your dataset. 


```
cfg.TEST.EVAL_PERIOD
```

This defines how often (in terms of iterations) the model will be evaluated on the validation set. Frequent evaluations provide more feedback during training but can also slow down the process.


#### Aditional Parameters
```
cfg.SOLVER.WARMUP_ITERS
```

Number of iterations for linear learning rate warmup. This helps in stabilizing training in the initial phase.

```
cfg.SOLVER.STEPS
```

A list of iteration numbers where the learning rate will be decreased. Common practice is to reduce the learning rate by a factor of 10 at these steps.

```
cfg.SOLVER.GAMMA
```

The factor by which the learning rate will be reduced when the iteration reaches the specified steps.

```
cfg.SOLVER.CHECKPOINT_PERIOD
```

How often to save checkpoints (in iterations). This helps in resuming training in case of interruptions.


```
cfg.SOLVER.IMS_PER_DEVICE
```

Number of images per batch per GPU. This is useful when you have multiple GPUs and you want to control the load on each GPU individually.


#### Example Configuration
```
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader
from detectron2.data.datasets import register_coco_instances

# Register the dataset (example for COCO format)
register_coco_instances("my_dataset_train", {}, "path/to/annotations.json", "path/to/images")

cfg = get_cfg()
cfg.merge_from_file(f"detectron2/configs/COCO-InstanceSegmentation/{ARCHITECTURE}.yaml")

# Set hyperparameters
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()  # No validation dataset
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100000
cfg.SOLVER.STEPS = (60000, 80000)  # Adjust based on your needs
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.CHECKPOINT_PERIOD = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Adjust based on your dataset

# Additional options
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold

# Output directory
cfg.OUTPUT_DIR = "./output"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

#### Notes:
- **Dataset Registration**: Make sure your dataset is registered properly if you are using a custom dataset;
- **Validation**: Consider adding a validation dataset to monitor overfitting;
- **Batch Size and Learning Rate**: If you change the **IMS_PER_BATCH**, you may need to adjust the **BASE_LR** accordingly. A common rule of thumb is to linearly scale the learning rate with the batch size;

### Hyperparameter tuning (optimization)
#### 1. Learning Rate and Schedule
- Warmup: Use a warmup period to stabilize training.
```
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
```

- Adjust Steps: Modify learning rate steps and gamma for learning rate decay.
```
cfg.SOLVER.STEPS = (30000, 60000)
cfg.SOLVER.GAMMA = 0.1
```


#### 2. Batch Size and Workers
- Increase Batch Size: If you have enough GPU memory, increase the batch size.
```
cfg.SOLVER.IMS_PER_BATCH = 8  # or higher, depending on your GPU capacity
```
and consequently change the **BASE_LR** parameter linerly as recommended.

- Increase Workers: Increase the number of data loading workers.
```
cfg.DATALOADER.NUM_WORKERS = 4  # or higher, depending on your CPU capacity
```

#### 3. Checkpointing
- Save checkpoints more frequently to prevent loss of progress.
```
cfg.SOLVER.CHECKPOINT_PERIOD = 2000
```

#### 4. Mixed Precision Training
- Use mixed precision training to speed up training and reduce memory usage.
```
cfg.SOLVER.AMP.ENABLED = True
```

#### 5. Data Augmentation
- Use data augmentation to improve model robustness. Common techniques include random cropping, horizontal flipping, and color jittering.
```
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
cfg.INPUT.RANDOM_FLIP = "horizontal"
cfg.INPUT.BRIGHTNESS = 0.2
cfg.INPUT.CONTRAST = 0.2
cfg.INPUT.SATURATION = 0.2
cfg.INPUT.HUE = 0.1
```

#### 4. Model Evaluation
- Use a validation set to monitor overfitting and adjust hyperparameters accordingly.
```
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.TEST.EVAL_PERIOD = 5000
```

#### Example Configuration
```
import os
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks, default_setup, launch
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.utils import comm

# Setup logger
setup_logger()

# Register dataset
register_coco_instances("my_dataset_train", {}, "path/to/train_annotations.json", "path/to/train_images")
register_coco_instances("my_dataset_val", {}, "path/to/val_annotations.json", "path/to/val_images")

# Configuration
cfg = get_cfg()
cfg.merge_from_file(f"detectron2/configs/COCO-InstanceSegmentation/{ARCHITECTURE}.yaml")

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100000
cfg.SOLVER.STEPS = (30000, 60000)
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.SOLVER.AMP.ENABLED = True
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # Change this based on your dataset
cfg.TEST.EVAL_PERIOD = 5000
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
cfg.INPUT.RANDOM_FLIP = "horizontal"

cfg.OUTPUT_DIR = "./output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Trainer class
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_hooks(cls):
        hooks = super().build_hooks()
        hooks.insert(-1, hooks.LRScheduler(cfg, warmup=True))
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        return evaluator

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

```