## 2. Early Stopping

### Manual Stopping
When using Detectron2's default trainer, you typically monitor several metrics to decide whether to stop the training process early. The choice of metrics depends on your specific task and objectives. However, some common metrics for various computer vision tasks include:

1. **Validation Loss**: The loss calculated on a separate validation set. Decreasing validation loss indicates improving model performance.

2. **Mean Average Precision (mAP)**: Particularly relevant for object detection and instance segmentation tasks. mAP combines precision and recall across different object categories.

3. **Accuracy**: The proportion of correctly classified instances, often used in classification tasks.

4. **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth bounding boxes or segmentation masks. Higher IoU values indicate better object localization.

5. **F1 Score**: Harmonic mean of precision and recall. Useful for imbalanced datasets.

6. **Precision and Recall**: Precision measures the proportion of true positive predictions among all positive predictions, while recall measures the proportion of true positive predictions among all actual positives.

7. **Mean IoU (mIoU)**: Similar to mAP but for semantic segmentation tasks. It calculates the average IoU across all classes.

When deciding whether to stop training early, you typically monitor these metrics on a validation set at regular intervals (e.g., after every epoch or a certain number of iterations). If the metrics consistently improve or plateau, training continues. However, if the metrics degrade or show no significant improvement for a predefined number of iterations, you might stop training early to prevent overfitting or wasting computational resources.

For example, you could stop training if the **validation loss** hasn't improved for several epochs or if the **mAP** or **accuracy** on the validation set starts to decrease.


### Code Implementation
Implementing early stopping in a Detectron2 training script involves monitoring certain metrics during training and stopping the process if those metrics do not improve or meet certain criteria over a specified number of iterations. Here's a general outline of how you can implement early stopping:

- **Choose Metrics**: Decide which metrics to monitor for early stopping. Common metrics include validation loss, validation accuracy, or any other relevant evaluation metric for your task.

- **Define Stopping Criteria**: Specify the conditions under which training should stop. For example, you might stop training if the validation loss hasn't improved for a certain number of epochs or iterations.

- **Monitoring Loop**: Periodically evaluate the chosen metrics on a validation set during training to monitor their progress.

- **Stop Training**: If the stopping criteria are met, stop the training process.


#### Example Configuration
```
import os
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, hooks, default_setup, launch
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances

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

# Trainer class with early stopping
class MyTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_val_loss = float('inf')
        self.patience = 5  # Number of epochs without improvement to wait before stopping
        self.counter = 0

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, hooks.LRScheduler(cfg, warmup=True))
        return hooks

    def build_evaluator(self, cfg, dataset_name, output_folder=None):
        from detectron2.evaluation import COCOEvaluator, inference_on_dataset
        evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        return evaluator

    def after_step(self):
        if self.iter % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        # Implement checkpoint saving logic here
        pass

    def validate(self):
        metrics = super().validate()
        val_loss = metrics["total_loss"]
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Validation loss hasn't improved for {} epochs. Early stopping...".format(self.patience))
                self._save_checkpoint()
                self._terminate()

    def _terminate(self):
        raise KeyboardInterrupt

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
try:
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted.")
```

In this example:

- We define a **MyTrainer** class that inherits from **DefaultTrainer**;
- Inside **MyTrainer**, we override the **validate()** method to perform validation at regular intervals and check if the validation loss improves;
- If the validation loss does not improve for a certain number of epochs **(self.patience)**, we stop training by raising a **KeyboardInterrupt**.

You can customize the early stopping criteria and the logic for saving checkpoints according to your specific requirements. Additionally, you may want to implement logic to handle saving the best model weights based on validation performance.