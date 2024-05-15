# Questions regarding Detectron Training

1. What are the hyperparameters and how can I optimize training?

2. How can I stop Detectron training automatically without reaching MAX number of iterations? If not how should I proceed?

3. How to resume training if needed?


### 1. Hyperparameters

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

This specifies the number of regions of interest (ROIs) per image used during training. For object detection, it determines the number of samples drawn from each image to compute the loss.


```
cfg.MODEL.ROI_HEADS.NUM_CLASSES
```

Number of classes for the dataset you are training on. This should match the number of classes in your dataset. 


```
cfg.TEST.EVAL_PERIOD
```

This defines how often (in terms of iterations) the model will be evaluated on the validation set. Frequent evaluations provide more feedback during training but can also slow down the process.