# Confusion Matrix for Object Detection

[![DOI](https://zenodo.org/badge/249415928.svg)](https://zenodo.org/badge/latestdoi/249415928)

The ConfusionMatrix class can be used to generate confusion matrix for the object detection task.

## Usage

In the test code, you need to declare the ConfusionMatrix class with the appropriate parameters.

`conf_mat = ConfusionMatrix(num_classes = 3, CONF_THRESHOLD = 0.3, IOU_THRESHOLD = 0.5)`

The class has a function called, process_batch, you can use it update the confusion matrix. 

An example usage can be:

`conf_mat.process_batch(preds, gt_boxes)` where preds are predictions made by model, [N, 6] x1, y1, x2, y2, confidence, class and gt_boxes are the ground truth labels, [M, 4] x1, y1, x2, y2, class.



## References

This repository uses a function from Pytorch repository, https://github.com/pytorch/vision/blob/ae228fef1ce176fa3e3949f5db7b6e87a1e33065/torchvision/ops/boxes.py#L138

and

The code is very similar to the repository below and main contribution of this repository is ConfusionMatrix class can be used with all deep learning frameworks.

https://github.com/svpino/tf_object_detection_cm

## Citations

If you would like to cite the concept, please use this doi: DOI: 10.5281/zenodo.3724203
