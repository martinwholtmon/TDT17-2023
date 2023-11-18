# TDT17-2023
VI-tasks on Cityscapes

### Expected folder structure
```
data/cityscapes
├── gtFine
|   ├── test
|   ├── train
|   ├── val
├── leftImg8bit
|   ├── test
|   ├── train
|   ├── val
```

## Tasks
 - [x] Select a network structure
 - [x] Train a network for image segmentation on cityscapes
 - [x] Perform semantic segmentation (inference) on cityscapes
 - [ ] Run inference on 3 random images from Trondheim



## Sources
- [CityScapes](https://www.cityscapes-dataset.com/) - large-scale dataset that focus on semantic understanding of urban street scenes
- [DeepLabV3+](https://paperswithcode.com/model/deeplabv3-1?variant=deeplabv3-r101-dc5-1) - Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
- [Timm](https://github.com/huggingface/pytorch-image-models/tree/main) - PyTorch Image Models
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - Segmentation models with pretrained backbones
- [Pytorch Dataset](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
- [TorchMetrics](hhttps://lightning.ai/docs/torchmetrics/stable/) - PyTorch metrics implementations and an easy-to-use API to create custom metrics
  - [Accuracy](https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html)
  - [Jaccard Index](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html) - intersetion over union or jaccard similarity coefficient
  - [F-Beta Score](https://lightning.ai/docs/torchmetrics/stable/classification/fbeta_score.html)
- [PyTorch lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)