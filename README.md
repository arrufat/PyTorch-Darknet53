# Darknet53

This is implementation of Darknet53 network discussed in [ [1] ](https://pjreddie.com/media/files/papers/YOLOv3.pdf) used for feature extractor of YOLOv3.

This new network is more efficient than ResNet-101 or ResNet-152.

Here are some ImageNet results:

- Framework: Darknet [ [2] ](https://github.com/pjreddie/darknet)
- GPU: Titan X
- Input Shape(CWH): 3 x 256 x 256 

![darknet_table](https://user-images.githubusercontent.com/35001605/53488653-4b288280-3ad2-11e9-9aba-f14cbfc65c0c.PNG)


**Darknet-53 is better than ResNet-101 and 1.5× faster.**

**Darknet-53 has similar performance to ResNet-152 and is 2× faster [ [1] ](https://pjreddie.com/media/files/papers/YOLOv3.pdf).** 


**But when I trained and tested this model with 224x224 input image, I could not get the good results like the above table.**

**I got 75.xx% accuracy on Imagenet dataset for validation.**

[Pretrained model_weight_download](https://drive.google.com/open?id=1keZwVIfcWmxfTiswzOKUwkUz2xjvTvfm)


## Network Structure

![webp net-resizeimage](https://user-images.githubusercontent.com/35001605/53487913-2df2b480-3ad0-11e9-9788-b2feab624786.png)


## Training

- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use [the following script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
- imagenet data is processed [as described here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

```
python train.py --data [imagenet-folder with train and val folders] --gpu 0 -b 64
```

## Benchmark
- Framework: PyTorch 1.6.0
- GPU: NVIDIA Quadro RTX 5000 Mobile / Max-Q 16 GiB
- CPU: Intel® Xeon(R) E-2276M CPU @ 2.80GHz × 12
- RAM: 32 GiB
- Batch Size: 1
- Input Shape(CWH): 3 x 224 x 224 

**On GPU**
```
     model  | inference | VRAM
 -----------+-----------+---------
   resnet50 |  9.809 ms | 1081 MiB
  resnet101 | 18.497 ms | 1135 MiB
  resnet152 | 26.811 ms | 1207 MiB
densenet121 | 19.345 ms |  981 MiB
  darknet53 | 11.414 ms | 1001 MiB
```

**On CPU**
```
     model  |  inference | RAM
 -----------+------------+---------
   resnet50 |  49.412 ms | 206 MiB
  resnet101 |  74.925 ms | 279 MiB
  resnet152 | 100.670 ms | 332 MiB
densenet121 |  48.791 ms | 133 MiB
  darknet53 |  69.895 ms | 289 MiB
```

## Reference
>[ [1] YOLOv3: An Incremental Improvement ](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

>[ [2] darknet framework ](https://github.com/pjreddie/darknet)

>[ [3] ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet)
