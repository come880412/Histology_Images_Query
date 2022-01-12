# Histology_Images_Query (Pytorch)
This repo is for the midtern project of the course Deep Learning in Medical Imaging (DLMI). Please refer to the technical document if you are interested in this task.

# Task description
In this final project, our task is histology images query. Given a histology image pair, determine whether they belong to the same category. For example, the query images are shown below, we need to dicide whether the two images are as the same category.
<p align="center">
<img src="https://github.com/come880412/Histology_Images_Query/blob/main/images/query_images/00a34fd0687e.png" width=20% height=20%>
<img src="https://github.com/come880412/Histology_Images_Query/blob/main/images/query_images/00b214d3c54c.png" width=20% height=20%>
</p>

# Method
Since we don’t have the label of each image, we solve this task by self-supervised learning. We employ the BYOL to learn the feature representations. Such representations are used to compare the two query images by using the cosine similarity metric. The model architecture is illustrated below.
<p align="center">
<img src="https://github.com/come880412/Histology_Images_Query/blob/main/images/model.jpg" width=50% height=50%>
</p>

# Experiments

# Environment
OS : Ubuntu 16.04 \
Language: Python37

# How to use
### Download the pretrained models
```bash
$ bash download.sh
```

### Training from scratch
```bash
$ python train.py --data path/to/train
```

### Testing
```bash
$ python inference.py --data path/to/test
```
