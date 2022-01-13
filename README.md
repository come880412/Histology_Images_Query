# Histology_Images_Query (Pytorch)
This repo is for the midtern project of the course Deep Learning in Medical Imaging (DLMI). Please refer to the technical document if you are interested in this task.

# Task description
In this final project, our task is histology images query. Given a histology image pair, determine whether they belong to the same category. For example, the query images are shown below, we need to dicide whether the two images are as the same category.
<p align="center">
<img src="https://github.com/come880412/Histology_Images_Query/blob/main/images/query_images/00a34fd0687e.png" width=20% height=20%>
<img src="https://github.com/come880412/Histology_Images_Query/blob/main/images/query_images/00b214d3c54c.png" width=20% height=20%>
</p>

# Method
Since we don’t have the label of each image, we solve this task by self-supervised learning. We employ the BYOL [1] to learn the feature representations. Such representations are used to compare the two query images by using the mean square error (MSE) metric. The model architecture is illustrated below.
<p align="center">
<img src="https://github.com/come880412/Histology_Images_Query/blob/main/images/model.jpg" width=50% height=50%>
</p>

# Experiments
- The metric for measuring performance is c-index \
 _**c-index = correct predcitions / # of query pairs**_

- The experiment of different threshold is illustrated below:

| Models | Threshold | Public(%) | Private(%) |
|:----------:|:----------:|:----------:|:----------:|
| ResNext50  | 0.95 | 82.99 | 83.03 |
| ResNext50 | 0.8 | 86.54 | 85.98 |
| ResNext50| 0.72 | 86.93 | 86.82 |
| ResNeXt101 | 0.65 | 86.93 | 87.19 |
| ResNeXt101 | 0.60 | 86.16 | 86.34 |
| ResNeXt101 | 0.63 | 86.61 | 86.87 |
| ResNeXt101 | 0.68 | 87.18 | 87.54 |
| ResNeXt101 | _**0.705**_ | _**87.21**_ | _**87.58**_
Where the threshold is used to determine whether the pair query images are in the same category after computing the mean square error between the pair query images. 

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

# References
[1] Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., ... & Valko, M. (2020). Bootstrap your own latent: A new approach to self-supervised learning. arXiv preprint arXiv:2006.07733.
