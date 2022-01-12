# Histology_Images_Query (Pytorch)
This repo is for the midtern project of the course Deep Learning in Medical Imaging (DLMI). Please refer to the technical document if you are interested in this task.

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
