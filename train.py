import torch
from byol_pytorch.byol_pytorch import BYOL
from torch.nn.modules.batchnorm import BatchNorm1d
from torchvision import models
from dataset import Histology_data, query_dataset
import argparse
from torch.utils.data import DataLoader
import tqdm
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def get_feature(model, img):
    for index, layer in enumerate(model.children()):
        if index <= 8:
            img = layer(img).squeeze()

    return img.squeeze()

def SSL(args):
    train_data = Histology_data(args.data)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

    val_data = query_dataset(args.data, 256, 'val')
    val_loader = DataLoader(val_data, batch_size=32, num_workers=args.n_cpu, shuffle=False)

    if args.model == 'resnet18':
        model = models.resnet18(pretrained=True).cuda()
    elif args.model == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True).cuda()
    elif args.model == 'resnext101':
        model = models.resnext101_32x8d(pretrained=True).cuda()
    print('Using the model of %s' % (args.model))

    max_acc = 0
    best_threshold = 0
    if args.load:
        print('Load pretrained model!')
        model.load_state_dict(torch.load(args.load))

    learner = BYOL(
        model,
        image_size = args.image_size,
        hidden_layer = 'avgpool',
        projection_size = 512,           # the projection size
        projection_hidden_size = 2048,   # the hidden dimension of the MLP for both the projection and prediction
        moving_average_decay = 0.996      # the moving average decay factor for the target encoder, already set at what paper recommends
    )
    if args.opt == 'adam':
        opt = torch.optim.Adam(learner.parameters(), lr=args.lr, weight_decay=4e-4)
    elif args.opt == 'sgd':
        opt = torch.optim.SGD(learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=1.5e-6)

    print('Start training!')
    for epoch in range(args.n_epochs):
        total_loss = 0.
        model.train()
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for x1, x2 in train_loader:
            x1, x2 = x1.cuda(), x2.cuda()
            loss = learner(x1, x2)
            opt.zero_grad()
            loss.backward()
            total_loss += loss.item()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
            )
        pbar.close()
        threshold, val_acc = validation(args, model, val_loader, epoch)
        if val_acc > max_acc:
            max_acc = val_acc
            best_threshold = threshold                                                                                                  
            # save your improved networkW
            torch.save(model.state_dict(), '%s/model_epoch%d_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, epoch, threshold, val_acc))

    torch.save(model.state_dict(), '%s/model_final_threshold_%.2f_acc%.2f.pth' % (args.save_model_path, threshold, val_acc))
    print('Best threshold: %f \t max_acc:%.2f%%' % (best_threshold, max_acc))

def validation(args, model, val_loader, epoch):
    model.eval()
    loss_list = []
    label_list = []

    max_acc = 0.
    best_threshold = 0.
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="Valid[%d/%d]"%(epoch, args.n_epochs), unit=" step")
        for image1, image2, label in val_loader:
            image1, image2 = image1.cuda(), image2.cuda()
            feature1 = get_feature(model, image1)
            feature2 = get_feature(model, image2)                                               

            loss = loss_fn(feature1, feature2).cpu().detach().numpy()
            for i, dis in enumerate(loss):
                loss_list.append(dis)
                label_list.append(label[i])
            pbar.update()

        label_list = np.array(label_list)
        num_label = len(label_list)
        loss_list = np.array(loss_list)
        loss_th_list = np.arange(0 ,2 , 0.01)

        # Threshold search
        for loss_th in loss_th_list:
            pred = [1 if loss <= loss_th else 0 for loss in loss_list]
            pred = np.array(pred)
            correct = np.equal(pred, label_list).sum()
            acc = round((correct / num_label) * 100, 2)

            if acc > max_acc:
                max_acc = acc
                best_threshold = loss_th
        pbar.set_postfix(
        val_acc = f"{max_acc:.2f}%",
        threshold = f"{best_threshold}"
        )
        pbar.close()
        return best_threshold, max_acc

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--n_epochs', default=30, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch_size for training')
    parser.add_argument('--image_size', default=224, type=int, help='size of image')
    parser.add_argument('--n_cpu', default=4, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.0003, type=float, help='Learning rate of the adam')
    parser.add_argument('--load', type=str, default='', help="Model checkpoint path")
    parser.add_argument('--data', default='../dataset/', type=str, help="Training images directory")
    parser.add_argument('--model', default='resnext101', type=str, help="feature backbone(resnet18/resnext50/resnext101)")
    parser.add_argument('--opt', default='sgd', type=str, help="adam/sgd")
    parser.add_argument('--save_model_path', default='./checkpoints/resnext101', type=str, help="Path to save model")
    args = parser.parse_args()
    os.makedirs(args.save_model_path, exist_ok=True)

    SSL(args)

    

    