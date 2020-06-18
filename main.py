import os 
import torch
import numpy 
import pandas as pd 
import torch.nn as nn
import pretrainedmodels
from torch.nn import functional as F
import albumentations 
from wtfml.data_loaders.image import ClassificationLoader

class SEResNext50_32x4d(nn.Module):
    def __init(self,pretrained="imagenet"):
        super(SEResNext50_32x4d,self).__init__()
        self.model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained=pretrained)
        self.out = nn.Linear(2048,1)
    
    def forward(self,image):
        bs,_,_,_ = image.shape 
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x,1)
        x = x.reshape(bs,-1)
        out = self.out(x)
        return out 
    
def train(fold):
    training_data_path = "/home/sushi/code/Kaggle/Melanoma-Detection-/input/jpeg/train224"
    df = pd.read_csv("/home/sushi/code/Kaggle/Melanoma-Detection-/input/train_folds.csv")
    device ="cuda"
    epochs=50
    train_bs = 32
    valid_bs=16

    df_train = df[df.kfold!=fold].reset_index(drop=True)
    df_valid = df[df.kfold==fold].reset_index(drop=True)

    train_aug = albumentations.Compose(
        albumentations.Normalize(always_apply=True)
    )

    valid_aug = albumentations.Compose(
        albumentations.Normalize(always_apply=True)
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path,i+".jpg") for i in train_images]
    train_targets = df_train.targets.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path,i+"jpg") for i in valid_images]
    valid_targets = df_valid.target.values

    train_dataset = ClassificationLoader(
        image_paths=train_images,
        train_targets,
        resize=None,
        augmentations=train_aug

    )

    valid_dataset = ClassificationLoader(
        image_paths=valid_images,
        train_targets,
        resize=None,
        augmentations=valid_aug

    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = train_bs,
        shuffle=False,
        num_workers=4

    )

    model = SEResNext50_32x4d(pretrained="imagenet")

    optimizer = torch.optim.Adam((model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max", 

    )






































    

    




        






