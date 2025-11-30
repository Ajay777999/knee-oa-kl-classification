from args import get_args
import os
import pandas as pd
import torch
from dataset import Knee_Xray_dataset
from torch.utils.data import DataLoader
from models import MyModel
from trainer import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args= get_args()
    
    for fold in range(5):
        print('Training fold: ', fold)

    train_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_train.csv'.format(str(fold))))
    val_set = pd.read_csv(os.path.join(args.csv_dir, 'fold_{}_val.csv'.format(str(fold))))
  

  #3. preparing Datasets

    train_dataset = Knee_Xray_dataset(train_set)
    val_dataset = Knee_Xray_dataset(val_set)

    # 4. create data loader

    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=0, pin_memory =torch.cuda.is_available())
    val_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=False,num_workers=0, pin_memory =torch.cuda.is_available())

# 5 initializing the model
    
    model = MyModel(args.backbone)


    # 6 Train the model 

    train_model(model, train_loader, val_loader)


    print()

     
    