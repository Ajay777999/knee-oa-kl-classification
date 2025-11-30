from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score
from utils import save_checkpoint
import torch.nn.functional as F
import numpy as np

def train_model(model, train_loader, val_loader, cur_fold):
    args = get_args()
     
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameter(), lr=args.lr)

    best_val_metric = None
    best_model_path = None

    for epoch in range (args.epochs):
        model.train()

        #for batch_inx, (data, target) in enumerate(train_loader):
        for batch in train_loader():
            inputs = batch['img']
            targets = batch['label']

            #reseting the gradients

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        print('Epoch--{}:{}'.format((epoch +1), training_loss / len(train_loader)))
        ba, y_true, y_pred = validate_model(model, val_loader, criterion)

        best_ba, best_model_path = save_checkpoint(cur_fold,
                        epoch,
                        model,
                        y_true,
                        y_pred,
                        ba,
                        best_val_metric = ba,
                        prev_model_path = best_model_path,
                        comparator='gt',
                        save_dir=args.out_dir)

        val_loss = validate_model(model, val_loader, criterion) 
  
        print('Validation loss:{}'.format(val_loss))  


        def validate_model(model, val_loader, criterion):
            model.eval() 
            val_loss = 0

            all_preds = []
            all_targets = []
        
        for batch in val_loader():
            inputs = batch['img']
            targets = batch['label']

            #reseting the gradients

            outputs = model(inputs)
            loss = criterion(outputs, targets)
    

            val_loss += loss.item()

        
        predictions = F.softmax(outputs, dim=1)
        pred_targets = predictions.max(dim=1)[1]

        all_preds.append(pred_targets.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        ba = balanced_accuracy_score(all_preds, all_targets)

        return ba, all_targets, all_preds
        return val_loss / len(val_loader)

    

