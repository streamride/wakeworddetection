import numpy as np
import pandas as pd
import gc
from sklearn import metrics
from tqdm import tqdm
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from datasets import WakeWordDataset, get_loaders
from model import SimpleRNN, SimpleCNN



def train(model, train_loader, loss_fn, optimizer, scheduler, epoch, tensorboard_writer, device):
    epoch_train_loss = 0
    train_metrics = 0.0
    model.train()
    for x, y in tqdm(train_loader, desc='training'):
        x,y = x.to(device), y.to(device)
        x = x.squeeze(1)
        optimizer.zero_grad()
        
        predictions = model(x)
        # print(predictions)
        predictions = torch.sigmoid(predictions)
        # predictions = predictions.squeeze()
        # y = y.squeeze()
        loss = loss_fn(predictions, y)
        # predictions = torch.sigmoid(predictions)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        predictions = torch.round(torch.sigmoid(predictions))
        predictions = predictions.detach().cpu().numpy()
        
        y = y.detach().cpu().numpy()
        
        f_score = metrics.f1_score(y, predictions)
    
        train_metrics += f_score
        tensorboard_writer.add_scalar("LR/train", optimizer.param_groups[0]["lr"], epoch)
        tensorboard_writer.add_scalar("Loss/train", loss, epoch)
        tensorboard_writer.add_scalar("F1Score/train", f_score, epoch)
        epoch_train_loss += loss.item()
    tqdm.write(f'Epoch {epoch}')
    tqdm.write(f'Train Epoch loss {epoch_train_loss / len(train_loader)}')
    tqdm.write(f'Train f1 score {train_metrics / len(train_loader)}')
    return train_metrics / len(train_loader), epoch_train_loss / len(train_loader)
    
def evaluate(model, test_loader, loss_fn, epoch, tensorboard_writer, device):
    model.eval()
    test_metrics = 0.0
    epoch_test_loss = 0.0
    for x, y in tqdm(test_loader, desc='eval'):
        x,y = x.to(device), y.to(device)
        x = x.squeeze(1)
        predictions = model(x)
       
        # print(predictions)
        # predictions = predictions.squeeze()
        predictions = torch.sigmoid(predictions)
        loss = loss_fn(predictions, y)
        
        epoch_test_loss += loss.item()
        tensorboard_writer.add_scalar("Loss/eval", loss, epoch)
        predictions = torch.round(predictions)
        predictions = predictions.detach().cpu().numpy()
        
        y = y.detach().cpu().numpy()
        f_score = metrics.f1_score(y, predictions)
        tensorboard_writer.add_scalar("F1Score/eval", f_score, epoch)
        test_metrics += f_score
    tqdm.write(f'Epoch {epoch}')
    tqdm.write(f'Eval Epoch loss {epoch_test_loss / len(test_loader)}')
    tqdm.write(f'Eval f1 score {test_metrics / len(test_loader)}')
    return test_metrics / len(test_loader), epoch_test_loss / len(test_loader)


def run_training(model, train_loader, test_loader, loss_fn, optimizer, scheduler, early_stop_patience: int = None, EPOCHS=100):
    eval_f1_score = 0.0
    max_loss = 1000
    early_stopping = 0
    writer = SummaryWriter()
    
    device = torch.device('cuda')
    # model.to(device)
    for epoch in range(EPOCHS):
        train_f1, train_loss = train(model, train_loader, loss_fn, optimizer, scheduler, epoch, writer, device)
        
        test_f1, test_loss = evaluate(model, test_loader, loss_fn, epoch, writer, device)
        
        if test_f1 > eval_f1_score:
            eval_f1_score = test_f1
            torch.save(model, 'wake_model_cnn.pth')
        
        if test_loss < max_loss:
            max_loss = test_loss
            early_stopping = 0
        else:
            print('Early stopping', early_stopping)
            early_stopping += 1
        
        if early_stop_patience and early_stop_patience == early_stopping:
            print('Early stopping stopped')
            break
    return eval_f1_score


if __name__ == '__main__':
    torch.cuda.empty_cache()
    gc.collect()
    import time
    time.sleep(1.0)
    model = SimpleCNN()
    df = pd.read_csv('data/upsampled_data.csv')
    train_loader, test_loader = get_loaders(df, batch_size=64)
    device = torch.device('cuda')
    model.to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    run_training(model, train_loader, test_loader, loss_fn, optimizer, scheduler=scheduler, early_stop_patience=50, EPOCHS=30)