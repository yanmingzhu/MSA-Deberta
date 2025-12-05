# Training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange
import os
from util import *

def train(model, train_loader, dev_loader, max_epochs, lr, save_path, device):
    model.to(device)
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss()

    max_iteration = len(train_loader) * max_epochs
    print(len(train_loader))

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iteration)

    batch_num = 0
    running_loss = 0
    best_acc = 0

    for epoch in range(max_epochs):
        for data_batch in (tqdm(train_loader, desc='iteration')):
            model.train()

            '''
            print(data_batch[0].shape)
            print(data_batch[1].shape)
            print(data_batch[2].shape)
            print(data_batch[3].shape)
            '''

            batch_num += 1
            text = data_batch[0].to(device)
            vision = data_batch[1].to(device)
            audio = data_batch[2].to(device)
            label = data_batch[3].to(device)
            masks = data_batch[4].to(device)


            pred = model(text, audio, vision, masks)
            pred = pred.squeeze()
            label = label.squeeze()

            optimizer.zero_grad()

            loss = criterion(pred, label)
            running_loss += loss.item()

            if batch_num % 50 == 0:
                #print(f"epoch {epoch}, batch {batch_num}")
                eval_loss, acc, _, _, _ = evaluation(model=model, 
                                                    dataloader=dev_loader, 
                                                    criterion=criterion, 
                                                    device=device)
                print(f"loss = {loss:.4f}, running loss = {running_loss/batch_num:.4f} eval loss = {eval_loss:.4f}, eval acc = {acc}")
                #print(f"learning rate {scheduler.get_lr()}")
                if best_acc < acc:
                    best_acc = acc
                    torch.save(model.state_dict(), save_path)
                    if os.path.isfile(save_path):
                        print(f"Successfully saved model to: {os.path.abspath(save_path)}")
                    else:
                        print(f"Error: File was not saved to {os.path.abspath(save_path)}")
                else:
                   print(f"model didn't improve, skip saving.")

            loss.backward()
            optimizer.step()
            scheduler.step()

def evaluation(model, dataloader, criterion, device):
    model.eval()

    all_loss = []
    all_pred = np.array([])
    all_label = np.array([])
    with torch.no_grad():
        for batch in dataloader:
            text = batch[0].to(device)
            vision = batch[1].to(device)
            audio = batch[2].to(device)
            label = batch[3].to(device)
            masks = batch[4].to(device)

            pred = model(text, audio, vision, masks).squeeze()
            pred = pred.squeeze()
            label = label.squeeze()
            all_loss.append(criterion(pred, label))
            all_pred = np.append(all_pred, pred.cpu().detach().numpy())
            all_label = np.append(all_label, label.cpu().detach().numpy())
    acc, f1, mae, corr = metrics(all_pred, all_label)
    return torch.tensor(all_loss).mean(), acc, f1, mae, corr