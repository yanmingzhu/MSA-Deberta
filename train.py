# Training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm, trange

def train(model, train_loader, dev_loader, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    MAX_EPOCH = 3
    max_iteration = len(train_loader) * MAX_EPOCH
    print(len(train_loader))

    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iteration)

    batch_num = 0
    running_loss = 0
    for epoch in range(MAX_EPOCH):
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

            eval_loss = 0
            if batch_num % 20 == 0:
                #print(f"epoch {epoch}, batch {batch_num}")
                eval_loss = evaluation(model, dev_loader)
                print(f"loss = {loss}, accu loss = {running_loss/batch_num} eval loss = {eval_loss}")
                print(f"learning rate {scheduler.get_lr()}")


            loss.backward()
            optimizer.step()
            scheduler.step()

def evaluation(model, dataloader):
  model.eval()

  all_loss = []
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
  return torch.tensor(all_loss).mean()