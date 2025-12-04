import torch
import argparse
from types import SimpleNamespace
from train import train
from dataset import get_dataloaders
import pickle
from LateFusion import LateDebertaMM

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--model", type=str, default="late-fusion")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--data_file", type=str, default="data/mosei.pkl")
    parser.add_argument("--lr", type=float, help="learning rate",
                        default=1e-5)

    args = parser.parse_args()
    return args

def get_model(model_name):
    return LateDebertaMM()

if __name__ == "__main__":
    args = get_args()
    #seed_everything(args.seed)
    #torch.autograd.set_detect_anomaly(True)

    with open(args.data_file, "rb") as handle:
        data = pickle.load(handle)
    train_loader, dev_loader, test_loader = get_dataloaders(data, args.batch_size)

    model = get_model(args.model)

    train(model, train_loader, dev_loader, epochs=args.epochs, lr=args.lr)

