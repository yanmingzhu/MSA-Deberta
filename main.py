import torch
import argparse
from types import SimpleNamespace
from train import train
from dataset import get_dataloaders
import pickle
from LateFusion import LateDebertaMM
from EarlyFusion import EarlyDebertaMM
from evaluation import eval_model
from TextDeberta import TextDebertaMM
from GatedFusion import GatedDebertaMM
from CrossModal import CrossModalDebertaMM
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fusion", 
                        type=str, 
                        default="late",
                        choices=["early", "late", "gated"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--modal", type=str, default="av", choices=["a", "v", "av", "none"])
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--data_file", type=str, default="data/mosei.pkl")
    parser.add_argument("--save_path", type=str, default="models/best_model.pth")
    parser.add_argument("--lr", type=float, help="learning rate",
                        default=1e-5)

    args = parser.parse_args()
    return args

def get_model(fusion):
    if fusion == 'early':
        return EarlyDebertaMM()
    elif fusion == 'gated':
        return GatedDebertaMM()
    elif fusion == 'cross':
        return CrossModalDebertaMM()
    else:
        return LateDebertaMM(modal=args.modal)

if __name__ == "__main__":
    args = get_args()
    #seed_everything(args.seed)
    #torch.autograd.set_detect_anomaly(True)

    if not os.path.exists('models'):
        os.mkdir('models')

    with open(args.data_file, "rb") as handle:
        data = pickle.load(handle)
    train_loader, dev_loader, test_loader = get_dataloaders(data, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args.fusion)
    print(f"Using fusion {type(model).__name__}")

    train(model, train_loader, dev_loader, max_epochs=args.epochs, lr=args.lr, save_path=args.save_path, device=device)

    print("evaluating the last model")
    eval_model(model=model, test_loader=test_loader, device=device)

    best_model = get_model(args.fusion)
    best_model.load_state_dict(torch.load(args.save_path))
    print("evaluating the best model")
    eval_model(model=best_model, test_loader=test_loader, device=device)

