import argparse
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def loader(train, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )
    data = datasets.CIFAR10("data", train=train, download=True, transform=transform)
    return DataLoader(data, batch_size=batch_size, shuffle=train)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    loss_fn = nn.CrossEntropyLoss()

    if args.mode == "train":
        train_data = loader(True, args.batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        model.train()
        for epoch in range(args.epochs):
            total_loss = 0.0
            for images, labels in train_data:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"epoch {epoch + 1}/{args.epochs} loss {total_loss / len(train_data):.4f}")
        torch.save(model.state_dict(), args.model)
        return

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"model not found: {args.model}; run with --mode train first")
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    test_data = loader(False, args.batch_size)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_data:
            predictions = model(images.to(device)).argmax(1).cpu()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    print(f"accuracy {correct / total:.2%}")


parser = argparse.ArgumentParser(description="Minimal CIFAR-10 trainer and tester")
parser.add_argument("--mode", choices=("train", "test"), default="train")
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--model", default="cifar10.pt")
run(parser.parse_args())
