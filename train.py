import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import RandomAffine

from model import CNN


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on MNIST.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    dataset = MNIST(args.data_root, train=True, download=True)
    images = dataset.data.unsqueeze(1).float() / 255
    labels = dataset.targets
    indices = torch.randperm(len(images))
    val_indices, train_indices = indices[:5000], indices[5000:]
    train_images, train_labels = images[train_indices], labels[train_indices]
    val_images, val_labels = images[val_indices], labels[val_indices]

    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    augment = RandomAffine(degrees=10, translate=(0.08, 0.08))
    Path("checkpoints").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)
    best_correct = -1
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch_indices in torch.randperm(len(train_images)).split(args.batch_size):
            batch = augment(train_images[batch_indices].to(device))
            targets = train_labels[batch_indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        model.eval()
        correct = 0
        with torch.no_grad():
            for batch, targets in zip(
                val_images.split(args.batch_size),
                val_labels.split(args.batch_size),
            ):
                predictions = model(batch.to(device)).argmax(1).cpu()
                correct += (predictions == targets).sum().item()
        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(train_images),
            "validation_correct": correct,
            "validation_accuracy": correct / len(val_images),
        })
        if correct > best_correct:
            best_correct = correct
            torch.save(
                {
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "seed": 42,
                    "training_examples": len(train_images),
                    "validation_examples": len(val_images),
                    "validation_accuracy": correct / len(val_images),
                },
                "checkpoints/cnn.pt",
            )
        print(
            f"Epoch {epoch}: loss {history[-1]['train_loss']:.4f}, "
            f"validation {100 * correct / len(val_images):.2f}%",
            flush=True,
        )

    with Path("results/training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0])
        writer.writeheader()
        writer.writerows(history)


if __name__ == "__main__":
    main()
