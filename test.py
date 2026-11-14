import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST

from model import CNN


def main():
    parser = argparse.ArgumentParser(description="Test three adversarial attacks on MNIST.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    checkpoint = torch.load("checkpoints/cnn.pt", map_location="cpu", weights_only=True)
    model = CNN().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.requires_grad_(False)
    dataset = MNIST(args.data_root, train=False, download=True)
    images = dataset.data.unsqueeze(1).float() / 255
    labels = dataset.targets
    clean_predictions = []
    with torch.no_grad():
        for batch in images.split(args.batch_size):
            clean_predictions.append(model(batch.to(device)).argmax(1).cpu())
    clean_correct = torch.cat(clean_predictions).eq(labels)
    clean_count = clean_correct.sum().item()
    print(f"Clean accuracy: {clean_count}/{len(labels)} ({100 * clean_count / len(labels):.2f}%)")

    results = []
    for name, attack in [("FGSM", fgsm), ("PGD", pgd), ("MI-FGSM", mi_fgsm)]:
        for epsilon in [0.1, 0.2, 0.3]:
            torch.manual_seed(42)
            correct = 0
            successes = 0
            max_linf = 0
            for start in range(0, len(images), args.batch_size):
                end = start + args.batch_size
                batch = images[start:end].to(device)
                targets = labels[start:end].to(device)
                adversarial = attack(model, batch, targets, epsilon)
                with torch.no_grad():
                    predictions = model(adversarial).argmax(1).cpu()
                correct += predictions.eq(labels[start:end]).sum().item()
                successes += (
                    clean_correct[start:end] & predictions.ne(labels[start:end])
                ).sum().item()
                max_linf = max(max_linf, (adversarial - batch).abs().max().item())
                assert adversarial.min() >= 0 and adversarial.max() <= 1
                assert max_linf <= epsilon + 1e-6

            row = {
                "attack": name,
                "epsilon": epsilon,
                "steps": 1 if name == "FGSM" else 20,
                "step_size": epsilon if name == "FGSM" else epsilon / 20,
                "test_examples": len(labels),
                "clean_correct": clean_count,
                "clean_accuracy": clean_count / len(labels),
                "adversarial_correct": correct,
                "adversarial_accuracy": correct / len(labels),
                "successful_attacks": successes,
                "attack_success_rate": successes / clean_count,
                "max_linf": max_linf,
            }
            results.append(row)
            Path("results").mkdir(exist_ok=True)
            with Path("results/test_results.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=row)
                writer.writeheader()
                writer.writerows(results)
            print(
                f"{name}, epsilon={epsilon}: accuracy {100 * correct / len(labels):.2f}%, "
                f"ASR {100 * successes / clean_count:.2f}% ({successes}/{clean_count})",
                flush=True,
            )


def fgsm(model, images, labels, epsilon):
    adversarial = images.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(adversarial), labels)
    gradient = torch.autograd.grad(loss, adversarial)[0]
    return (images + epsilon * gradient.sign()).clamp(0, 1).detach()


def pgd(model, images, labels, epsilon, steps=20):
    adversarial = (images + torch.empty_like(images).uniform_(-epsilon, epsilon)).clamp(0, 1)
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        adversarial = adversarial.detach() + (epsilon / steps) * gradient.sign()
        perturbation = (adversarial - images).clamp(-epsilon, epsilon)
        adversarial = (images + perturbation).clamp(0, 1)
    return adversarial.detach()


def mi_fgsm(model, images, labels, epsilon, steps=20, decay=1.0):
    adversarial = images.detach().clone()
    momentum = torch.zeros_like(images)
    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]
        norm = gradient.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        momentum = decay * momentum + gradient / norm
        adversarial = adversarial.detach() + (epsilon / steps) * momentum.sign()
        perturbation = (adversarial - images).clamp(-epsilon, epsilon)
        adversarial = (images + perturbation).clamp(0, 1)
    return adversarial.detach()


if __name__ == "__main__":
    main()
