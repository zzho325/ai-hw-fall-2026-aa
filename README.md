# Adversarial Attacks on MNIST

I trained a CNN to recognize handwritten digits and tested FGSM, PGD, and Momentum I-FGSM
against the same saved model.

## Results

Before attacks, the CNN correctly classified 9,860 of the 10,000 test images: 98.60% accuracy.

| Attack | Epsilon | Accuracy after attack | Attack success rate |
|---|---:|---:|---:|
| FGSM | 0.1 | 72.86% | 26.11% |
| FGSM | 0.2 | 8.87% | 91.00% |
| FGSM | 0.3 | 1.01% | 98.98% |
| PGD | 0.1 | 59.22% | 39.94% |
| PGD | 0.2 | 0.10% | 99.90% |
| PGD | 0.3 | 0.00% | 100.00% |
| MI-FGSM | 0.1 | 55.70% | 43.51% |
| MI-FGSM | 0.2 | 0.32% | 99.68% |
| MI-FGSM | 0.3 | 0.00% | 100.00% |

In this run, both iterative attacks had higher success rates than FGSM at every epsilon.
At 0.1, MI-FGSM fooled 43.51% of initially correct images, compared with 26.11% for FGSM.

Accuracy uses all 10,000 test images. Attack success rate (ASR) is the number of initially
correct images made incorrect by the attack, divided by the 9,860 initially correct images.
These are untargeted attacks: any incorrect digit counts as a successful attack.

The exact counts and settings are in [test_results.csv](results/test_results.csv).

## Setup

MNIST has 60,000 official training images and 10,000 test images. I split the training set
into 55,000 training images and 5,000 validation images using seed 42. The CNN has two
convolution and pooling blocks with 16 and 32 channels, followed by a 64-unit hidden layer
and ten output logits.

Training used Adam with learning rate 0.001, batch size 256, and eight epochs. Training batches
received random rotations within 10 degrees and translations within 8% of image size. I kept
the checkpoint with the highest validation accuracy, which was epoch 7 at 98.38%.
The epoch results are in [training.csv](results/training.csv).

Pixels are scaled to [0, 1]. Epsilon bounds the maximum absolute change to any pixel, and
attacked images are clipped to [0, 1]. FGSM takes one gradient-sign step. PGD uses one random
start and 20 projected steps. MI-FGSM uses 20 steps with per-image L1 gradient normalization
and momentum decay 1.0. Both iterative attacks use a step size of epsilon / 20.

## Run

Use Python 3.12 and run these commands from the repository directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python test.py
```

MNIST downloads into `data/`. The saved checkpoint is included, so `python test.py` also works
on its own after installing the dependencies. CPU is the default device.
