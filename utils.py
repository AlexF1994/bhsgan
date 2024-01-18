import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def get_noise(n_samples, z_dim, device="cpu"):
    return torch.randn(n_samples, z_dim, device=device)


def init_weights(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
    if isinstance(layer, nn.BatchNorm2d):
        torch.nn.init.normal_(layer.weight, 0.0, 0.02)
        torch.nn.init.constant_(layer.bias, 0)


def plot_tensor_images(
    image_tensor, num_images=25, size=(1, 28, 28), save_fig=False, epoch=0
):
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.axis("off")
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if save_fig:
        plt.savefig("image_at_epoch_{:04d}.png".format(epoch))

    plt.show()


def plot_losses(generator_losses, discriminator_losses):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="G-Loss")
    plt.plot(discriminator_losses, label="D-Loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_models_state_dict(trained_model, path):
    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True)

    dis_path = path / "discriminator.pt"
    gen_path = path / "generator.pt"
    torch.save(trained_model.discriminator.state_dict(), dis_path)
    torch.save(trained_model.generator.state_dict(), gen_path)


def load_model_state_dict(model_instance, path):
    return model_instance.load_state_dict(torch.load(path))


class ActivationFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.config = {"name": self.name}


class Positive(ActivationFunction):
    def forward(self, x):
        return torch.abs(x)


class TanhScale(ActivationFunction):
    def forward(self, x):
        return 4 * torch.tanh(x)


class RevKlActivation(ActivationFunction):
    def forward(self, x):
        return -torch.abs(x)


class GanGanActivation(ActivationFunction):
    def forward(self, x):
        return -torch.log(1 + torch.exp(-x))
