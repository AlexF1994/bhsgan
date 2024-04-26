from dataclasses import dataclass
from time import time
from typing import List

import torch
import torch.nn as nn

from fid import InceptionV3
from utils import get_noise, plot_tensor_images


@dataclass()
class TrainingParams:
    lr_dis: float
    lr_gen: float
    beta_1: float
    beta_2: float
    num_epochs: int
    num_dis_updates: int
    num_gen_updates: int
    batch_size: int
    weight_decay: float
    lr_annealing: bool


class Trainer:
    def __init__(
        self,
        training_params,
        generator,
        discriminator,
        device="cpu",
        calculate_fid=False,
    ):
        self.training_params = training_params
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.calculate_fid = calculate_fid
        if self.calculate_fid:
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            model = InceptionV3([block_idx])
            self.inception_model = model.to(device)
        self.discriminator_optimizer = self._init_dis_optimizer(training_params)
        self.generator_optimizer = self._init_gen_optimizer(training_params)

    def _init_dis_optimizer(self, training_params):
        lr = training_params.lr_dis
        beta_1 = training_params.beta_1
        beta_2 = training_params.beta_1
        weight_decay = training_params.weight_decay
        return torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta_1, beta_2),
            weight_decay=weight_decay,
        )

    def _init_gen_optimizer(self, training_params):
        lr = training_params.lr_gen
        beta_1 = training_params.beta_1
        return torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(beta_1, 0.9999)
        )

    def train_gan(
        self,
        dataloader,
        get_dis_loss,
        get_gen_loss,
        gradient_penalty_enabled,
        flatten_dim=None,
        is_color_picture=False,
        print_intermediate=True,
    ):
        num_epochs = self.training_params.num_epochs
        num_dis_updates = self.training_params.num_dis_updates
        num_gen_updates = self.training_params.num_gen_updates
        batch_size = self.training_params.batch_size
        noise_dim = self.generator.z_dim
        use_lr_annealing = self.training_params.lr_annealing

        generator_losses = []
        discriminator_losses = []
        dis_mean_losses = []
        gen_mean_losses = []
        total_steps = 0

        for epoch in range(num_epochs):
            print("Epoch " + str(epoch + 1) + " start training...", end="\n")
            current_step = 0
            start = time()
            for real_sample, _ in dataloader:
                if len(list(real_sample.size())) == 1:
                    real_sample = torch.reshape(real_sample, (batch_size, 1))
                # print(f"batch number: {batch}")
                if isinstance(real_sample, list):
                    real_sample = real_sample[0]
                if flatten_dim:
                    real_sample = real_sample.view(-1, flatten_dim)
                real_sample = real_sample.to(self.device)
                batch_size = len(real_sample)
                noise = get_noise(batch_size, noise_dim, device=self.device)
                if is_color_picture:
                    noise = torch.reshape(noise, (batch_size, noise_dim, 1, 1))

                mean_iteration_gen_loss = 0
                # noise = get_noise(batch_size, noise_dim, device=self.device)

                for _ in range(num_gen_updates):
                    ### Update generator ###
                    self.generator_optimizer.zero_grad()
                    fake_2 = self.generator(noise)
                    fake_score = self.discriminator(fake_2)

                    gen_loss = get_gen_loss(fake_score)
                    gen_loss.backward()

                    # Update the weights
                    # clip_grad_value_(self.generator.parameters(), 1000.0)
                    self.generator_optimizer.step()

                    # Keep track of the average generator loss
                    mean_iteration_gen_loss += gen_loss.item() / num_gen_updates

                generator_losses += [mean_iteration_gen_loss]

                mean_iteration_dis_loss = 0

                for _ in range(num_dis_updates):
                    ### Update discriminator ###
                    self.discriminator_optimizer.zero_grad()
                    fake_sample = self.generator(noise)
                    fake_score = self.discriminator(fake_sample.detach())
                    real_score = self.discriminator(real_sample)
                    # print(f"fake_score: {torch.mean(fake_score)}")
                    # print("=========================")
                    # print(f"real_score: {torch.mean(real_score)}")
                    if gradient_penalty_enabled:
                        if is_color_picture:
                            epsilon = torch.rand(
                                len(real_score),
                                1,
                                1,
                                1,
                                device=self.device,
                                requires_grad=True,
                            )
                        else:
                            epsilon = torch.rand(
                                len(real_score),
                                1,
                                device=self.device,
                                requires_grad=True,
                            )
                        gradient = get_gradient(
                            self.discriminator,
                            real_sample,
                            fake_sample.detach(),
                            epsilon,
                            self.device,
                        )
                        gradient_penalty = get_gradient_penalty(gradient)
                        discriminator_loss = get_dis_loss(
                            real_score, fake_score, gradient_penalty
                        )
                    else:
                        discriminator_loss = get_dis_loss(real_score, fake_score)

                    # Keep track of the average discriminator loss in this batch
                    mean_iteration_dis_loss += (
                        discriminator_loss.item() / num_dis_updates
                    )
                    # Update gradients
                    discriminator_loss.backward(retain_graph=True)
                    # print(torch.norm(self.discriminator.main[3][0].weight.grad))
                    # Update optimizer
                    # clip_grad_value_(self.discriminator.parameters(), 1000.0)
                    self.discriminator_optimizer.step()
                discriminator_losses += [mean_iteration_dis_loss]

                current_step += 1
                total_steps += 1

                print_val = f"Epoch: {epoch + 1}/{num_epochs} Steps:{current_step}/{len(dataloader)}\t"
                print_val += f"Epoch_Run_Time: {(time() - start):.6f}\t"
                print_val += f"Loss_C : {mean_iteration_dis_loss:.6f}\t"
                print_val += f"Loss_G : {mean_iteration_gen_loss :.6f}\t"
                print(print_val, end="\r", flush=True)
                # free up gpu disk space
                # del real_sample
                # torch.cuda.empty_cache()

            gen_loss_mean = sum(generator_losses[-current_step:]) / current_step
            dis_loss_mean = sum(discriminator_losses[-current_step:]) / current_step

            dis_mean_losses.append(dis_loss_mean)
            gen_mean_losses.append(gen_loss_mean)

            print_val = f"Epoch: {epoch + 1}/{num_epochs} Total Steps:{total_steps}\n"
            print_val += f"Total_Time : {(time() - start):.6f}\n"
            print_val += f"Loss_C : {mean_iteration_dis_loss:.6f}\n"
            print_val += f"Loss_G : {mean_iteration_gen_loss:.6f}\n"
            print_val += f"Loss_C_Mean : {dis_loss_mean:.6f}\n"
            print_val += f"Loss_G_Mean : {gen_loss_mean:.6f}\n"
            print(print_val)
            print("----------------------------------------------\n")

            current_step = 0

            if print_intermediate and (((epoch + 1) % 1) == 0):
                test_noise = get_noise(5, noise_dim, device=self.device)
                if is_color_picture:
                    test_noise = torch.reshape(test_noise, (5, noise_dim, 1, 1))
                test_images = self.generator(test_noise)
                plot_tensor_images(
                    test_images, num_images=5, unflat=False, tanh_activation=True
                )

            if use_lr_annealing:
                if (epoch + 1) == 20:
                    self.discriminator_optimizer.param_groups[0]["lr"] /= 10
                    self.generator_optimizer.param_groups[0]["lr"] /= 10
                if (epoch + 1) == 35:
                    self.discriminator_optimizer.param_groups[0]["lr"] /= 10
                    self.generator_optimizer.param_groups[0]["lr"] /= 10

        return TrainedGan(
            self.discriminator, self.generator, discriminator_losses, generator_losses
        )


@dataclass
class TrainedGan:
    discriminator: nn.Module
    generator: nn.Module
    discriminator_losses: List[float]
    generator_losses: List[float]


## KL GAN


def get_conjugate_score_kl(scores):
    conjugate_score = torch.exp(scores - 1)
    return conjugate_score


def get_gen_loss_kl(fake_scores):
    gen_loss = -1.0 * torch.mean(get_conjugate_score_kl(fake_scores))
    return gen_loss


def get_dis_loss_kl(real_scores, fake_scores):
    dis_loss = torch.mean(get_conjugate_score_kl(fake_scores)) - torch.mean(real_scores)
    return dis_loss


## RV KL GAN


def get_conjugate_score_rkl(scores):
    conjugate_score = -1.0 - torch.log(-scores)
    return conjugate_score


def get_gen_loss_rkl(fake_scores):
    gen_loss = -1.0 * torch.mean(get_conjugate_score_rkl(fake_scores))
    return gen_loss


def get_dis_loss_rkl(real_scores, fake_scores):
    dis_loss = torch.mean(get_conjugate_score_rkl(fake_scores)) - torch.mean(
        real_scores
    )
    return dis_loss


## GAN GAN


def get_conjugate_score_gan(scores):
    conjugate_score = -torch.log(1 - torch.exp(scores))
    return conjugate_score


def get_gen_loss_gan(fake_scores):
    gen_loss = -1.0 * torch.mean(torch.log(fake_scores))
    return gen_loss


def get_dis_loss_gan(real_scores, fake_scores):
    dis_loss = -1.0 * torch.mean(torch.log(1 - fake_scores)) - torch.mean(
        torch.log(real_scores)
    )
    return dis_loss


## Pearson GAN


def get_conjugate_score_p(scores):
    conjugate_score = (1 / 4) * torch.pow(scores, 2) + scores
    return conjugate_score


def get_gen_loss_p(fake_scores):
    gen_loss = -1.0 * torch.mean(get_conjugate_score_p(fake_scores))
    return gen_loss


def get_dis_loss_p(real_scores, fake_scores):
    dis_loss = torch.mean(get_conjugate_score_p(fake_scores)) - torch.mean(real_scores)
    return dis_loss


## BHS GAN
def get_gen_loss_bhs(fake_scores):
    gen_loss = -1.0 * torch.mean(get_conjugate_score(fake_scores))
    return gen_loss


def get_dis_loss_bhs(real_scores, fake_scores):
    dis_loss = torch.mean(get_conjugate_score(fake_scores)) - torch.mean(real_scores)
    return dis_loss


def get_conjugate_score(scores):
    conjugate_score_1 = (
        2.0 * (-1 + torch.sqrt(1 + scores)) * torch.exp(-1 + torch.sqrt(1 + scores))
    )
    conjugate_score_2 = (
        2.0 * (-1 - torch.sqrt(1 + scores)) * torch.exp(-1 - torch.sqrt(1 + scores))
    )
    return torch.where(torch.ge(scores, 0), conjugate_score_1, conjugate_score_2)


def get_gen_loss_wasserstein(fake_scores):
    gen_loss = -1.0 * torch.mean(fake_scores)
    return gen_loss


def get_dis_loss_wasserstein(real_scores, fake_scores, gradient_penalty):
    dis_loss = (
        torch.mean(fake_scores) - torch.mean(real_scores) + 10.0 * gradient_penalty
    )
    return dis_loss


def get_gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)

    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def get_gradient(discriminator, real_numbers, fake, epsilon, device):
    mixed_numbers = real_numbers * epsilon + fake * (1 - epsilon)

    mixed_scores = discriminator(mixed_numbers)

    gradient = torch.autograd.grad(
        inputs=mixed_numbers,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def get_gen_loss_ipm(fake_scores):
    gen_loss = -1.0 * torch.mean(fake_scores)
    return gen_loss


def get_dis_loss_ipm(real_scores, fake_scores):
    dis_loss = (
        torch.mean(fake_scores)
        - torch.mean(real_scores)
        + 0.1 * (torch.mean(torch.square(real_scores)))
    )
    return dis_loss


## universal f-Gan
def get_gen_loss_uf(fake_scores):
    gen_loss = torch.norm(fake_scores, 1)
    return gen_loss


def get_dis_loss_uf(real_scores, fake_scores):
    dis_loss = -torch.norm(real_scores, 1) - torch.norm(fake_scores, 1)
    return dis_loss
