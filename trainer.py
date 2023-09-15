import torch
import numpy as np
from time import time
from utils import get_noise
from dataclasses import dataclass
from typing import List
import torch.nn as nn


@dataclass()
class TrainingParams:
    lr : float
    beta_1: float
    num_epochs: int
    num_dis_updates: int
    batch_size: int
    

class Trainer:
    def __init__(self, training_params, generator, discriminator):
        self.training_params = training_params
        self.generator = generator
        self.discriminator = discriminator
        self.discriminator_optimizer = self._init_dis_optimizer(training_params)
        self.generator_optimizer = self._init_dis_optimizer(training_params)
        
    def _init_dis_optimizer(self, training_params):
        lr = training_params.lr
        beta_1 = training_params.beta_1
        return torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta_1, 0.999))
    
    def _init_gen_optimizer(self, training_params):
        lr = training_params.lr
        beta_1 = training_params.beta_1
        return torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta_1, 0.999))

    def train_gan(self, 
                dataloader, 
                get_dis_loss, 
                get_gen_loss,
                gradient_penalty_enabled):
        
        num_epochs = self.training_params.num_epochs
        num_dis_updates = self.training_params.num_dis_updates
        batch_size = self.training_params.batch_size
        
        generator_losses = []
        discriminator_losses = []
        tmp_d_losses = []
        batches_done = 0
        for epoch in range(num_epochs):
            print('Epoch ' + str(epoch) + ' training...' , end=' ')
            start = time()
            for i, real_sample in enumerate(dataloader):
                real_sample = torch.reshape(real_sample, (batch_size, 1))
                # train Discriminator
                self.discriminator_optimizer.zero_grad()
                # sample noise as generator input
                noise = get_noise(batch_size, 1)
                # generate a batch of images
                fake_sample = self.generator(noise)
                # Adversarial loss
                real_scores = self.discriminator(real_sample)
                fake_scores = self.discriminator(fake_sample.detach())
                if gradient_penalty_enabled:
                    gradient = get_gradient(self.discriminator, real_sample, fake_sample.detach())
                    gradient_penalty = get_gradient_penalty(gradient)
                    discriminator_loss = get_dis_loss(real_scores, fake_scores, gradient_penalty)
                else:
                    discriminator_loss = get_dis_loss(real_scores, fake_scores)
                tmp_d_losses.append(discriminator_loss.item())
                discriminator_loss.backward(retain_graph=True)
                self.discriminator_optimizer.step()
                # train the generator every num_dis_updates iterations
                if i % num_dis_updates == 0:
                    discriminator_losses.append(np.mean(tmp_d_losses))
                    tmp_d_losses = []
                    # train Generator
                    self.generator_optimizer.zero_grad()
                    # generate a batch of fake images
                    fake_sample = self.generator(noise)
                    # Adversarial loss
                    fake_scores = self.discriminator(fake_sample)
                    generator_loss = get_gen_loss(fake_scores)
                    generator_losses.append(generator_loss.item())
                    generator_loss.backward()
                    self.generator_optimizer.step()
                batches_done += 1
            end = time()
            elapsed = end - start
            print('done, took %.1f seconds.' % elapsed)
            
        return TrainedGan(self.discriminator, self.generator, discriminator_losses, generator_losses)


@dataclass
class TrainedGan:
    discriminator: nn.Module
    generator: nn.Module
    discriminator_losses: List[float]
    generator_losses: List[float]


def get_gen_loss_bhs(fake_scores):
    gen_loss = torch.mean(get_conjugate_score(fake_scores))
    return gen_loss


def get_dis_loss_bhs(real_scores, fake_scores):
    dis_loss = -torch.mean(real_scores) + torch.mean(get_conjugate_score(fake_scores))
    return dis_loss


def get_conjugate_score(scores):
    return 2 * (-1 + torch.sqrt(1 + scores)) * torch.exp(torch.sqrt(1 + scores))
    

def get_gen_loss_wasserstein(fake_scores):
    gen_loss = -1 * torch.mean(fake_scores)
    return gen_loss


def get_dis_loss_wasserstein(real_scores, fake_scores, gradient_penalty):
    dis_loss = torch.mean(fake_scores) - torch.mean(real_scores) + 0.01 * gradient_penalty
    return dis_loss


def get_gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)

    gradient_norm = gradient.norm(2, dim=1)
    
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty


def get_gradient(discriminator, real_numbers, fake):
    epsilon = torch.rand(1)
    mixed_numbers = (real_numbers * epsilon + fake * (1 - epsilon)).requires_grad_(True)

    mixed_scores = discriminator(mixed_numbers)
    
    gradient = torch.autograd.grad(
        inputs=mixed_numbers,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, requires_grad=False), 
        create_graph=True,
        retain_graph=True,
        
    )[0]
    return gradient