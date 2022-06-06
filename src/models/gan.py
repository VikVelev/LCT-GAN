from torch import batch_norm, nn, optim, rand
from torch.nn import functional as F
import torch.autograd as autograd
import random

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from torch.autograd import Variable
from .architectures import FCDiscriminator, FCGenerator
from src.utils.ctabgan_synthesizer import Condvec, Sampler, cond_loss
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class LatentGAN:

    # input_size should be input_size + cond_vector
    def __init__(self, input_size, latent_dim=100):
        self.input_size = input_size
        self.generator = None
        self.discriminator = None
        self.latent_dim = latent_dim
        # Loss weight for gradient penalty
        self.lambda_gp = 10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, latent_data, original_data, transformer, batch_size=256, b1=0.5, b2=0.999, clip_value=0.01, lr=0.0002, epochs=5, n_critic=5):
        
        assert len(latent_data) == len(original_data)

        transformer_output_info = transformer.output_info
        
        cond_generator = Condvec(original_data, transformer_output_info)
        data_sampler = Sampler(original_data, transformer_output_info)

        self.cond_generator = cond_generator
        self.data_sampler = data_sampler
        self.batch_size = batch_size
        self.transformer = transformer

        self.generator = FCGenerator(self.input_size, self.latent_dim + cond_generator.n_opt) 
        self.discriminator = FCDiscriminator(self.input_size + cond_generator.n_opt, batch_size=batch_size)

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        for epoch in tqdm(range(epochs)):
            
            cond_vecs = cond_generator.sample_train(batch_size)
            c, m, col, opt = cond_vecs
            c = torch.from_numpy(c).to(self.device)
            m = torch.from_numpy(m).to(self.device)

            original_idx = data_sampler.sample_idx(batch_size, col, opt)
            real = latent_data[original_idx]
            real = torch.from_numpy(real.astype('float32')).to(self.device)
            
            ### TRAIN DISCRIMINATOR
            optimizer_D.zero_grad()

            z = Tensor(np.random.uniform(0, 1, (batch_size, self.latent_dim)))
            z = torch.cat([z , c], dim=1)
            z = Variable(z)

            fake = self.generator(z)
            
            fake_cat_d = torch.cat([fake, c], dim=1).to(self.device)
            real_cat_d = torch.cat([Variable(real.type(Tensor)), c], dim=1).to(self.device)

            real_probability = self.discriminator(real_cat_d)
            fake_probability = self.discriminator(fake_cat_d)

            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_cat_d.data, fake_cat_d.data)
            # Adversarial loss
            loss_d = -torch.mean(real_probability) + torch.mean(fake_probability) + self.lambda_gp * gradient_penalty
            loss_d.backward()

            optimizer_D.step()

            ### TRAIN GENERATOR
            optimizer_G.zero_grad()

            if random.randint(0, n_critic) == n_critic:

                # Generate a batch of data
                fake = self.generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake data
                fake_probability = self.discriminator(torch.cat([fake, c], dim=1).to(self.device))

                # cross_entropy = cond_loss(fake, transformer_output_info, c, m)
                # computing the loss to train the generator where we want y_fake to be close to 1 to fool the discriminator 
                # and cross_entropy to be close to 0 to ensure generator's output matches the conditional vector  
                loss_g = -torch.mean(fake_probability)
                loss_g.backward()

                optimizer_G.step()

                print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" 
                    % (epoch, epochs, loss_d.item(), loss_g.item())
                )
    

    def compute_gradient_penalty(self, D, real_samples, fake_samples):

        """Calculates the gradient penalty loss for WGAN GP"""

        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)

        fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def sample(self, n, decoder):
        
        # turning the generator into inference mode to effectively use running statistics in batch norm layers
        self.generator.eval()
        
        # generating synthetic data in batches accordingly to the total no. required
        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            # generating synthetic data using sampled noise and conditional vectors
            ### Generating a batch
            z = Tensor(np.random.uniform(0, 1, (self.batch_size, self.latent_dim)))
            z_cond = Tensor(self.cond_generator.sample(self.batch_size))
            z = torch.cat([z, z_cond], dim=1).to(self.device)
            fake = self.generator(z)

            data.append(decoder.decode(fake, batch=True))

        data = pd.concat(data)
        return data[0:n]