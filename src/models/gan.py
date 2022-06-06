from torch import batch_norm, nn, optim
from torch.nn import functional as F
import torch.autograd as autograd

import numpy as np
import torch
from tqdm import tqdm

from torch.autograd import Variable
from .architectures import FCDiscriminator, FCGenerator
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class GAN:

    def __init__(self, generator, discriminator, input_size, latent_dim=100):
        self.generator = FCGenerator(input_size, latent_dim) if generator is None else generator
        self.discriminator = FCDiscriminator(input_size) if discriminator is None else discriminator
        self.latent_dim = self.generator.latent_dim if generator is None else latent_dim
        # Loss weight for gradient penalty
        self.lambda_gp = 10

    def fit(self, dataloader, b1=0.5, b2=0.999, clip_value=0.01, lr=0.0002, epochs=5, n_critic=5):

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

        # Optimizers
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        batches_done = 0
        for epoch in tqdm(range(epochs)):
            for i, imgs in enumerate(dataloader):
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                #  Train Discriminator
                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z)

                # Real images
                real_validity = self.discriminator(real_imgs)
                # Fake images
                fake_validity = self.discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = self.compute_gradient_penalty(self.discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self.lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if i % n_critic == 0:

                    # Generate a batch of images
                    fake_imgs = self.generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = self.discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" 
                        % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )

                    batches_done += n_critic

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