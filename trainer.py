import os
import tqdm
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class Trainer(ABC):
    """Base class for training models"""
    def __init__(self):
        pass

    @abstractmethod
    def train(self, model, dataloaders, optimizers, device, epochs):
        pass


class GANTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def bce_loss(self, output, target):
        """Binary Cross Entropy loss"""
        return F.binary_cross_entropy(output, target)

    def gan_loss(self, real_output, fake_output):
        """Calculate the GAN loss"""
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss
    
    def train(self, model, dataloaders, optimizers, device, epochs):
        """Train a GAN model"""
        model.train()
        model.to(device)
        generator, discriminator = model.generator, model.discriminator
        optimizer_g, optimizer_d = optimizers['generator'], optimizers['discriminator']
        train_loader = dataloaders['train']
        test_loader = dataloaders['test']
        latent_dim_gan = model.latent_dim
        best_loss = float('inf')
        os.makedirs('models', exist_ok=True)

        for epoch in tqdm.tqdm(range(epochs)):
            total_d_loss = 0.0
            total_g_loss = 0.0
            for real_data in train_loader:
                real_data = real_data.to(device)

                # Train Discriminator
                optimizer_d.zero_grad()
                z = torch.randn(real_data.size(0), latent_dim_gan, device=device)
                fake_data = generator(z)
                real_output = discriminator(real_data)
                fake_output = discriminator(fake_data.detach())
                d_loss = self.gan_loss(real_output, fake_output)
                d_loss.backward()
                optimizer_d.step()

                # Train Generator
                optimizer_g.zero_grad()
                fake_output = discriminator(fake_data)
                g_loss = self.bce_loss(fake_output, torch.ones_like(fake_output))
                g_loss.backward()
                optimizer_g.step()

                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                
                if total_d_loss + total_g_loss < best_loss:
                    best_loss = total_g_loss + total_d_loss
                    torch.save(generator.state_dict(), 'models/best_generator.pth')
                    torch.save(discriminator.state_dict(), 'models/best_discriminator.pth')

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], D Loss: {total_d_loss}, G Loss: {total_g_loss}')
        torch.save(generator.state_dict(), f'models/best_generator_final.pth')
        torch.save(discriminator.state_dict(), f'models/best_discriminator_final.pth')


class VAETrainer(Trainer):
    def __init__(self):
        super().__init__()

    def vae_loss(self, x_recon, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl_loss

    def train(self, model, dataloaders, optimizers, device, epochs):
        """Train a VAE model"""
        model.train()
        model.to(device)
        optimizer = optimizers['vae']
        train_loader = dataloaders['train']
        test_loader = dataloaders['test']
        best_loss = float('inf')
        os.makedirs('models', exist_ok=True)

        for epoch in tqdm.tqdm(range(epochs)):
            total_loss = 0.0
            for data in train_loader:
                data = data.to(device)

                optimizer.zero_grad()
                x_recon, mu, logvar = model(data)
                loss = self.vae_loss(x_recon, data, mu, logvar)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), 'models/best_vae.pth')

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader)}')
        
        torch.save(model.state_dict(), 'models/best_vae_final.pth')