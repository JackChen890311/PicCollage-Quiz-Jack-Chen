import torch
import torch.nn as nn
from abc import ABC, abstractmethod

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class BaseModel(nn.Module, ABC):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()
    
    @abstractmethod
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        pass

    @abstractmethod
    def load_state(self, paths: dict) -> None:
        """Load model state from given paths"""
        pass


class GANModel(BaseModel):
    """Base class for GAN models"""
    def __init__(self, latent_dim, input_dim, hidden_dim, dg_ratio=5):
        super(GANModel, self).__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dg_ratio = dg_ratio
        self.generator = Generator(latent_dim, input_dim, hidden_dim)
        self.discriminator = Discriminator(input_dim, hidden_dim)

    def sample(self, num_samples, device):
        """Generate samples using the generator"""
        self.generator.eval()
        self.generator.to(device)
        return self.generator.sample(num_samples, device)
    
    def load_state(self, paths):
        """Load model state from given paths"""
        self.generator.load_state_dict(torch.load(paths['generator']))
        self.discriminator.load_state_dict(torch.load(paths['discriminator']))


class Generator(nn.Module):
    """GAN generator"""
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Generator, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.model = nn.Sequential(*layers)
        self.model.apply(weights_init)
        self.input_dim = input_dim

    def forward(self, z):
        return self.model(z)
    
    def sample(self, num_samples, device):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.input_dim, device=device)
            samples = self.forward(z)
        return samples.cpu().numpy()
    

class Discriminator(nn.Module):
    """GAN discriminator"""
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())
        for i in range(len(hidden_dim) - 1):
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim[-1], 1))
        self.model = nn.Sequential(*layers)
        self.model.apply(weights_init)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
    

class VAEModel(BaseModel):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAEModel, self).__init__()

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim[0]))
        encoder_layers.append(nn.ReLU())
        for i in range(len(hidden_dim) - 1):
            encoder_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder.apply(weights_init)

        self.fc_mu = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim[-1], latent_dim)
        self.fc_mu.apply(weights_init)
        self.fc_logvar.apply(weights_init)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim[-1]))
        decoder_layers.append(nn.ReLU())
        for i in range(len(hidden_dim) - 1, 0, -1):
            decoder_layers.append(nn.Linear(hidden_dim[i], hidden_dim[i-1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dim[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder.apply(weights_init)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def sample(self, num_samples, device):
        self.to(device)
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
            samples = self.decode(z)
        return samples.cpu().numpy()
    
    def load_state(self, paths):
        """Load model state from given paths"""
        self.load_state_dict(torch.load(paths['vae']))
        self.eval()