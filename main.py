import os
import yaml
import torch
import torch.optim as optim
import numpy as np

from dataloader import PixelDataLoader
from model import GANModel, VAEModel
from trainer import GANTrainer, VAETrainer


if __name__ == '__main__':
    # Load configuration
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    DEVICE = torch.device(CONFIG['hyperparameter']['device'])
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    # Load data and standardize
    p_loader = PixelDataLoader(
        CONFIG['data_path']['x_pos'], 
        CONFIG['data_path']['y_pos'],
        CONFIG['data_path']['image'],)
    p_loader.standardize()
    train_loader, test_loader = p_loader.get_dataloader(
        batch_size=CONFIG['hyperparameter']['batch_size'],
        split_ratio=CONFIG['hyperparameter']['split_ratio']
    )

    # Initialize model, trainer, optimizers, and load state paths
    if CONFIG['model']['type'] == 'vae':
        model = VAEModel(
            input_dim=CONFIG['model']['input_dim'],
            hidden_dim=CONFIG['model']['hidden_dim'],
            latent_dim=CONFIG['model']['latent_dim']
        )
        trainer = VAETrainer()
        optimizers = {
            'vae': optim.Adam(model.parameters(), lr=CONFIG['hyperparameter']['learning_rate_vae'])
        }
        load_state_paths = {
            'vae': 'models/best_vae.pth'
        }
    elif CONFIG['model']['type'] == 'gan':
        model = GANModel(
            latent_dim=CONFIG['model']['latent_dim'],
            input_dim=CONFIG['model']['input_dim'],
            hidden_dim=CONFIG['model']['hidden_dim'],
            dg_ratio=CONFIG['model']['dg_ratio']
        )
        trainer = GANTrainer()
        optimizers = {
            'generator': optim.Adam(model.generator.parameters(),
                                    betas=(CONFIG['hyperparameter']['beta1'], CONFIG['hyperparameter']['beta2']),
                                    lr=CONFIG['hyperparameter']['learning_rate_g']),
            'discriminator': optim.Adam(model.discriminator.parameters(),
                                        betas=(CONFIG['hyperparameter']['beta1'], CONFIG['hyperparameter']['beta2']),
                                        lr=CONFIG['hyperparameter']['learning_rate_d'])
        }
        load_state_paths = {
            'generator': 'models/final_generator.pth',
            'discriminator': 'models/final_discriminator.pth'
        }


    # Train Model and load the best state
    if CONFIG['task']['train']:
        print(f'Training {CONFIG["model"]["type"].upper()} model...')
        trainer.train(model, {'train': train_loader, 'test': test_loader}, 
                      optimizers, DEVICE, CONFIG['hyperparameter']['epochs'])
    if CONFIG['task']['sample']:
        print(f'Sampling from {CONFIG["model"]["type"].upper()} model...')
        model.load_state(load_state_paths)


    # Sample the points and save the results
    samples = model.sample(num_samples=CONFIG['hyperparameter']['num_samples'], device=DEVICE)
    samples = p_loader.destandardize(samples)
    np.save(CONFIG['data_path']['output_samples'], samples)
    print(f'Samples saved to {CONFIG["data_path"]["output_samples"]}')