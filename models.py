import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
from torch.nn import functional as F
import pytorch_lightning as pl


class Generator(nn.Module):
    def __init__(self, latent_dim, input_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, np.prod(input_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        generated_audio = self.fc(z)
        generated_audio = generated_audio.view(generated_audio.size(0), *self.input_shape)
        return generated_audio

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, audio):
        audio_flat = audio.view(audio.size(0), -1)
        validity = self.fc(audio_flat)
        return validity


class GAN(pl.LightningModule):
    def __init__(self, latent_dim, data_shape):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        self.data_shape = data_shape

        self.generator = Generator(latent_dim, data_shape)
        self.discriminator = Discriminator(data_shape)

        self.loss = nn.BCELoss()

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_data = batch
        batch_size = real_data.size(0)

        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # Train Generator
        if optimizer_idx == 0:
            z = torch.randn(batch_size, self.latent_dim)
            generated_data = self.generator(z)
            g_loss = self.adversarial_loss(
                self.discriminator(generated_data), valid
            )
            self.log("g_loss", g_loss.item(), on_step=False, on_epoch=True)
            return g_loss

        # Train Discriminator
        if optimizer_idx == 1:
            real_loss = self.adversarial_loss(
                self.discriminator(real_data), valid
            )
            z = torch.randn(batch_size, self.latent_dim)
            generated_data = self.generator(z).detach()
            fake_loss = self.adversarial_loss(
                self.discriminator(generated_data), fake
            )
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss.item(), on_step=False, on_epoch=True)
            return d_loss

    def configure_optimizers(self):
        lr = 0.0002
        betas = (0.5, 0.999)

        optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas
        )

        return [optimizer_G, optimizer_D], []

if __name__ == "__main__":
    # Example usage
    latent_dim = 100
    data_shape = 16384  # Update with the actual size of your music signals
    batch_size = 64

    music_dataset = MusicDataset(data_path="path/to/your/music/file.wav")
    dataloader = DataLoader(music_dataset, batch_size=batch_size, shuffle=True)

    gan = GAN(latent_dim=latent_dim, data_shape=data_shape)

    trainer = pl.Trainer(max_epochs=50, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(gan, dataloader)

      