import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import AnimeDataset

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)

# Training Loop
def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AnimeDataset("./data/processed")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(100):
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            D.zero_grad()
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Real images
            outputs = D(real_imgs)
            loss_real = criterion(outputs, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, 100, 1, 1).to(device)
            fake_imgs = G(z)
            outputs = D(fake_imgs.detach())
            loss_fake = criterion(outputs, fake_labels)
            
            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            # Train Generator
            G.zero_grad()
            outputs = D(fake_imgs)
            loss_G = criterion(outputs, real_labels)
            loss_G.backward()
            opt_G.step()

        print(f"Epoch {epoch}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}")

if __name__ == "__main__":
    train_gan()
