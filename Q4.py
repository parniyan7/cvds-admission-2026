import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

# ===================================================================
# EXERCISE 4 - GAN Training
# ===================================================================

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output


def train_gan(batch_size: int = 32, num_epochs: int = 50, 
              device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):
    """
    Trains a Generative Adversarial Network (GAN) to generate MNIST digits.
    
    Original bugs:
    1. Structural bug: When batch_size > 32, torch.cat((real_samples, generated_samples))
       created 2*batch_size samples, but the labels were still sized batch_size.
       This caused a shape mismatch error in BCELoss (e.g. [128,1] vs [96,1]).
    
    2. Cosmetic bug: The condition `if n == batch_size - 1` was incorrect.
       It only showed generated images on the very last batch of each epoch.
    
    How I fixed it:
    - Used the actual current batch size (important for the last incomplete batch).
    - Properly created labels matching the real batch size.
    - Separated real and fake loss calculation clearly.
    - Changed progress display to every 100 batches for better monitoring.
    - Added clear comments explaining the fixes.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load MNIST dataset
    train_set = torchvision.datasets.MNIST(root=".", train=True, 
                                           download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                               shuffle=True)

    # Show some real images as verification
    real_samples, _ = next(iter(train_loader))
    fig = plt.figure(figsize=(8, 8))
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis('off')
    fig.suptitle("Real MNIST Images (before training)")
    fig.tight_layout()
    display(fig)
    time.sleep(3)

    # Initialize models and optimizers
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    lr = 0.0001
    loss_function = nn.BCELoss()
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)

    print(f"Starting GAN training on {device}...")

    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            real_samples = real_samples.to(device)
            current_batch_size = real_samples.size(0)   # Important for last batch

            # ===================== TRAIN DISCRIMINATOR =====================
            discriminator.zero_grad()

            # Real images
            real_labels = torch.ones((current_batch_size, 1), device=device)
            output_real = discriminator(real_samples)
            loss_real = loss_function(output_real, real_labels)

            # Fake images
            noise = torch.randn((current_batch_size, 100), device=device)
            fake_samples = generator(noise)
            fake_labels = torch.zeros((current_batch_size, 1), device=device)
            output_fake = discriminator(fake_samples.detach())
            loss_fake = loss_function(output_fake, fake_labels)

            # Total discriminator loss
            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # ===================== TRAIN GENERATOR =====================
            generator.zero_grad()

            noise = torch.randn((current_batch_size, 100), device=device)
            fake_samples = generator(noise)
            output_fake = discriminator(fake_samples)
            loss_g = loss_function(output_fake, real_labels)   # Fool the discriminator
            loss_g.backward()
            optimizer_g.step()

            # Show progress every 100 batches
            if n % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{n}/{len(train_loader)}] "
                      f"| Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")

                # Display generated samples
                with torch.no_grad():
                    generated = fake_samples.detach().cpu().numpy()
                    fig = plt.figure(figsize=(8, 8))
                    for i in range(min(16, current_batch_size)):
                        sub = fig.add_subplot(4, 4, 1 + i)
                        sub.imshow(generated[i].reshape(28, 28), cmap="gray_r")
                        sub.axis('off')
                    fig.suptitle(f"Epoch {epoch+1} | Loss D: {loss_d.item():.4f} | Loss G: {loss_g.item():.4f}")
                    fig.tight_layout()
                    clear_output(wait=True)
                    display(fig)

    print("GAN Training completed successfully!")


# ===================================================================
# Test for Exercise 4
# ===================================================================

if __name__ == "__main__":
    print("Running Exercise 4 Test (small number of epochs for quick check)...")
    train_gan(batch_size=32, num_epochs=5)   # Use small epochs for testing