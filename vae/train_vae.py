""" Training VAE """
import argparse
import time
from os.path import join, exists
from os import mkdir

import random

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from models import VAE

from utils.misc import save_checkpoint
from utils.misc import LSIZE
## WARNING : THIS SHOULD BE REPLACE WITH PYTORCH 0.5
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from utils.loaders import ImagePairDataset


import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

parser = argparse.ArgumentParser(description='VAE Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--logdir', type=str, default="log_dir",  help='Directory where results are logged')
parser.add_argument('--reload', action='store_true',
                    help='Best model is reloaded if specified')
parser.add_argument('--samples', action='store_true',
                    help='Does save samples during training if specified')


args = parser.parse_args()
cuda = torch.cuda.is_available()



torch.manual_seed(4435)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if cuda else "cpu")
print(device)




# Define the location of the image directory
image_rgb = '/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/images/rgb'
image_seg = '/home/albertomate/Documentos/carla/PythonAPI/my-carla/vae/images/segmentation'


# Define the transformations to apply to the images
transform = transforms.Compose(
    [transforms.ToTensor()])

# Create the dataset from the image directory
dataset = ImagePairDataset(image_rgb, image_seg, transform)
train_size = int(len(dataset) * 0.8)
val_size = int(len(dataset) * 0.2)
train_set, val_set = random_split(dataset, [train_size, val_size])

# Create the data loader with a batch size of 32
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)




model = VAE(LSIZE).to(device)
optimizer = optim.Adam(model.parameters(), 0.003)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.3, patience=10)
earlystopping = EarlyStopping('min', patience=25)

print(f"Training VAE with latent size {LSIZE}\n")
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return BCE + KLD


def train(epoch):
    """ One training epoch """
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        input_image, output_image = data
        input_image = input_image.to(device)
        output_image = output_image.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(input_image)

        loss = loss_function(recon_batch, output_image, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_image), len(train_loader)*batch_size,
                100. * batch_idx / len(train_loader),
                loss.item() / len(input_image)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader)))
    return train_loss / len(train_loader)

def test():
    """ One test epoch """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            input_image, output_image = data
            input_image = input_image.to(device)
            output_image = output_image.to(device)
            recon_batch, mu, logvar = model(input_image)
            test_loss += loss_function(recon_batch, output_image, mu, logvar).item()
    test_loss /= len(val_loader)
    print('====> Test set loss: {:.4f}\n'.format(test_loss))
    return test_loss

# check vae dir exists, if not, create it
vae_dir = join(args.logdir, f'vae_{LSIZE}')
os.makedirs(join(vae_dir, 'samples'), exist_ok=True)
if not exists(vae_dir):
    mkdir(vae_dir)
    mkdir(join(vae_dir, 'samples'))

reload_file = join(vae_dir, 'best.tar')
if args.reload and exists(reload_file):
    state = torch.load(reload_file)
    print("Reloading model at epoch {}"
          ", with test error {}".format(
              state['epoch'],
              state['precision']))
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    earlystopping.load_state_dict(state['earlystopping'])


cur_best = None
acc_train_loss=[]
acc_val_loss=[]
for epoch in range(1, args.epochs + 1):
    train_loss = train(epoch)
    val_loss = test()

    acc_train_loss.append(train_loss)
    acc_val_loss.append(val_loss)
    scheduler.step(val_loss)
    earlystopping.step(val_loss)

    # checkpointing
    best_filename = join(vae_dir, 'best.tar')
    filename = join(vae_dir, 'checkpoint.tar')
    is_best = not cur_best or val_loss < cur_best
    if is_best:
        cur_best = val_loss

    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'precision': val_loss,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'earlystopping': earlystopping.state_dict()
    }, is_best, filename, best_filename)



    if args.samples:
        with torch.no_grad():
            sample = torch.randn(1,LSIZE).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view( 1, 3, 80, 160),
                join(vae_dir, 'samples/sample_' + str(epoch) + '.png'))

    if earlystopping.stop:
        print("End of Training because of early stopping at epoch {}".format(epoch))
        break
def plot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(vae_dir, 'loss.png'))
    print(f"Plots generated in {join(vae_dir, 'loss.png')}")

plot_loss(acc_train_loss, acc_val_loss)