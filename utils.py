import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# image detail max 256 unless more than 8 gigs of ram
discriminator_features = 256
generator_features = 256

# Hyperparameters
lr = 0.00005
batch_size = 8
image_size = 64
channels_img = 3
channels_noise = 5
num_epochs = 5000

### IF USING MNIST DATASET
# my_transforms = transforms.Compose(
#     [
#         transforms.Resize(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,)),
#     ]
# )

# dataset = datasets.MNIST(
#     root="dataset/", train=True, transform=my_transforms, download=True
# )

### IF USING CUSTOM DATASET
transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(3)],
            [0.5 for _ in range(3)],
        ),
    ]
)

dataset = datasets.ImageFolder(root="D:\WPI\Classes\ML\FINAL\GAN\\flowers", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)