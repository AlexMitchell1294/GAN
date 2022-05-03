import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# image detail max 256 unless more than 8 gigs of ram
discriminator_features = 50
generator_features = 150

# Hyperparameters
lr = 0.0005
batch_size = 16
image_size = 64
channels_img = 1
channels_noise = 30
num_epochs = 20000

## IF USING MNIST DATASET
my_transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = datasets.MNIST(
    root="dataset/", train=True, transform=my_transforms, download=True
)

# ### IF USING CUSTOM DATASET
# transform = transforms.Compose(
#     [
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5 for _ in range(3)],
#             [0.5 for _ in range(3)],
#         ),
#     ]
# )

# dataset = datasets.ImageFolder(root="D:\WPI\Classes\ML\FINAL\\flower test\\flower_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)