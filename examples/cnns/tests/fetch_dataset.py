import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(sys.path)

# print('Start training...')

from datasets import get_cifar_dataloaders, get_mnist_dataloaders

# train_loader, test_loader = get_cifar_dataloaders(128, 128, 'cifar10', 4, datadir='/media/data/Unknown/cifar10', skip_download_check=True)
# train_loader, test_loader = get_cifar_dataloaders(128, 128, 'cifar100', 4, datadir='/media/data/Unknown/cifar100', skip_download_check=True)
# train_loader, test_loader = get_mnist_dataloaders(128, 128, 4, datadir='/media/data/Unknown/mnist')

print('Done.')

# show some images
# cifar10
train_loader, test_loader = get_cifar_dataloaders(64, 64, 'cifar10', 4, datadir='/media/data/Unknown/cifar10', skip_download_check=True)
iter_loader = iter(train_loader)
images, labels = next(iter_loader)
print(images.shape)
print(labels.shape)
# visualize
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
to_img = ToPILImage()
img = make_grid(images)
img = to_img(img)
plt.imshow(np.asarray(img))
plt.savefig('cifar10.png')