'''
Author: Unknown
Date: 2024-03-01 00:42:32
LastEditTime: 2024-03-01 01:02:02
LastEditors: Unknown
Description: 
FilePath: /Unknown/tests/test_transform.py
'''

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, Resize, RandomHorizontalFlip


def test_transform():
    transform1 = Compose([
        Resize(32),
        ToTensor(),
    ])
    transform2 = Compose([
        ToTensor(),
    ])

    random_crop = RandomCrop(32, padding=4)

    img = Image.open('./test_image.png')
    
    img1 = transform1(img)
    img2 = transform2(img)




if __name__ == '__main__':
    test_transform()



