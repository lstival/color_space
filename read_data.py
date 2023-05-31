import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
import kornia as K
from utils import *
import random
import numpy as np

#https://www.cambridgeincolour.com/tutorials/color-spaces.htm

class ColorizationDataset(Dataset):
    """
    This class create a dataset from a path, where the data must be
    divided in subfoldes for each scene or video.
    The return is a list with 3 elements, frames to ve colorized,
    the direclety next frames in the video sequence and example image with color.
    """

    def __init__(self, path, image_size, color_transform, norm_params, constrative=False):
        super(Dataset, self).__init__()

        self.path = path
        self.image_size = image_size
        self.constrative = constrative
        self.color_transform = color_transform
        self.norm_params = norm_params

        self.scenes = os.listdir(path)

        self.dataset = torchvision.datasets.ImageFolder(self.path, self.__transform__)

    def __colorization_transform__(self, x):

        colorization_transform=transforms.Compose([
                torchvision.transforms.Resize(280),  # args.image_size + 1/4 *args.image_size
                torchvision.transforms.RandomResizedCrop(self.image_size, scale=(0.8, 1.0)),
                torchvision.transforms.RandomRotation((0, 365)),
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
                self.color_transform,
                transforms.Normalize(mean=self.norm_params[0], std=self.norm_params[1]),
            ])
        
        return colorization_transform(x)

    def __transform__(self, x):
        """
        Recives a sample of PIL images and return
        they normalized and converted to a tensor.
        """

        x_transformed = self.__colorization_transform__(x)
        return x_transformed
        
    def __len__(self):
        """
        Return hou much samples as in the dataset.
        """
        return len(self.dataset)

    
    def __getitem__(self, index):
        """
        Return the frames that will be colorized, the next frames and 
        the color example frame (first of the sequence).
        """
        # Get the next indices
        keyframe_index = random.randint(0,10)
        next_index = min(index + 1, len(self.dataset) - 1)
        
        if self.constrative:
            random_idx = random.randint(20, 100)
            random_idx = min(random_idx + 1, len(self.dataset) - 1)
            return self.dataset[index], self.dataset[keyframe_index], self.dataset[next_index], self.dataset[random_idx] 
        else:
            return self.dataset[index], self.dataset[keyframe_index], self.dataset[next_index]
        

# Create the dataset
class ReadData():

    # Initilize the class
    def __init__(self) -> None:
        super().__init__()

    def create_dataLoader(self, dataroot, image_size, color_transform, norm_params, batch_size=16, shuffle=False, pin_memory=True, constrative=False , train=None):

        self.datas = ColorizationDataset(dataroot, image_size, color_transform, norm_params=norm_params, constrative=constrative)

        self.dataloader = torch.utils.data.DataLoader(self.datas, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory)

        return self.dataloader
    
def color_params(color_space):
        rgb_to_lab = K.color.RgbToLab()
        rgb_to_yuv = K.color.RgbToYuv()
        rgb_to_hsv = K.color.RgbToHsv()

        ## Color space transformation
        match color_sapce:
            case "lab":
                color_transform = rgb_to_lab
                mean = torch.tensor([50., 0., 0.])
                std = torch.tensor([100.0, 255.0, 255.0])
            case "yuv":
                color_transform = rgb_to_yuv
                mean = torch.tensor([0.5, 0., 0.])
                std = torch.tensor([0.5, 0.5, 0.5])
            case "hsv":
                color_transform = rgb_to_hsv
                mean = torch.tensor([0., 0., 0.])
                std = torch.tensor([1., 1., 1.])
            case _:
                raise ValueError(f"Color space {color_sapce} not implemented")

        return color_transform, [mean, std]
    
def color_transform_plot(color_space, img, l=5):
    match color_sapce:
        case "lab":
            new_img = tensor_lab_2_rgb(img[:l])
        case "yuv":
            new_img = tensor_yuv_2_rgb(img[:l])
        case "hsv":
            new_img = tensor_hlv_2_rgb(img[:l])
    return new_img

if __name__ == '__main__':

    from torch.utils.data import Dataset, ConcatDataset
    from color_space_convert import *
    from PIL import Image
    import os
    
    print("main")

    img_size=224
    batch_size = 5
    used_dataset = "mini_DAVIS"

    color_sapce = "lab"
    actual_color_params = color_params(color_sapce)

    dataroot = f"C:/video_colorization/data/train/{used_dataset}/"

    color_transform = actual_color_params[0]
    norm_params = actual_color_params[1]

    dataLoader = ReadData()
    dataloader = dataLoader.create_dataLoader(dataroot, img_size, color_transform, norm_params, batch_size=batch_size, shuffle=True)

    data = next(iter(dataloader))
    img, _, _, _, = create_samples(data)

    plot_images(tensor_2_img(img[:5]))

    print(f"max: {img[0].max()}")
    print(f"min: {img[0].min()}")
