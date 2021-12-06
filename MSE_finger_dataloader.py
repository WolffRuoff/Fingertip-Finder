import torch
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from PIL import Image

# create dataset and split into training, validation, and test sets

# img_dir is the directory where the images are located
# please modify as needed to match the folder structure
img_dir = './training_data/color/'
mask_dir = './training_data/mask/'

data_filenames = sorted(os.listdir(img_dir))
mask_filenames = sorted(os.listdir(mask_dir))
img_paths = [img_dir + p for p in data_filenames if '.jpg' in p]
mask_paths = [mask_dir + p for p in mask_filenames if '.png' in p]
take_idx = np.arange(28000)
np.random.shuffle(take_idx)

img_paths = np.take(img_paths, take_idx)
mask_paths = np.take(mask_paths, take_idx)

img_train, mask_train = img_paths[:25000], mask_paths[:25000]
img_val, mask_val = img_paths[25000:26500], mask_paths[25000:26500]
img_test, mask_test = img_paths[26500:], mask_paths[26500:]


class FingerDataset(Dataset):
    def __init__(self, data_paths, mask_paths, img_transform=None, mask_transform=None):
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transform
        # necessary if using transformations like rotation, flipping or cropping
        # in which case need to apply to both image and mask
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path)
        image = np.array(image)

        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Get the value for the mask
        obj_ids = np.unique(mask)
        # obj_ids will be [0, 127, 255]. 0 is the background which we don't want
        obj_ids = obj_ids[1:]
        
        # Grab the bounding box for the hand
        fingertip_coor = np.where(mask == obj_ids[0])
        coors = np.where(mask == obj_ids[1])
        coors = (np.concatenate([coors[0], fingertip_coor[0]]), np.concatenate([coors[1], fingertip_coor[1]]))
        ymin = np.max([np.min(coors[0])-5,0])
        ymax = np.max(coors[0])+5
        xmin = np.max([np.min(coors[1])-5,0])
        xmax = np.max(coors[1])+5

        # Crop the image and mask to the bounding box
        image = image[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]

        # Make the 255's 0 now and the coordinate 255 to be consistent with our training method
        mask[mask==obj_ids[1]] = 0
        mask[mask==obj_ids[0]] = 255

        # Add padding to the right or bottom of the image to make it a square
        larger_dim = np.max(mask.shape)
        padded_image = np.zeros((larger_dim, larger_dim, 3), dtype=np.uint8)
        padded_image[0:image.shape[0], 0:image.shape[1], :] = image
        # Make the padding on the mask a 1 value so that it can easily be removed later
        padded_mask = np.zeros((larger_dim, larger_dim))
        padded_mask[0:mask.shape[0], 0:mask.shape[1]] = mask 
        
        if self.img_transform:
            image = Image.fromarray(padded_image)
            image = self.img_transform(image)
        if self.mask_transform:
            mask = Image.fromarray(padded_mask)
            mask = self.mask_transform(mask)
        
        # This grabs the coordinate of the fingertip from the cropped mask
        fingertip_coors = np.where(mask == 1)
        
        fingertip_coor = np.rint((np.mean(fingertip_coors[1]), np.mean(fingertip_coors[0])))
        fingertip_coor = torch.tensor(fingertip_coor, dtype=torch.int32)
        return image, fingertip_coor

# pre-processing transformations
# threshold the images with opencv to reduce noise and improve generalization
# reference: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
def threshold_mask(img):
    # Get the img as grayscale
    pic = cv.imread(img, 0)

    # Convert pixels below 5 to white and above 5 to black
    th, threshed = cv.threshold(
        pic, 1, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    return threshed

def array_threshold(img):
    img[img > 0] = 1
    return img

# Reference: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
# utility function to reverse normalization (multiply by standard dev and add back mean)
def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main(batch_size=8, num_workers=2):
    # Resize and CenterCrop accepts either PIL Image or Tensor.
    # Previous attempts to normalize tensor from torchvision.io.read_image yielded incorrect outputs
    # Use Pillow to read PIL image then convert to tensor worked well
    totensor = transforms.ToTensor()
    # Resize the image to be 99x99
    resize = transforms.Resize(99)
    # # center square crop of (250, 250)
    # crop = transforms.CenterCrop(250)
    # To use pretrained models, input must be normalized as follows (pytorch models documentation):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Data augmentation transforms:
    # applied randomly with probability p instead of explicitly appending transformed data to original dataset
    # which means the model will eventually be trained on the original + augmented dataset over many epochs
    grayscale = transforms.RandomGrayscale(p=0.1)
    # randomly changes the brightness, saturation, and other properties of an image
    jitter = transforms.ColorJitter(brightness=.4, hue=.2)

    # compose the pre-processing and augmentation transforms
    composed_aug_transforms = transforms.Compose(
        (resize, grayscale, jitter, np.array, totensor, normalize))
    # test data typically should not be augmented
    no_aug_transforms = transforms.Compose((resize, np.array, totensor, normalize))
    # mask also need to be converted to tensor
    mask_transform = transforms.Compose((resize, np.array, array_threshold))

    data_train = FingerDataset(
        img_train, mask_train, img_transform=composed_aug_transforms, mask_transform=mask_transform)
    data_val = FingerDataset(
        img_val, mask_val, img_transform=no_aug_transforms, mask_transform=mask_transform)
    data_test = FingerDataset(
        img_test, mask_test, img_transform=no_aug_transforms, mask_transform=mask_transform)

    # initialize dataloaders for the dataset
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    loader_val = DataLoader(data_val, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=True)
    loader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    print(data_train[0][0].shape)
    
    figure = plt.figure(figsize=(12, 6))
    cols, rows = 3, 2
    for i in range(1, (cols * rows) + 1):
        sample_idx = torch.randint(len(data_train), size=(1,)).item()
        img, fingertip_coor = data_train[sample_idx]
        img = unnormalize(img).cpu().permute(1, 2, 0).numpy().copy()
        img = cv.circle(img, (fingertip_coor[0].item(), fingertip_coor[1].item()), 7, (255,0,0), 1)
        figure.add_subplot(rows, cols, i)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.imshow(img)
    plt.show()
    return loader_train, loader_val, loader_test
