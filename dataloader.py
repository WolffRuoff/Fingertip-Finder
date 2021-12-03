import torch
import numpy as np
import os
import cv2 as cv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from PIL import Image, UnidentifiedImageError

# create dataset and split into training, validation, and test sets

# img_dir is the directory where the images are located
# please modify as needed to match the folder structure
img_dir = './training_data/color/'
mask_dir = './training_data/mask/'
dep8_dir = './training_data/depth8/'

data_filenames = sorted(os.listdir(img_dir))
mask_filenames = sorted(os.listdir(mask_dir))
dep8_filenames = sorted(os.listdir(dep8_dir))

img_paths = [img_dir + p for p in data_filenames if '.jpg' in p]
mask_paths = [mask_dir + p for p in mask_filenames if '.png' in p]
dep8_paths = [dep8_dir + p for p in dep8_filenames if '.jpg' in p]

take_idx = np.arange(28000)
np.random.shuffle(take_idx)

# take_idx = take_idx[:700]
# img_paths = np.take(img_paths, take_idx)
# mask_paths = np.take(mask_paths, take_idx)
# dep8_paths = np.take(dep8_paths, take_idx)
# img_train, mask_train, dep8_train = img_paths[:500], mask_paths[:500], dep8_paths[:500]
# img_val, mask_val, dep8_val = img_paths[500:600], mask_paths[500:600], dep8_paths[500:600]
# img_test, mask_test, dep8_test = img_paths[600:], mask_paths[600:], dep8_paths[600:]

img_paths = np.take(img_paths, take_idx)
mask_paths = np.take(mask_paths, take_idx)
dep8_paths = np.take(dep8_paths, take_idx)
img_train, mask_train, dep8_train = img_paths[:25000], mask_paths[:25000], dep8_paths[:25000]
img_val, mask_val, dep8_val = img_paths[25000:26500], mask_paths[25000:26500], dep8_paths[25000:26500]
img_test, mask_test, dep8_test = img_paths[26500:], mask_paths[26500:], dep8_paths[26500:]

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

# Reference: https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
# utility function to reverse normalization (multiply by standard dev and add back mean)
def unnormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def dep8_threshold_mask(mask, dep8, threshold_val):
    # remove the far background before thesholding
    dep8 = np.where(dep8==0, 255, dep8)
    mask = np.stack((mask,)*3, axis=-1)
    threshold = cv.threshold(dep8, threshold_val, 255, cv.THRESH_BINARY)[1]
    # thresholding eliminate 
    invert = np.where(threshold==255, 0, 255)
    product = invert * mask
    # combine the color channels
    product = np.sum(product, axis=-1)
    mask = np.mean(mask, axis=-1)
    # convert to binary labels
    product = np.where(product == 0, 0, 1)
    mask_size = len(np.nonzero(mask)[0]) + len(np.nonzero(mask)[1])
    product_size = len(np.nonzero(product)[0]) + len(np.nonzero(product)[1])
    ratio = product_size/mask_size
    return invert, product, ratio, mask_size


def adaptive_threshold(mask, dep8, init_threshold=150, interval_scale=1, attempts=1):
    threshold_val = init_threshold
    invert, product, ratio, mask_size = dep8_threshold_mask(mask, dep8, threshold_val)
    # optinally can scale target ratio by mask size, larger the mask, smaller the ratio should be
    while ratio == 1:
        threshold_val -= 5*interval_scale
        invert, product, ratio, mask_size = dep8_threshold_mask(mask, dep8, threshold_val)
    while ratio == 0.9:
        threshold_val -= 2*interval_scale
        invert, product, ratio, mask_size = dep8_threshold_mask(mask, dep8, threshold_val)
    # the ratio boundary here with the scalar are tunable hyperparameters:
    # larger -- tend to include more area in the final mask, vice versa for smaller
    while ratio >= 0.40:
        threshold_val -= 1*interval_scale
        invert, product, ratio, mask_size = dep8_threshold_mask(mask, dep8, threshold_val)
    # the ratio boundary here with the scalar are tunable hyperparameters
    # if the ratio is below this threshold, restart with larger init threshold and smaller step size
    # increase to reduce cases where the final mask is too small or predominantly background area 
    if ratio < 0.15 and attempts < 2:
        invert, product, ratio, threshold_val = adaptive_threshold(mask, dep8, 200, interval_scale*0.5, attempts+1)
    #print('ratio:', ratio, 'mask_size:', mask_size, 'attempt #', attempts)
    return invert, product, ratio, threshold_val


# USE THIS AS DATASET IF NOT USING DEP8 (BOUNDING BOX ONLY)
class FingerDataset(Dataset):
    def __init__(self, data_paths, mask_paths, dep8_paths=None, img_transform=None, mask_transform=None):
        data_filenames = os.listdir(img_dir)
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

        image, mask = Image.open(img_path), Image.open(mask_path)
        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)   
        return image, mask    

# USE THIS AS DATASET IN MAIN IF USING DEP8 FOR MASK GENERATION
class FingerDataset(Dataset):
    def __init__(self, data_paths, mask_paths, dep8_paths, img_transform=None, mask_transform=None):
        data_filenames = os.listdir(img_dir)
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.dep8_paths = dep8_paths
        self.img_transform = img_transform
        # necessary if using transformations like rotation, flipping or cropping
        # in which case need to apply to both image and mask
        self.mask_transform = mask_transform
        
        data_dir = '/'.join(img_dir.rstrip('/').split('/')[:-1])
        self.dep8_mask_dir = data_dir + '/dep8_mask'
        if 'dep8_mask' not in os.listdir(data_dir):
            os.mkdir(self.dep8_mask_dir)

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        mask_path = self.mask_paths[idx]
        dep8_path = self.dep8_paths[idx]
        
        image = Image.open(img_path)
        if self.img_transform:
            image = self.img_transform(image)
           
        # if processed dep8 mask already exist, load from file
        if dep8_path.split('/')[-1] in os.listdir(self.dep8_mask_dir):
            try:
                product = Image.open(self.dep8_mask_dir + '/' + dep8_path.split('/')[-1])
                product = np.asarray(product)
                product = np.mean(product, axis=-1)
                product = np.where(product>80, 1, 0)

                if self.mask_transform:
                    product = self.mask_transform(product)
                return image, product
            except UnidentifiedImageError:
                os.remove(self.dep8_mask_dir + '/' + dep8_path.split('/')[-1])
            
        mask = np.asarray(Image.open(mask_path))
        dep8 = np.asarray(Image.open(dep8_path))
        
#         mask, dep8 = cv.imread(mask_path, 255), cv.imread(dep8_path, 255)
        # element-wise product of mask and thresholded dep8 mapping, should have only the hand portion in the mask
        invert, product, ratio, self.threshold_val = adaptive_threshold(mask, dep8)
        # the adaptive threshold processing is computationally costly, save results for future epochs
        imsave(self.dep8_mask_dir + '/' + dep8_path.split('/')[-1], product)
        if self.mask_transform:
            product = self.mask_transform(product)
        return image, product    

def main(batch_size=8, num_workers=2, resize_enabled=False):
    totensor = transforms.ToTensor()
    # smaller edge of the image will be matched to 224
    resize = transforms.Resize(224)
    # center square crop of (250, 250)
    crop = transforms.CenterCrop(250)
    # To use pretrained models, input must be normalized as follows (pytorch models documentation):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Data augmentation transforms:
    # applied randomly with probability p instead of explicitly appending transformed data to original dataset
    # which means the model will eventually be trained on the original + augmented dataset over many epochs
    grayscale = transforms.RandomGrayscale(p=0.1)
    # randomly changes the brightness, saturation, and other properties of an image
    jitter = transforms.ColorJitter(brightness=.4, hue=.2)

    if resize_enabled:
        # compose the pre-processing and augmentation transforms
        composed_aug_transforms = transforms.Compose(
            (resize, crop, grayscale, jitter, np.array, totensor, normalize))
        # test data typically should not be augmented
        no_aug_transforms = transforms.Compose((resize, crop, np.array, totensor, normalize))
        # mask also need to be converted to tensor
        mask_transform = transforms.Compose((totensor, resize, crop))
    
    else:
        # compose the pre-processing and augmentation transforms
        composed_aug_transforms = transforms.Compose(
            (grayscale, jitter, np.array, totensor, normalize))
        # test data typically should not be augmented
        no_aug_transforms = transforms.Compose((np.array, totensor, normalize))
        # mask also need to be converted to tensor
        mask_transform = totensor

    data_train = FingerDataset(
        img_train, mask_train, dep8_train, img_transform=composed_aug_transforms, mask_transform=mask_transform)
    data_val = FingerDataset(
        img_val, mask_val, dep8_val, img_transform=no_aug_transforms, mask_transform=mask_transform)
    data_test = FingerDataset(
        img_test, mask_test, dep8_test, img_transform=no_aug_transforms, mask_transform=mask_transform)

    # initialize dataloaders for the dataset
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    loader_val = DataLoader(data_val, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=True)
    loader_test = DataLoader(
        data_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    print(data_train[0][0].shape, data_train[0][1].shape)
    
    figure = plt.figure(figsize=(30, 10))
    cols, rows = 5, 2
    for i in range(1, int((cols * rows)/2 + 1)):
        img, mask = next(iter(loader_val))
        sample_idx = torch.randint(img.shape[0], size=(1,)).item()
        img, mask = img[sample_idx], mask[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.imshow(unnormalize(img).permute(1, 2, 0))
        j = cols + i
        figure.add_subplot(rows, cols, j)
        plt.title(f"Label {i}")
        plt.axis("off")
        plt.imshow((mask.permute(1, 2, 0)*255))
    plt.show()
    return loader_train, loader_val, loader_test
main()