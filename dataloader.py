import torch
import numpy as np
import os
import cv2 as cv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from PIL import Image, UnidentifiedImageError

# load data file paths and split into training, validation, and test sets
# can be used for egofinger dataset or IPN dataset
# IPN dataset should be used with appropriate path, use_dep8=False, and sortkey = lambda x: (x.split('_')[-4], x.split('_')[-2], x[-10:])
def load_paths(folder_path='./training_data', use_dep8=False, sortkey=None):
    img_dir = folder_path + '/color/'
    mask_dir = folder_path + '/mask/'
    
    img_filenames = sorted(os.listdir(img_dir), key=sortkey)
    mask_filenames = sorted(os.listdir(mask_dir), key=sortkey)

    img_paths = [img_dir + p for p in img_filenames if '.jpg' in p]
    mask_paths = [mask_dir + p for p in mask_filenames if '.png' in p]

    if use_dep8:
        dep8_dir = folder_path + '/depth8/'
        dep8_filenames = sorted(os.listdir(dep8_dir), key=sortkey)
        dep8_paths = [dep8_dir + p for p in dep8_filenames if '.jpg' in p]
    
    take_idx = np.arange(len(img_paths))
    np.random.shuffle(take_idx)
    
    img_paths = np.take(img_paths, take_idx)
    mask_paths = np.take(mask_paths, take_idx)

    bounds = [int(len(img_paths)*0.8), int(len(img_paths)*0.9)]
    
    img_train, mask_train = img_paths[:bounds[0]], mask_paths[:bounds[0]]
    img_val, mask_val = img_paths[bounds[0]:bounds[1]], mask_paths[bounds[0]:bounds[1]]
    img_test, mask_test = img_paths[bounds[1]:], mask_paths[bounds[1]:]
    
    if use_dep8:
        dep8_paths = np.take(dep8_paths, take_idx)
        dep8_train, dep8_val, dep8_test = dep8_paths[:bounds[0]], dep8_paths[bounds[0]:bounds[1]], dep8_paths[bounds[1]:]
    
    if not use_dep8:
        return (img_train, img_val, img_test, mask_train, mask_val, mask_test)
    else:
        return (img_train, img_val, img_test, mask_train, mask_val, mask_test, dep8_train, dep8_val, dep8_test)

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
    threshold = cv.threshold(dep8, threshold_val, 255, cv.THRESH_BINARY)[1]
    mask = np.stack((mask,)*3, axis=-1)
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

def get_convex_hull(product):
    # Convert product to cv2 array so that we can use opencv methods
    #product = Image.fromarray(product.astype(np.uint8))
    product = (product * 255).astype(np.uint8)
    #product = cv.cvtColor(product, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(product, 1, 255, cv.THRESH_BINARY)
    cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[-2]

    hull = []
    # Combine all the contours and calculate the convex hull
    chull = cv.convexHull(np.vstack(cnts[i] for i in range(len(cnts))), False)
    hull.append(chull)

    # Draw it onto the thresh
    cv.drawContours(thresh, hull, -1, (255,255,255), thickness=cv.FILLED)
    thresh = np.array(thresh, dtype=np.integer)
    return thresh

# USE THIS AS DATASET IF NOT USING DEP8 (BOUNDING BOX ONLY)
class FingerDataset(Dataset):
    def __init__(self, data_paths, mask_paths, dep8_paths=None, img_transform=None, mask_transform=None):
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
        mask = torch.where(mask > 0, 1, 0)
        return image, mask    

# USE THIS AS DATASET IN MAIN IF USING DEP8 FOR MASK GENERATION
class FingerDataset_dep8(Dataset):
    def __init__(self, data_paths, mask_paths, dep8_paths, img_transform=None, mask_transform=None):
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.dep8_paths = dep8_paths
        self.img_transform = img_transform
        # necessary if using transformations like rotation, flipping or cropping
        # in which case need to apply to both image and mask
        self.mask_transform = mask_transform
        data_dir = '/'.join(data_paths[0].rstrip('/').split('/')[:-2])
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
        
        # mask, dep8 = cv.imread(mask_path, 255), cv.imread(dep8_path, 255)
        # element-wise product of mask and thresholded dep8 mapping, should have only the hand portion in the mask
        invert, product, ratio, self.threshold_val = adaptive_threshold(mask, dep8)

        # Get the convex hull of the product mask
        product = get_convex_hull(product)

        # the adaptive threshold processing is computationally costly, save results for future epochs
        imsave(self.dep8_mask_dir + '/' + dep8_path.split('/')[-1], product)
        if self.mask_transform:
            product = self.mask_transform(product)
        product = torch.where(product > 0, 1, 0)
        return image, product    

def create_IPN_dataset(IPN_path, composed_aug_transforms, no_aug_transforms, mask_transform):
    img_train, img_val, img_test, mask_train, mask_val, mask_test = load_paths(
        folder_path = IPN_path, use_dep8=False, sortkey=lambda x: (x.split('_')[-4], x.split('_')[-2], x[-10:]))

    data_train = FingerDataset(
        img_train, mask_train, img_transform=composed_aug_transforms, mask_transform=mask_transform)
    data_val = FingerDataset(
        img_val, mask_val, img_transform=no_aug_transforms, mask_transform=mask_transform)
    data_test = FingerDataset(
        img_test, mask_test, img_transform=no_aug_transforms, mask_transform=mask_transform)
    
    return data_train, data_val, data_test

def create_finger_dataset(finger_path, use_dep8, composed_aug_transforms, no_aug_transforms, mask_transform):
    if use_dep8:
        img_train, img_val, img_test, mask_train, mask_val, mask_test, dep8_train, dep8_val, dep8_test = load_paths(
            folder_path=finger_path, use_dep8=True, sortkey=None)

        data_train = FingerDataset_dep8(
            img_train, mask_train, dep8_train, img_transform=composed_aug_transforms, mask_transform=mask_transform)
        data_val = FingerDataset_dep8(
            img_val, mask_val, dep8_val, img_transform=no_aug_transforms, mask_transform=mask_transform)
        data_test = FingerDataset_dep8(
            img_test, mask_test, dep8_test, img_transform=no_aug_transforms, mask_transform=mask_transform)
    else:
        img_train, img_val, img_test, mask_train, mask_val, mask_test = load_paths(
            folder_path=finger_path, use_dep8=False, sortkey=None)

        data_train = FingerDataset(
            img_train, mask_train, img_transform=composed_aug_transforms, mask_transform=mask_transform)
        data_val = FingerDataset(
            img_val, mask_val, img_transform=no_aug_transforms, mask_transform=mask_transform)
        data_test = FingerDataset(
            img_test, mask_test, img_transform=no_aug_transforms, mask_transform=mask_transform)
        
    return data_train, data_val, data_test
    
# please change the finger_path and IPN_path accordingly to match your folder structure, they should contain color and mask folders
def main(batch_size=8, num_workers=2, resize_enabled=False, use_dep8=True, dataset='finger', visualize=True, 
         finger_path='./training_data', IPN_path='./training_data/IPN_Hand'):
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
        mask_transform = transforms.Compose((np.array, totensor))

    if dataset == 'IPN':           
        data_train, data_val, data_test = create_IPN_dataset(IPN_path, composed_aug_transforms, no_aug_transforms, mask_transform)
        
    elif dataset == 'finger':
        data_train, data_val, data_test = create_finger_dataset(finger_path, use_dep8, composed_aug_transforms, no_aug_transforms, mask_transform)
            
    else:
        assert dataset == 'both'
        data_train, data_val, data_test = create_IPN_dataset(IPN_path, composed_aug_transforms, no_aug_transforms, mask_transform)
        data_train_2, data_val_2, data_test_2 = create_finger_dataset(finger_path, use_dep8, composed_aug_transforms, no_aug_transforms, mask_transform)
        
        # since training dataloader below (loader_train) is instantiated with shuffle=True, don't need to shuffle here
        data_train = torch.utils.data.ConcatDataset([data_train, data_train_2])
        data_val = torch.utils.data.ConcatDataset([data_val, data_val_2])
        data_test = torch.utils.data.ConcatDataset([data_test, data_test_2])

    # initialize dataloaders for the dataset
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True)
    loader_val = DataLoader(data_val, batch_size=batch_size,
                            shuffle=True, num_workers=0, drop_last=True)
    loader_test = DataLoader(data_test, batch_size=batch_size, 
                             shuffle=True, num_workers=0, drop_last=True)

    print(data_train[0][0].shape, data_train[0][1].shape)
    
    if visualize:
        figure = plt.figure(figsize=(30, 10))
        cols, rows = 5, 2
        img_batch, mask_batch = next(iter(loader_val))
        for i in range(1, int((cols * rows)/2 + 1)):
            img, mask = img_batch[i], mask_batch[i]
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
