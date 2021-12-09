import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2

import evaluator
import finger_evaluator

from PIL import Image
from dataloader import unnormalize
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

totensor = transforms.ToTensor()
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_model(n):
    model = models.resnet18(pretrained=True)
    if n == 1:
        model.fc = nn.Linear(512, 480*640)
        model.load_state_dict(torch.load(
            'saved_models/resnet18_notile_full.model', map_location=torch.device(device)))
    elif n == 2:
        model.fc = nn.Linear(512, 2)
        model.load_state_dict(torch.load(
            'saved_models/resnet18_notile_full_finger_MSE.model', map_location=torch.device(device)))
    model = model.to(device)
    return model


model1 = get_model(1)
model2 = get_model(2)


def find_fingertip(img):
    if img.shape != (480, 640, 3):
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

    _, output = get_img_output(img, model1, device)

    # If no hand detected
    if output.all() == 0:
        return img, (0,0)
        
    crop_img, anchor = crop_image(img, output)
    mask_shape = crop_img.shape

    pad_img = pad_image(crop_img)
    pad_shape = pad_img.shape
    pad_img = Image.fromarray(pad_img)

    sq_img, finger_coor = get_finger_coor(pad_img, model2, device)

    ''' Uncomment to save images within the pipeline
    cv2.imwrite("Graphics/Demo/input.jpg", img)
    coor = (int(round(finger_coor[0])),int(round(finger_coor[1])))
    sq_img = cv2.resize(np.array(pad_img), (99, 99), interpolation=cv2.INTER_AREA)
    sq_write_image = cv2.circle(sq_img, coor, 4, (255, 0, 0), -1)
    cv2.imwrite(f"Graphics/Demo/model2.jpg", sq_write_image)
    '''
    # Reverse what we did to the image to get the actualy finger coordinate
    finger_coor_x = (pad_shape[0] * finger_coor[0]) / 99
    finger_coor_y = (pad_shape[1] * finger_coor[1]) / 99
    finger_prediction = (finger_coor_x + anchor[0], finger_coor_y + anchor[1])

    finger_prediction = np.rint((finger_prediction[0], finger_prediction[1]))
    finger_prediction = (int(finger_prediction[0]), int(finger_prediction[1]))

    # Add the labels to the images
    prediction_image = cv2.rectangle(
        img, (anchor[0], anchor[1]), (anchor[2], anchor[3]), (0, 255, 0), 1)
    # Uncomment to save model 1 output
    #cv2.imwrite(f"Graphics/Demo/model1.jpg", img)

    prediction_image = cv2.circle(
        prediction_image, finger_prediction, 4, (255, 0, 0), -1)
    return prediction_image, finger_prediction

# take rgb image (in numpy array form) and return model output
def get_img_output(img, model, device='cuda'):
    assert img.shape == (480, 640, 3)
    no_aug_transforms = transforms.Compose((totensor, normalize))
    img = no_aug_transforms(img)
    # adding a batch dimension (batch of one)
    img = torch.unsqueeze(img, 0)
    x_out = evaluator.get_inference_output(model, img, device)
    x_out = torch.where(x_out < 0, 1, 0).cpu()
    # reshape model output to have shape (batch_size, channel, height, width)
    x_out = x_out.reshape(1, 480, 640)
    # take image and model output out of batch dimension
    img, prediction = img[0], x_out[0]
    return (img, prediction)


def get_finger_coor(img, model, device='cuda'):
    # Perform data augmentation and get the prediction
    no_aug_transforms = transforms.Compose(
        (transforms.Resize(99), np.array, totensor, normalize))
    img = no_aug_transforms(img)
    test_img = torch.unsqueeze(img, 0)
    finger_coor = finger_evaluator.get_inference_output(
        model2, test_img, device)
    return img, (finger_coor[0, 0].item(), finger_coor[0, 1].item())


def crop_image(img, mask):
    # Grab the bounding box for the hand
    coors = np.where(mask == 1)
    ymin = np.max([np.min(coors[0])-5, 0])
    ymax = np.min([np.max(coors[0])+5, 480])
    xmin = np.max([np.min(coors[1])-5, 0])
    xmax = np.min([np.max(coors[1])+5, 640])

    # Crop the image and mask to the bounding box
    img = img[ymin:ymax, xmin:xmax]

    return img, (xmin, ymin, xmax, ymax)


def pad_image(img):
    # Add padding to the right or bottom of the image to make it a square
    larger_dim = np.max(img.shape)
    padded_image = np.zeros((larger_dim, larger_dim, 3), dtype=np.uint8)
    padded_image[0:img.shape[0], 0:img.shape[1], :] = img

    return padded_image


def array_threshold(img):
    img[img > 0] = 1
    return img


def test():
    #img, finger_prediction = find_fingertip(
    #    np.array(Image.open('training_data/color/color_img0025479.jpg')))
    img, finger_prediction = find_fingertip(
        np.array(Image.open('training_data/color/color_img0025500.jpg')))

    plt.title(f"Predicted {finger_prediction}")
    plt.axis("off")
    plt.imshow(img)
    plt.show()