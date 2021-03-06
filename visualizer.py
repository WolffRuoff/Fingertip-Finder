import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import trainer
import finger_trainer
import evaluator
import finger_evaluator
from dataloader import unnormalize
from torchvision import transforms
from PIL import Image
        
def visualize_model_output(dataloader, model, device='cuda'):        
    cols, rows = 8, 3
    figure = plt.figure(figsize=(cols*3, rows*3))
    
    # evaluate a single batch and reshape for display
    x, y = next(iter(dataloader))
    batchsize = x.shape[0]
    x_out = evaluator.get_inference_output(model, x, device)
    x_out = torch.where(x_out > 0, 1, 0).cpu()
    # flatten y for computing evaluation metrics
    y_flatten = y[:,0,:,:].reshape(-1, y.shape[-2] * y.shape[-1]).cpu()
    print(f"Accuracy = {trainer.get_acc(x_out, y_flatten).item() * 100}%")
    print(f"Precision = {trainer.get_precision(x_out, y_flatten).item() * 100}%")
    print(f"Recall = {trainer.get_recall(x_out, y_flatten).item() * 100}%")
    # reshape model output to have shape (batch_size, channel, height, width)
    x_out = x_out.reshape(-1, y.shape[1], y.shape[2], y.shape[3]).cpu()
    for i in range(1, cols+1):
        sample_idx = torch.randint(x.shape[0], size=(1,)).item()
        img, prediction, mask = x[i], x_out[i], y[i]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.imshow(unnormalize(img).permute(1, 2, 0))
        j = cols + i
        figure.add_subplot(rows, cols, j)
        plt.title(f"Predicted {i}")
        plt.axis("off")
        k = cols * 2 + i
        plt.imshow(prediction.permute(1, 2, 0))
        figure.add_subplot(rows, cols, k)
        plt.title(f"Actual {i}")
        plt.axis("off")
        plt.imshow(mask.permute(1, 2, 0))
    plt.show()

def visualize_finger_model_output(dataloader, model, device='cuda'):        
    cols, rows = 8, 3
    figure = plt.figure(figsize=(cols*3, rows*3))
    
    # evaluate a single batch and reshape for display
    x, y = next(iter(dataloader))
    batchsize = x.shape[0]
    x_out = evaluator.get_inference_output(model, x, device)
    #x_out = torch.where(x_out > 0, 1, 0).cpu()
    x_out = x_out.cpu()
    print(f"Accuracy = {finger_trainer.get_acc_fingertip(x_out, y).item() * 100}%")
    # reshape model output to have shape (batch_size, channel, height, width)
    #x_out = x_out.reshape(-1, y.shape[1], y.shape[2], y.shape[3]).cpu()
    for i in range(1, cols+1):
        sample_idx = torch.randint(x.shape[0], size=(1,)).item()
        img, finger_prediction, finger_label = x[i], x_out[i], y[i]
        img = unnormalize(img).cpu().permute(1, 2, 0).numpy().copy()
        figure.add_subplot(rows, cols, i)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.imshow(img)
        j = cols + i
        figure.add_subplot(rows, cols, j)
        finger_prediction = np.rint((finger_prediction[0].item(), finger_prediction[1].item()))
        finger_prediction = (int(finger_prediction[0]), int(finger_prediction[1]))
        plt.title(f"Predicted {i} Coor {finger_prediction}")
        plt.axis("off")
        prediction_image = cv.circle(img, finger_prediction, 7, (255,0,0), 1)
        plt.imshow(prediction_image)
        k = cols * 2 + i
        figure.add_subplot(rows, cols, k)
        finger_label = (finger_label[0].item(), finger_label[1].item())
        plt.title(f"Actual {i} Coor {finger_label}")
        plt.axis("off")
        label_image = cv.circle(img, finger_label, 7, (255,0,0), 1)
        plt.imshow(label_image)
    plt.show()
    
# visualize model outputs of images from test_dir
def visualize_test_data(test_dir, model, device='cuda'):  
    test_filenames = sorted(os.listdir(test_dir))
    test_paths = [test_dir + p for p in test_filenames if '.jpg' in p]
    
    totensor = transforms.ToTensor()
    resize = transforms.Resize((480, 640))
    
    # To use pretrained models, input must be normalized as follows (pytorch models documentation):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    no_aug_transforms = transforms.Compose((resize, np.array, totensor, normalize))
    
    cols, rows = 8, 3
    figure = plt.figure(figsize=(cols*3, rows*3))

    for i in range(1, cols+1):
        x = Image.open(test_paths[i])
        x = no_aug_transforms(x)
        x = torch.unsqueeze(x, 0)
        
        x_out = evaluator.get_inference_output(model, x, device)
        x_out = torch.where(x_out > 0, 1, 0).cpu()
        # reshape model output to have shape (batch_size, channel, height, width)
        x_out = x_out.reshape(1, x.shape[2], x.shape[3]).cpu()
        
        img, prediction = x[0], x_out[0]
        figure.add_subplot(rows, cols, i)
        plt.title(f"Image {i}")
        plt.axis("off")
        plt.imshow(unnormalize(img).permute(1, 2, 0))
        j = cols + i
        figure.add_subplot(rows, cols, j)
        plt.title(f"Predicted {i}")
        plt.axis("off")
        plt.imshow(prediction)
    plt.show()
    
# take rgb image (in numpy array form) and return model output
def get_img_output(img, model, device='cuda'):
    assert img.shape == (480, 640, 3)
    totensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    
    
