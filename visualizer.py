import torch
import numpy as np
import matplotlib.pyplot as plt

import trainer
import evaluator

TILE_HEIGHT = 80
TILE_WIDTH = 80

def plot_tiles(img, num_tiles_v, num_tiles_h, channel=None):
    if channel != None:
        img = img[channel]      
    v, h = num_tiles_v, num_tiles_h 
    if v == h == 1:
        print(img.shape)
        plt.imshow(np.absolute(img.squeeze()))
        plt.axis('off')
        plt.show()
    else:
        fig, axs = plt.subplots(v, h) 
        for i in range(v):
            for j in range(h):
                # absolute is used because plt.imshow has odd habit of showing -0.0 as bright blocks
                axs[i,j].imshow(np.absolute(img[i*h+j]))
                axs[i,j].axis('off')
        plt.subplots_adjust(wspace=0.05, hspace=0.05)

def visualize_model_output(dataloader, model, num, TILE_HEIGHT=TILE_HEIGHT, TILE_WIDTH=TILE_WIDTH, device='cuda'):
    while num > 0:
        x, y = next(iter(dataloader))
        print(x.shape, y.shape)
        batchsize = x.shape[0]
        x_out = evaluator.get_inference_output(model, x.reshape(-1, x.shape[1], x.shape[3], x.shape[4]), device)
        x_out = torch.where(x_out > 0, 1, 0).cpu()
        y_reshape = y[:,0,:,:,:].reshape(-1, y.shape[-2] * y.shape[-1]).cpu()
        print(f"Accuracy = {trainer.get_acc(x_out, y_reshape).item() * 100}%")
        print(f"Precision = {trainer.get_precision(x_out, y_reshape).item() * 100}%")
        print(f"Recall = {trainer.get_recall(x_out, y_reshape).item() * 100}%")
        x_out_reshaped = x_out.reshape(-1, y.shape[2], y.shape[3], y.shape[4]).cpu()

        v, h = int(480/TILE_HEIGHT), int(640/TILE_WIDTH)
        for i in range(min(num, batchsize)):
            plot_tiles(x[i], v, h, 0)
            plot_tiles(x_out_reshaped[i], v, h)
            plot_tiles(y[i], v, h, 0)

        num -= min(num, batchsize)