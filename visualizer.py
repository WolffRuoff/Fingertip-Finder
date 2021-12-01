import torch
import numpy as np
import matplotlib.pyplot as plt
import trainer
import evaluator
from dataloader import unnormalize
        
def visualize_model_output(dataloader, model, device='cuda'):        
    cols, rows = 5, 3
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