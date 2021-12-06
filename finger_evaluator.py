import torch
import numpy as np
import gc
import finger_trainer

# forward pass for inference and evaluation
def get_inference_output(model, X_in, device):
    model.eval()
    X_in = X_in.to(device)
    with torch.no_grad():
        output = model(X_in.float())
    return output

def evaluate_model_fingertip(model, loader_val, loss_fn, device):
    acc_val = []
    loss_val = []
    err_x, err_y, err_pred = np.array([]), np.array([]), np.array([])
    val_batch_num, val_num_batches = 0, len(loader_val)
    # set model to eval mode when evaluating on validation set
    model.eval()
    for X_val, y_val in loader_val:
        X_val, y_val = X_val.to(device), y_val.to(device)
        with torch.no_grad():
            output = get_inference_output(model, X_val, device)
            batch_loss = loss_fn(output, y_val.float())
            
        batch_acc = finger_trainer.get_acc_fingertip(output, y_val)
        
        acc_val.append(batch_acc.item())
        loss_val.append(batch_loss.item())
        
        print('evaluating batch %d/%d'%(val_batch_num+1, val_num_batches), end='\r')
        val_batch_num += 1  
        
    del X_val
    del y_val
    torch.cuda.empty_cache()
    gc.collect()    

    # get validation metrics for this epoch
    total_acc_val = np.mean(acc_val)
    total_loss_val = np.mean(loss_val)
    # set model back to training mode after finishing evaluation
    model.train()
    
    return (total_loss_val, total_acc_val)