import torch
import numpy as np
import os
import gc
import torch.nn as nn
import finger_evaluator as evaluator

def get_acc_fingertip(predictions, labels):
    return torch.sum(torch.round(predictions) == labels)/(labels.shape[0])

def get_acc_fingertip_2(predictions, labels, precision):
    return torch.sum(torch.round(predictions)-precision < labels < torch.round(predictions)+precision)/(labels.shape[0])

# train for the regression task of estimating the fingertip coordinates (x is distance from top, y is distance from left)
def train(model, loader_train, loader_val, lr=1e-4, num_epochs=10, device='cpu', patience=5, evaluation_interval=None):
    # initialize lists to store logs of the validation loss and validation accuracy
    val_loss_hist = []
    val_acc_hist = []

    # initialize optimizer with specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_fn = nn.MSELoss()

    # early-stopping parameters
    param_hist = []
    best_n_loss = None
    current_patience = patience
    # the number of training steps between evaluations
    stop_early = False

    for e in range(num_epochs):
        # set model to training mode
        model.train()
        # current batch index
        batch_num, num_batches = 0, len(loader_train)
        batch_acc_train = []
        batch_loss_train = []

        # Training pass
        for X_batch, y_batch in loader_train:

            # torch tensor can be loaded to GPU, when applicable
            X_batch, y_batch = X_batch.float().to(device), y_batch.to(device)

            # reset gradients for the optimizer, need to be done each training step
            optimizer.zero_grad()
            
            output = model(X_batch)
            
            batch_loss = loss_fn(output, y_batch.float())
            # compute the gradients and take optimization step
            batch_loss.backward()
            optimizer.step()

            batch_acc = get_acc_fingertip(output, y_batch)

            batch_acc_train.append(batch_acc.item())
            batch_loss_train.append(batch_loss.item())
            
            # running average
            avg_train_acc = np.mean(batch_acc_train)
            avg_train_loss = np.mean(batch_loss_train)

            print('Training epoch %d batch %d/%d, train loss = %f, train acc = %f'
                  % (e+1, batch_num+1, num_batches, avg_train_loss, avg_train_acc), end='\r')

            batch_num += 1

            if batch_num % 20 == 0:
                del X_batch
                del y_batch
                torch.cuda.empty_cache()
                gc.collect()

            if not evaluation_interval:
                evaluation_interval = num_batches//2

            # evaluate on validation set every 100 epochs, invoke early-stopping as needed (with patience)
            if batch_num % evaluation_interval == 0 or batch_num == num_batches:

                # evaluate the model
                print()
                total_loss_val, total_acc_val = evaluator.evaluate_model_fingertip(model, loader_val, loss_fn, device)

                print('validation metrics at epoch %d batch %d: val loss = %f, val acc = %f'
                      % (e+1, batch_num, total_loss_val, total_acc_val))

                val_loss_hist.append(total_loss_val)
                val_acc_hist.append(total_acc_val)

                # early stopping with patience
                save_path = 'epoch_%d_batch_%d_fingertip.model' % (e, batch_num)
                torch.save(model.state_dict(), save_path)
                param_hist.append(save_path)
                # only need to keep weights needed for earlystopping
                if len(param_hist) > patience+1:
                    del_path = param_hist.pop(0)
                    os.remove(del_path)  # delete unnecessary state dicts
                if best_n_loss and total_loss_val >= best_n_loss:
                    current_patience -= 1
                    print('current_patience = %d' % current_patience)
                    if current_patience == 0:
                        print('\nstopping early after no validation accuracy improvement in %d steps'
                              % (patience * evaluation_interval))
                        best_weights_path = param_hist[-(patience+1)]
                        # restore to last best weights when stopping early
                        model.load_state_dict(torch.load(best_weights_path))
                        stop_early = True
                        break

                # if performance improves, reset patience and best accuracy
                else:
                    current_patience = patience
                    best_n_loss = total_loss_val

        if stop_early:
            break

        # get epoch-wide training metrics
        epoch_loss_train = np.mean(batch_loss_train)
        epoch_acc_train = np.mean(batch_acc_train)

        print('='*80+'\nEpoch %d/%d train loss = %f, train acc = %f, val loss = %f, val acc = %f'
              % (e+1, num_epochs, epoch_loss_train, epoch_acc_train, total_loss_val, total_acc_val))

    if device == 'cuda':
        torch.cuda.empty_cache()  # free gpu memory if loaded to cuda
    else:
        gc.collect()  # free memory if not using cuda

    # remove cached weights after stopping and loading best weights (if applicable)
    cached_weight_paths = [f for f in os.listdir(
        '.') if ('epoch' in f and 'fingertip.model' in f)]
    for p in cached_weight_paths:
        os.remove(p)

    return (val_loss_hist, val_acc_hist)
