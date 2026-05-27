import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib

from dataclasses import dataclass, field
from typing import List, Tuple, Sequence



#train dnn with the generated data
def backprop(dnn_list: Sequence[torch.nn.Module], x: torch.Tensor, y: torch.Tensor, num_epochs: int = 5000,
              learning_rate: float = 0.001, delta_factor: float = 0.9, lambda_reg: float = 0.01, config_name: str = "", scaler=None, batch_size: int = None) -> Tuple[List[torch.nn.Module], List[float], List[List[float]]]:
    dnn_list = list(dnn_list) if isinstance(dnn_list, (list, tuple)) else [dnn_list]

    criterion = torch.nn.MSELoss()
    final_losses = [0.0] * len(dnn_list)
    loss_histories = [[] for _ in range(len(dnn_list))]

    # Optimizers
    optimizers = [torch.optim.Adam(filter(lambda p: p.requires_grad, d.parameters()), lr=learning_rate) for d in dnn_list]

    # Schedulers: ReduceLROnPlateau
    schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.75, patience=50, verbose=False) for opt in optimizers]

    # Early Stopping variables
    best_losses = [float('inf')] * len(dnn_list)
    patience_counters = [0] * len(dnn_list)
    patience_limit = 250
    active_models = [True] * len(dnn_list)
    best_weights = [None] * len(dnn_list)

    # DataLoader setup
    if batch_size is None:
        batch_size = x.size(0)

    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        if not any(active_models):
            print(f"All models stopped early at epoch {epoch}")
            break

        # Initialize epoch losses
        current_epoch_losses = [0.0] * len(dnn_list)

        # Batch training
        for batch_x, batch_y in dataloader:
            for i, dnn in enumerate(dnn_list):
                if not active_models[i]:
                    continue

                dnn.train()
                optimizers[i].zero_grad()

                outputs = dnn(batch_x)
                loss = criterion(outputs, batch_y)

                l2_reg = torch.tensor(0.0, device=batch_x.device)
                for param in dnn.parameters():
                    l2_reg = l2_reg + torch.norm(param, p=2) ** 2
                loss = loss + lambda_reg * l2_reg

                loss.backward()
                optimizers[i].step()

                # Accumulate loss (weighted by batch size)
                current_epoch_losses[i] += loss.item() * batch_x.size(0)

        # Average losses over all batches
        for i in range(len(dnn_list)):
            if active_models[i]:
                current_epoch_losses[i] /= x.size(0)
                loss_histories[i].append(current_epoch_losses[i])

        # Update schedulers and check early stopping
        for i, dnn in enumerate(dnn_list):
            if not active_models[i]:
                continue

            loss_val = current_epoch_losses[i]
            final_losses[i] = loss_val
            schedulers[i].step(loss_val)

            if loss_val < best_losses[i]:
                best_losses[i] = loss_val
                patience_counters[i] = 0
                best_weights[i] = dnn.state_dict()
            else:
                patience_counters[i] += 1

            if patience_counters[i] >= patience_limit:
                active_models[i] = False
                print(f"Model {i} stopped early at epoch {epoch}")
                if best_weights[i] is not None:
                    dnn.load_state_dict(best_weights[i])

            if (epoch + 1) % 10 == 0:
                for module in dnn.modules():
                    if hasattr(module, 'delta'):
                        module.delta *= delta_factor

            if (epoch + 1) % 100 == 0:
                current_lr = optimizers[i].param_groups[0]['lr']
                print(f'{dnn.name} ({config_name}), epoch={epoch+1}, loss={current_epoch_losses[i]:.4f}, lr={current_lr:.6f}')

        #if epoch <= 10 or epoch % 100 == 0 or epoch == num_epochs - 1 or not any(active_models):
            #plotNN_list(dnn_list, x, y, n=epoch, config_name=config_name, scaler=scaler)

    # Restore best weights for any models that finished without early stopping triggering (optional, but good practice)
    for i, dnn in enumerate(dnn_list):
        if active_models[i] and best_weights[i] is not None:
             # Check if current loss is worse than best loss?
             # Usually we want the best model found during training.
             if final_losses[i] > best_losses[i]:
                 dnn.load_state_dict(best_weights[i])
                 final_losses[i] = best_losses[i]

    # plot_loss_history(loss_histories, dnn_list, config_name)

    return dnn_list, final_losses, loss_histories