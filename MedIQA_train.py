import torch
import numpy as np
from MedIQA_functions import get_data_loader, test, index
from backbone_network import Backbone


class Config:
    """Configuration class for hyperparameters."""
    lr = 1e-5
    batch_size = 1
    gamma = 0.75
    epochs = 30
    log_interval = 50
    step_size = 1
    weight_decay = 0.01



def one_hot_encode(info):
    c_s = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_seq = info
    target_code = torch.from_numpy(np.array([1 if i == target_seq else 0 for i in range(c_s)])).reshape((1, c_s)).to(
        device=device, dtype=torch.float32)
    return target_code


if __name__ == '__main__':
    # Configuration
    config = Config()

    # Model initialization
    model = Backbone()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Load pre-trained weights
    pre_dict = torch.load("")
    model_dict = model.state_dict()
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    # Data loaders
    train_loader = get_data_loader(random=True, batch_size=config.batch_size, split='', model=model)
    val_loader = get_data_loader(random=False, batch_size=config.batch_size, split='', model=model)

    # Optimizer, scheduler, and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    criterion = torch.nn.MSELoss()

    # Track best validation loss
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config.epochs):
        total_loss = 0
        count = 0
        output_list = []
        label_list = []

        for data1, data2, data3, data4, data5, data6, data7, target1, target2 in train_loader:
            # Move data to device
            data_list = [data1, data2, data3, data4, data5, data6, data7]
            data_list = [data.to(device) for data in data_list]
            target1, target2 = target1.to(device, dtype=torch.float32), target2.to(device)

            # One-hot encode target2
            info = one_hot_encode(target2, device=device)

            optimizer.zero_grad()

            # Compute model outputs and loss
            outputs = [model(data, info) for data in data_list]
            output = sum(outputs) / len(outputs)
            output_list.append(output.item())
            label_list.append(target1.item())

            losses = [criterion(out, target1) for out in outputs]
            loss = sum(losses) / len(losses)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            # Log progress
            if count % config.log_interval == 0:
                progress = 100 * count / len(train_loader)
                print(f"Epoch: {epoch + 1}  {progress:.2f}%  Loss: {loss.item():.6f}")

        # Evaluate training metrics
        train_loss = total_loss / count
        SRCC_train, PLCC_train, RMSE_train = index(output_list, label_list)

        # Validation
        model.eval()
        val_loss, SRCC_val, PLCC_val, RMSE_val = test(model, val_loader, device)
        model.train()

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "MedIQA_best_state_dict.pt")
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")

        # Update scheduler
        scheduler.step()

        # Print epoch results
        print("\n" + "="*80)
        print(f"Epoch: {epoch + 1}")
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Training SRCC: {SRCC_train:.4f}  PLCC: {PLCC_train:.4f}  RMSE: {RMSE_train:.4f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Validation SRCC: {SRCC_val:.4f}  PLCC: {PLCC_val:.4f}  RMSE: {RMSE_val:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.8f}")
        print("="*80 + "\n")

    # Save final model
    torch.save(model.state_dict(), "MedIQA_final_state_dict.pt")
    torch.save(model, "MedIQA_final_model.pt")