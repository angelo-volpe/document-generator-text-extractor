import torch
from torch import nn, optim
import mlflow


# Binary Cross Entropy loss is often better for binary outputs
def binary_loss(outputs, targets):
    return nn.BCELoss()(outputs, targets)


# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Binary Cross Entropy Loss for binary classification task
    criterion = binary_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []

    mlflow.set_tracking_uri("http://localhost:5000")  # Set your MLflow tracking URI
    
    with mlflow.start_run():
        params = {
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "loss_function": criterion.__name__,
            "optimizer": optimizer.__class__.__name__,
        }
        mlflow.log_params(params)

        for epoch in range(num_epochs):
            # Training
            model.train()
            running_loss = 0.0
            
            for batch, (noisy_imgs, clean_imgs) in enumerate(train_loader):
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                
                # Forward pass
                outputs = model(noisy_imgs)
                loss = criterion(outputs, clean_imgs)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()

                #if batch % 100 == 0:
                #    loss, current = loss.item(), batch
                #    mlflow.log_metric("loss", f"{loss:3f}", step=(batch // 10))
                #    print(f"loss: {loss:3f} [{current} / {len(train_loader)}]")
            
            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for noisy_imgs, clean_imgs in val_loader:
                    noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
                    outputs = model(noisy_imgs)
                    loss = criterion(outputs, clean_imgs)
                    val_loss += loss.item()
            
            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
            mlflow.log_metric("train_loss", f"{epoch_train_loss:3f}", step=epoch)
            mlflow.log_metric("val_loss", f"{epoch_val_loss:3f}", step=epoch)

            mlflow.pytorch.log_model(model, f"model_epoch_{epoch}")
        
    return model, train_losses, val_losses
