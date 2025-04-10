def train_model(model, data_loader, optimizer, device, num_epochs):
    """
    Train the Faster R-CNN model
    """
    # Set model to training mode
    model.train()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0
        
        for images, targets in data_loader:
            # Move images and targets to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            optimizer.step()
            
            # Update epoch loss
            epoch_loss += losses.item()
        
        # Print epoch loss
        print(f"Loss: {epoch_loss/len(data_loader):.4f}")
    
    return model