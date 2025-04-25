import torch
from matplotlib import pyplot as plt
import numpy as np
from train import binary_loss


# Visualize binary denoising results
def visualize_binary_results(model, test_loader, threshold=0.5, num_samples=5):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (noisy_imgs, clean_imgs) in enumerate(test_loader):
            if i >= num_samples:
                break

            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

            # Get continuous outputs
            continuous_outputs = model(noisy_imgs)

            # Get binary outputs (thresholded)
            binary_outputs = (continuous_outputs > threshold).float()

            # Convert to CPU and numpy for visualization
            noisy_np = noisy_imgs.cpu().numpy().transpose(0, 2, 3, 1)  # [B,H,W,C]
            clean_np = clean_imgs.cpu().numpy()  # [B,1,H,W]
            continuous_np = continuous_outputs.cpu().numpy()  # [B,1,H,W]
            binary_np = binary_outputs.cpu().numpy()  # [B,1,H,W]

            # Plot results
            plt.figure(figsize=(15, 5))

            for j in range(noisy_imgs.size(0)):
                # Noisy Input
                plt.subplot(4, noisy_imgs.size(0), j + 1)
                plt.imshow(np.clip(noisy_np[j], 0, 1))
                plt.title("Noisy")
                plt.axis("off")

                # Continuous Output
                plt.subplot(4, noisy_imgs.size(0), noisy_imgs.size(0) + j + 1)
                plt.imshow(1 - continuous_np[j, 0], cmap="gray", vmin=0, vmax=1)
                plt.title("Continuous")
                plt.axis("off")

                # Binary Output
                plt.subplot(4, noisy_imgs.size(0), 2 * noisy_imgs.size(0) + j + 1)
                plt.imshow(binary_np[j, 0], cmap="binary", vmin=0, vmax=1)
                plt.title("Binary")
                plt.axis("off")

                # Ground Truth
                plt.subplot(4, noisy_imgs.size(0), 3 * noisy_imgs.size(0) + j + 1)
                plt.imshow(clean_np[j, 0], cmap="binary", vmin=0, vmax=1)
                plt.title("Ground Truth")
                plt.axis("off")

            plt.tight_layout()
            # plt.savefig(f'binary_denoising_results_{i}.png')
            plt.show()


# Calculate binary accuracy
def binary_accuracy(outputs, targets, threshold=0.5):
    binary_outputs = (outputs > threshold).float()
    correct = (binary_outputs == targets).float().sum()
    total = targets.numel()  # Total number of elements
    return (correct / total).item()


# Evaluate model on test set
def evaluate_model(model, test_loader, threshold=0.5):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.mps.is_available() else "cpu"
    )
    print(f"using_device: {device}")
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_accuracy = 0.0

    criterion = binary_loss

    with torch.no_grad():
        for noisy_imgs, clean_imgs in test_loader:
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)

            # Calculate loss
            loss = criterion(outputs, clean_imgs)
            test_loss += loss.item()

            # Calculate accuracy
            accuracy = binary_accuracy(outputs, clean_imgs, threshold)
            test_accuracy += accuracy

    avg_test_loss = test_loss / len(test_loader)
    avg_test_accuracy = test_accuracy / len(test_loader)

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_accuracy:.4f}")
    return avg_test_loss, avg_test_accuracy
