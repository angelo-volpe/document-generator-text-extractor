from torch.utils.data import Dataset
import os
from PIL import Image


# Custom dataset for loading pairs of noisy and clean images
class BinaryDenoisingDataset(Dataset):
    def __init__(
        self,
        noisy_dir,
        clean_dir,
        transform=None,
        convert_to_binary=True,
        threshold=0.5,
    ):
        """
        Args:
            noisy_dir (string): Directory with noisy images
            clean_dir (string): Directory with clean/ground truth images
            transform (callable, optional): Optional transform to be applied
            convert_to_binary (bool): Whether to convert clean images to binary
            threshold (float): Threshold for binarization (0-1)
        """
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        self.convert_to_binary = convert_to_binary
        self.threshold = threshold

        self.noisy_images = sorted(
            [
                f
                for f in os.listdir(noisy_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            ]
        )
        self.clean_images = sorted(
            [
                f
                for f in os.listdir(clean_dir)
                if f.endswith(".png") or f.endswith(".jpg")
            ]
        )

        # Ensure corresponding pairs
        assert len(self.noisy_images) == len(
            self.clean_images
        ), "Number of noisy and clean images don't match"

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_path = os.path.join(self.noisy_dir, self.noisy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.clean_images[idx])

        # Load images
        noisy_image = Image.open(noisy_path).convert("RGB")

        # Convert clean image to grayscale for binary output
        clean_image = Image.open(clean_path).convert("L")  # 'L' is grayscale

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

            # Binarize the clean image if requested
            if self.convert_to_binary:
                clean_image = (clean_image < self.threshold).float()

        return noisy_image, clean_image
