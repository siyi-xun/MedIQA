import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from skimage.transform import resize as sk_resize
import os
import random
import tifffile as tf
import cv2
import pydicom

class MedDataset(Dataset):
    """Medical image dataset, loads image paths from a txt file, supports .tif, .dcm, .jpg, .jpeg, and directories."""
    def __init__(self, split: str):
        super().__init__()
        self.split = split

        # Load image paths from txt file
        try:
            with open(f"./{split}_data.txt", "r") as fp:
                self.file_list = [line.strip() for line in fp if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Image paths file './{split}_data.txt' not found.")

        self.size = len(self.file_list)

    def __len__(self) -> int:
        """Return the size of the dataset."""
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        """Fetch an image by index."""
        file_path = self.file_list[index]

        try:
            if file_path.endswith('.tif'):
                img = tf.imread(file_path)
            elif os.path.isdir(file_path):
                filenames = os.listdir(file_path)
                if not filenames:
                    raise FileNotFoundError(f"No files found in directory: {file_path}")
                length = len(filenames)
                medium = length // 2
                space = length // 3
                number = random.randint((medium-space), (medium+space))
                filepath = os.path.join(file_path, filenames[number])
                ds = pydicom.dcmread(filepath)
                img = ds.pixel_array
            elif file_path.endswith('.dcm'):
                ds = pydicom.dcmread(file_path)
                img = ds.pixel_array
            elif file_path.endswith(('.jpg', '.jpeg')):
                img = cv2.imread(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Resize and convert to tensor
            img = sk_resize(img, (256, 256, 3), order=1, anti_aliasing=True)
            image = torch.FloatTensor(np.transpose(img, (2, 0, 1)))

            return image

        except Exception as e:
            raise RuntimeError(f"Error processing file {file_path}: {str(e)}")

def get_data_loader(shuffle: bool, batch_size: int, split: str, num_workers: int = os.cpu_count() // 2) -> DataLoader:
    """Create a data loader for the dataset."""
    dataset = MedDataset(split)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader

def test(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> list:
    """Run inference with the model and return classification results."""
    model.eval()
    output_list = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data).to(torch.float32)
            pred = output.argmax(dim=1).cpu().numpy()
            output_list.extend(pred.tolist())

    return output_list

def main():
    """Main function to load models and perform inference."""
    # Configuration
    batch_size = 1
    split = "" #dataset name
    model_paths = {
        "modality": "",
        "position": "",
        "type": ""
    }

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data loader
    test_loader = get_data_loader(
        shuffle=False,
        batch_size=batch_size,
        split=split
    )

    # Load and test models
    results = {}
    for model_name, model_path in model_paths.items():
        try:
            model = torch.load(model_path, map_location=device)
            results[model_name] = test(model, test_loader, device)
        except Exception as e:
            raise RuntimeError(f"Error processing model {model_path}: {str(e)}")

    # Print results
    for model_name, preds in results.items():
        print(f"{model_name} predictions: {preds}")

if __name__ == "__main__":
    main()