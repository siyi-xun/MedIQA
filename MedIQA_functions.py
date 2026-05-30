import numpy as np
from skimage.transform import resize as sk_resize
import os
import pickle
import tifffile as tf
import cv2
import pydicom
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.stats import spearmanr


def one_hot_encode(info):
    c_s = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_seq = info
    target_code = torch.from_numpy(np.array([1 if i == target_seq else 0 for i in range(c_s)])).reshape((1, c_s)).to(
        device=device, dtype=torch.float32)
    return target_code


# Helper function to read image based on file type
def read_image(path):
    if '.dcm' in path:
        ds = pydicom.dcmread(path)
        return ds.pixel_array
    elif '.jpg' in path or '.jpeg' in path:
        return cv2.imread(path)
    elif '.tif' in path:
        return tf.imread(path)
    raise ValueError(f"Unsupported file type: {path}")


# Helper function to process and resize image
def process_image(img):
    img_resized = sk_resize(img, (224, 224, 3), order=2)
    return torch.FloatTensor(np.transpose(img_resized, (2, 0, 1)))


class MedDataset(Dataset):

    def __init__(self, split, model=""):
        super().__init__()
        self.split = split
        with open("../data/"+split+"_data.txt", "rb") as fp:#data path, format: 'txt'
            self.file_list = pickle.load(fp)

        with open("../data/"+split+"_labels.txt", "rb") as fp:#labels, format: 'txt'
            self.labels = pickle.load(fp)

        self.size = len(self.file_list)
        self.model_name = model

    def __len__(self):
        return len(self.file_list)
    def __getlist__(self):
        return self.file_list

    def __getitem__(self, index):
        fpath = self.file_list[index]
        a = fpath

        if os.path.isdir(a):
            filenames = sorted(os.listdir(a))
            num_files = len(filenames)

            if num_files >= 7:
                # Select evenly spaced indices
                space = num_files // 2
                space2 = num_files // 6
                indices = [0, space - 2 * space2, space - space2, space, space + space2, space + 2 * space2, num_files - 1]
                paths = [os.path.join(a, filenames[i]) for i in indices]
            else:
                # Pad with repeated files if fewer than 7
                paths = [os.path.join(a, f) for f in filenames]
                paths = (paths * (7 // len(paths) + 1))[:7]  # Repeat to ensure at least 7 paths

            # Read and process images
            images = [process_image(read_image(path)) for path in paths]

        elif '.dcm' in a or '.jpg' in a or '.jpeg' in a or '.tif' in a:
            img = read_image(a)
            images = [process_image(img)] * 7  # Replicate single image 7 times
        else:
            raise ValueError(f"Invalid input: {a}")

        # Assign to individual variables if needed
        image1, image2, image3, image4, image5, image6, image7 = images

        label = self.labels[index]
        label1 = label[0]  # score
        label2 = label[1]  # info

        return image1, image2, image3, image4, image5, image6, image7, label1, label2


def get_data_loader(random, batch_size, split, model):
    set = MedDataset(split, model)
    loader = DataLoader(set, batch_size=batch_size, shuffle=random, num_workers=4)
    return loader


def index(X,Y):
    SRCC = spearmanr(X, Y)
    PLCC = np.corrcoef(X, Y)
    PLCC_1 = PLCC[0]
    PLCC_2 = PLCC_1[1]

    sum_mean = 0
    for i in range(len(X)):
        sum_mean += (X[i] - Y[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(X))
    RMSE = sum_erro
    return SRCC[0], PLCC_2, RMSE


def test(model, test_loader, device):
    model.eval()
    criterion_reg = torch.nn.MSELoss()

    count = 0
    current_loss = 0

    label_list = []
    output_list = []

    with torch.no_grad():
        for data1, data2, data3, data4, data5, data6, data7, target1, target2 in test_loader:
            data_list = [data1, data2, data3, data4, data5, data6, data7]
            data_list = [data.to(device) for data in data_list]
            target1, target2 = target1.to(device), target2.to(device)
            info = one_hot_encode(target2)

            # Process model outputs in a loop
            outputs = [model(data, info).to(torch.float32) for data in data_list]
            output = sum(outputs) / len(outputs)
            output_list.append(output.item())

            target1 = target1.to(torch.float32)
            label_list.append(target1.item())

            # Compute losses in a loop
            losses = [criterion_reg(out, target1) for out in outputs]
            loss = sum(losses) / len(losses)

            count += 1
            current_loss += loss.item()


    current_loss /= len(test_loader)

    SRCC, PLCC, RMSE = index(output_list, label_list)


    return current_loss, SRCC, PLCC, RMSE





