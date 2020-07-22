import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd
import os
import re
from sklearn import metrics

from PIL import Image
import pandas as pd
from glob import glob
from tqdm.notebook import tqdm


def process_meta_df(df_path, train_path='image/train', test_path='image/test'):
    meta_data = pd.read_csv(df_path).set_index('image_id')
    meta_data['train_test'] = None
    meta_data['path'] = None
    
    for dir in os.listdir(train_path):
        for img in glob(os.path.join(train_path,dir,'*')):
            file_name = img.split('/')[-1]
            img_id = re.search(r'(\w+)\.',file_name).group()[:-1]
            meta_data.loc[img_id,'train_test'] = 'train'
            meta_data.loc[img_id,'path'] = img
    
    for dir in os.listdir(test_path):
        for img in glob(os.path.join(test_path,dir,'*')):
            file_name = img.split('/')[-1]
            img_id = re.search(r'(\w+)\.',file_name).group()[:-1]
            meta_data.loc[img_id,'train_test'] = 'test'
            meta_data.loc[img_id,'path'] = img
    return meta_data


class MultiItemDataset(Dataset):
    def __init__(self, meta_data, train_test, label2idx, max_images=None, transform=None):
        self.meta_data = meta_data[meta_data.train_test==train_test]
        self.image_paths = self.meta_data.reset_index().groupby('item_id')['image_id'].apply(list)
        self.transform = transform
        self.label2idx = label2idx
        if max_images is None:
            self.max_images = self.image_paths.apply(len).max()
        else:
            self.max_images = max_images
        self.max_channels = self.max_images*3

    def __len__(self):
        return len(self.meta_data.item_id.unique())

    def __getitem__(self, idx):
        image_ids = self.image_paths.iloc[idx]
        item_id = self.image_paths.index[idx]
        sample = {'images':torch.Tensor()}
        label = self.meta_data[self.meta_data.item_id==item_id].iloc[0,:]['label']
        sample['target'] = self.label2idx[label]
        
        for image_id in image_ids:
            image = self.get_image(image_id)
            sample['images'] = torch.cat((sample['images'], image), dim=0)
        sample['image_channels'] = sample['images'].size(0)
        sample['num_images'] = sample['images'].size(0) // 3
        
        pad_dims = self.max_channels - sample['images'].size(0)
        pads = torch.zeros(pad_dims,sample['images'].size(1),sample['images'].size(2))
        sample['images'] = torch.cat((sample['images'], pads), dim=0)
        return sample

    def get_image(self, image_id):
        image_path = self.meta_data.loc[image_id,'path']
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image


def resize(original_image, target_size=(500, 500), bg_color=(255, 255, 255)):
    """resize image by custom height and width"""

    original_image.thumbnail(target_size, Image.ANTIALIAS)
    resized_image = Image.new("RGB", target_size, bg_color)  # blank canvas

    # paste imge into canvas
    resized_image.paste(
        original_image,
        (
            int((target_size[0] - original_image.size[0]) / 2),
            int((target_size[1] - original_image.size[1]) / 2),
        ),
    )
    return resized_image


def train_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs,
    device,
    save_model=False,
    ):
    """Main generic training loop. Includes test loss recording
       returns: train losses and test losses"""

    train_losses, test_losses = [], []

    best_loss = np.inf

    for ep in tqdm(range(epochs)):
        train_batch_loss = []
        model.train()
        for batch in train_loader:
            batch = {key:val.to(device) for key, val in batch.items()}
            optimizer.zero_grad()

            outputs = model(batch['images']).squeeze(-1).squeeze(-1)
            loss = criterion(outputs, batch['target'])

            loss.backward()
            optimizer.step()

            train_batch_loss.append(loss.item())
        train_mean_batch_loss = np.mean(train_batch_loss)

        test_batch_loss = []
        model.eval()
        for batch in test_loader:
            batch = {key:val.to(device) for key, val in batch.items()}
            outputs = model(batch['images']).squeeze(-1).squeeze(-1)
            loss = criterion(outputs, batch['target'])
            test_batch_loss.append(loss.item())
        test_mean_batch_loss = np.mean(test_batch_loss)

        train_losses.append(train_mean_batch_loss)
        test_losses.append(test_mean_batch_loss)

        print(
            f"Epoch {ep+1}/{epochs}, Train Loss:{train_mean_batch_loss:.4f}, \
                Test Loss: {test_mean_batch_loss:.4f}"
        )
        if save_model:
            if test_mean_batch_loss < best_loss:
                best_loss = test_mean_batch_loss
                torch.save(model, f"{model.__class__.__name__}.pt")
                print(
                    f"saving new best model with Test Loss: {test_mean_batch_loss:.4f}"
                )

    return np.array(train_losses), np.array(test_losses)


def get_transforms():
    """Transformations for train and test images"""

    train_transform = transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def get_dataset(meta_file, train_path):
    """Image dataset from path"""

    color2idx = {dir:i for i,dir in enumerate(os.listdir(train_path))}

    train_transform, test_transform = get_transforms()
    train_dataset = MultiItemDataset(meta_file,'train',color2idx,transform=train_transform)
    test_dataset = MultiItemDataset(meta_file,'test',color2idx,
                                    max_images=train_dataset.max_images, transform=test_transform)
    return train_dataset, test_dataset


def get_dataloaders(meta_file, train_path, batch_size=64, shuffle=True, sampler=False):
    """Image dataset loaders"""

    train_dataset, test_dataset = get_dataset(meta_file, train_path)
    if sampler:
        weighted_sampler = get_weighted_sampler(pd.Series(train_dataset.targets))
        train_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=batch_size, sampler=weighted_sampler
        )
    else:
        train_loader = torch.utils.data.DataLoader(
          train_dataset, batch_size=batch_size, shuffle=True
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def get_weighted_sampler(targets):
    """Get the weighted sampler based on class balance for data loader"""

    class_weights = 1 / targets.value_counts()
    class_index = targets.value_counts().index
    weights_map = {class_index[i]:class_weights.iloc[i] for i in range(len(class_index))}
    class_weights_all = targets.map(weights_map).tolist()

    return WeightedRandomSampler(
            weights=class_weights_all,
            num_samples=len(class_weights_all),
            replacement=True
            )



def accuracy(data_loader, model, device):
    """Calculate accuracy"""

    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = {key:val.to(device) for key, val in batch.items()}
            outputs = model(batch['images']).squeeze(-1).squeeze(-1)
            targets = batch['target']
            _, predictions = torch.max(outputs, 1)

            n_correct += (predictions == targets).sum().item()
            n_total += targets.size(0)

    return n_correct / n_total


def get_classification_report(data_loader, model, device):
    """Get a sklearn multivariable classification report"""

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in data_loader:
            batch = {key:val.to(device) for key, val in batch.items()}
            outputs = model(batch['images']).squeeze(-1).squeeze(-1)
            targets = batch['target']
            _, predictions = torch.max(outputs, 1)

            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
    return metrics.classification_report(y_true, y_pred, zero_division=True)