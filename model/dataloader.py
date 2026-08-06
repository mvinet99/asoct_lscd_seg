import torch
import numpy as np
import os
from torch.utils import data
from torchvision import transforms as T
from PIL import Image
from torch.utils.data.distributed import DistributedSampler

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, mode='train'):
        self.imList = imList
        self.labelList = labelList
        self.mode = mode

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image_name = self.imList[idx]
        label_name = self.labelList[idx]

        # Load image and label
        image = np.load(image_name)
        label = np.load(label_name)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = Image.fromarray(image)

        if label.ndim > 2:
            label = np.squeeze(label)
        label = np.array(label, dtype=np.float32)
        label[label != 0.0] = 1.0  

        label = Image.fromarray(label)

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor()
        ])
        image = transform(image)
        label = transform(label)

        # Normalize image
        image = T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(image.float())

        return image, label.float(),  os.path.basename(image_name)

def get_loader(imList, labelList, batch_size, num_workers=1, mode='train', drop_last=False, distributed=False):
    """
    Builds and returns a DataLoader with optional DistributedSampler.
    """

    dataset = MyDataset(imList, labelList, mode=mode)

    sampler = None
    shuffle_data = (mode == 'train')

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=(mode == 'train'))
        shuffle_data = False

    loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle_data,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True)

    return loader, sampler
