from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1):
        # cf. 논문에선 RandomResizedCrop 사용
        trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = datasets.ImageNet(root=self.data_dir, transforms=trsfm, split='train')
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)
    

    def split_validation(self):
        trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])

        self.val_dataset = datasets.ImageNet(root=self.data_dir, transforms=trsfm, split='val')
        self.init_kwargs = {
            'dataset': self.val_dataset,
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }

        return DataLoader(**self.init_kwargs)