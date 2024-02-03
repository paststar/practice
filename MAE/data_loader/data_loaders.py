import os

from glob import glob
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

class ImageNetDataset(Dataset):
    def __init__(self, root, split):
        super().__init__()
        
        # path = 'ILSVRC2012_img_train' if split == 'train' else 'ILSVRC2012_img_val'
        #self.L = glob(os.path.join(root, path,'*'))
        self.L = glob(os.path.join(root,'ILSVRC2012_img_val','*'))

        if split == 'train':
            self.L = self.L[:100]

            self.transform =  transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, idx):
        try:
            img = Image.open(self.L[idx]).convert("RGB")
            return self.transform(img)
        except Exception as e:
            print(e)
            print(self.L[idx])

    
    def __len__(self):
        return len(self.L)

class ImageNetDataLoader(DataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=1):
        # cf. 논문에선 RandomResizedCrop 사용

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = ImageNetDataset(root=self.data_dir, split='train')
        self.val_dataset = ImageNetDataset(root=self.data_dir, split='val')

        datasets.ImageNet

        train_init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': self.batch_size,
            'shuffle': shuffle,
            'num_workers': self.num_workers
        }
        super().__init__(**train_init_kwargs)
    

    def split_validation(self):
        val_init_kwargs = {
            'dataset': self.val_dataset, 
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers
        }
        return DataLoader(**val_init_kwargs)

if __name__ == '__main__':
    tmp = ImageNetDataset("/root/data",split='val')