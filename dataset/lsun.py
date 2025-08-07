import torchvision.datasets as datasets
import torchvision.transforms as transforms

from monai.data import Dataset
from monai.transforms import Compose, Identity

class LSUNDatasetWrapper(Dataset):
    def __init__(self, root='/mimer/NOBACKUP/Datasets/LSUN', classes=['classroom_train']):

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor() #(To (0,1))
        ])

        self.torchvision_dataset = datasets.LSUN(root=root, classes=classes, transform=transform)

    def __len__(self):
        return len(self.torchvision_dataset)

    def __getitem__(self, idx):
        image, _ = self.torchvision_dataset[idx]

        monai_transforms = Compose([
            Identity()
        ])

        image = monai_transforms(image)
        
        return image