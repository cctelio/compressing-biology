import pandas as pd
from monai.data import CSVDataset
import numpy as np
import monai

class CPG0000:
    
    def __init__(self, path_csv):

        self.channel_names = ["channel_1","channel_2", "channel_3", "channel_4", "channel_5"]

        self.val_transforms = monai.transforms.Compose(
            [
                monai.transforms.LoadImaged(keys=self.channel_names, ensure_channel_first=True),
                monai.transforms.EnsureTyped(keys=self.channel_names, dtype=np.float32, track_meta=False),
                monai.transforms.ConcatItemsd(keys=self.channel_names, name="IMAGE"),
                monai.transforms.NormalizeIntensityd(keys="IMAGE", subtrahend=0., divisor=255.),
                monai.transforms.CenterSpatialCropd(keys="IMAGE", roi_size=(1024,1024)),
                monai.transforms.GridPatchD(
                            keys=["IMAGE"],
                            patch_size=(256, 256),
                            offset=(0, 0),
                            stride=(256, 256)
                ),
            ]
        )
                
        self.df = pd.read_csv(path_csv)

        self.csv_ds = CSVDataset(self.df)