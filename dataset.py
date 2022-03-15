from matplotlib import transforms
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import PIL.Image as Image
from utils import convert_img


data_transform = {
        "train": transforms.Compose([transforms.ToPILImage(),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5), (0.5))]),
    
        "val": transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize(int(224 * 1.143)),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5), (0.5))])
        }

class LiverDataset(Dataset):
    def __init__(self, images_path: list, images_class: list, mode='train'):
        self.images_path = images_path
        self.images_class = images_class
        self.mode = mode
        if self.mode == 'train':
            self.transform = data_transform['train']
        if self.mode == 'val':
            self.transform = data_transform['val']
        
            
            
    def __len__(self):
        return len(self.images_path)-1

    def __getitem__(self, item):
        original_img = sitk.ReadImage(self.images_path[item])
        img_array = sitk.GetArrayFromImage(original_img)

        img = torch.from_numpy(img_array)
     
        if len(img.shape) == 4:
            img = img[:,:,:,0]

        
        if self.transform is not None:          
            img = self.transform(img)
   
        label = self.images_class[item]

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels