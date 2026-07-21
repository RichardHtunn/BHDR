import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

class BurmeseDigitDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]], transform: Optional[transforms.Compose] = None) -> None:
        self.data = data_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        current_item = self.data[idx]
        image_data = current_item['image']
        label_data = current_item['label']

        # Convert raw numpy array into a PIL Image
        pil_image = Image.fromarray(image_data)

        # Apply mutations/transformations
        if self.transform:
            image_tensor = self.transform(pil_image)
        else:
            image_tensor = transforms.ToTensor()(pil_image)

        label_tensor = torch.tensor(label_data).long()

        return image_tensor, label_tensor