import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms

import os
#
# class Space_Galaxi(Dataset):
#     def __init__(self,csv_path,img_dir_path,transform=None):
#         self.df = pd.read_csv(
#             csv_path,
#             sep=',',  # разделитель — запятая
#             quotechar='"',  # кавычки — "
#             engine='python',  # если регулярный движок падает
#         )
#         self.img_dir_path = img_dir_path
#         self.transform = transform
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, item):
#         id = self.df['id'].iloc[item]
#         obj = self.df['class'].iloc[item]
#         z = self.df['z'].iloc[item]
#         z_err = self.df['z_err'].iloc[item]
#
#         z_features = torch.tensor([z,z_err])
#         image_path = os.path.join(self.img_dir_path, f"{id}.jpg")
#         image = Image.open(image_path)  # Открываем изображение
#         if self.transform:
#             image = self.transform(image)
#         return image,obj,z_features

class Subset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]  # ← ключевая строчка!

    def __len__(self):
        return len(self.indices)

class Space_Galaxi(Dataset):
    def __init__(self,img_dir_path,csv_path=None,transform=None):
        self.img_dir_path = img_dir_path
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        self.image_names = [f for f in os.listdir(img_dir_path) if os.path.isfile(os.path.join(img_dir_path, f)) and os.path.splitext(f)[1].lower() in image_extensions]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        img = self.image_names[item]
        label = 0 if 'dog' in img else 1 if 'cat' in img else None
        image_path = os.path.join(self.img_dir_path,img)
        image = Image.open(image_path).convert("RGB")  # Открываем изображение
        if self.transform:
            transformed = self.transform(image)  # ← возвращает {'image': tensor, 'rotation_angle': float}
            # image_tensor = transformed['image']
            # rotation_angle = transformed['rotation_angle']
            # Возвращаем тензор, угол и метку
        return transformed,label

# # Корректные пути
# csv_path = r'D:\Code\Space_Warps\spall_csv_chunks_lazy\spall_chunk_0001.csv'
# img_dir_path = r'D:\Code\Space_Warps\data\image_data\img_csv_0001'
#
# # Проверка
# if not os.path.exists(csv_path):
#     raise FileNotFoundError(f"CSV файл не найден: {csv_path}")
# if not os.path.exists(img_dir_path):
#     raise FileNotFoundError(f"Папка с изображениями не найдена: {img_dir_path}")
#
# reg = Space_Galaxi(csv_path, img_dir_path=img_dir_path)
#
# # Пример вывода
# for i, (ra, dec, image) in enumerate(reg):
#     print(f"{ra}, {dec}")
#     plt.figure(figsize=(5, 5))
#     plt.imshow(image)
#     plt.axis('off')
#     plt.title(f"ID: {reg.df['id'].iloc[i]}, RA: {ra}, Dec: {dec}")
#     plt.show()
#     print(i)
#     print('-' * 50)
#
# # files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# # print(files)
