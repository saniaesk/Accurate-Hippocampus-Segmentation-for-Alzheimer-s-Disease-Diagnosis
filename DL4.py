import os
import numpy as np  
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

class MRIDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_shape=(215, 234, 42)):
        self.root_dir = root_dir
        self.subjects = sorted(os.listdir(root_dir))
        self.transform = transform
        self.target_shape = target_shape

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_dir = os.path.join(self.root_dir, self.subjects[idx])
        image_path = os.path.join(subject_dir, 't1.nii.gz')
        image = nib.load(image_path).get_fdata().astype(np.float32)
        image = self.resize_image(image, self.target_shape)

        if self.transform:
            image = self.transform(image)

        return image

    @staticmethod
    def resize_image(image, target_shape):
        if image.shape != target_shape:
            factors = [n / o for n, o in zip(target_shape, image.shape)]
            image = zoom(image, factors, order=1)
        return image

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust these values based on your data's characteristics
])

# Setup
root_dir = 'E:\\zenodo_upload_v2'
dataset = MRIDataset(root_dir, transform=transform)

# Splitting dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)

# Example usage
batch_count = 0
for batch in train_dataloader:
    print(batch.shape)
    batch_count += 1
print("Total batches processed:", batch_count)

# Display an image example
image = dataset[0].numpy()
image = np.flip(np.rot90(image, k=1, axes=(1, 2)), axis=1)  # Flip vertically and rotate by 90 degrees counterclockwise

plt.imshow(image[26], cmap='gray', origin='lower')  # Display one slice of the image
plt.title(f'Slice Index: 26')
plt.axis('on')
#plt.colorbar(label='Intensity')
#plt.xlabel('X')
#plt.ylabel('Y')
plt.show()
