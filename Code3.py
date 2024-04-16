import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scipy.ndimage import zoom

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
            image = zoom(image, factors, order=0)
        return image

def load_image(subject_dir, filename):
    # Check if the directory exists
    if not os.path.isdir(subject_dir):
        print(f"Directory does not exist: {subject_dir}")
        return None

    # Path to the image file
    image_path = os.path.join(subject_dir, filename)

    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"File does not exist: {image_path}")
        return None

    # Load the image
    image = nib.load(image_path).get_fdata()

    return image

def register_atlas_to_image(sample_image, atlas_image, slice_range):
    registered_atlas_slices = []

    for slice_index in range(*slice_range):
        moving_slice = atlas_image[:, :, slice_index]
        fixed_slice = sample_image[slice_index]

        # Convert slices to SimpleITK images
        moving_image = sitk.GetImageFromArray(moving_slice)
        fixed_image = sitk.GetImageFromArray(fixed_slice)

        # Perform registration
        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                                    minStep=0.001,
                                                                    numberOfIterations=100,
                                                                    relaxationFactor=0.5)
        registration_method.SetInterpolator(sitk.sitkLinear)

        transform = sitk.TranslationTransform(2)
        registration_method.SetInitialTransform(transform, inPlace=False)

        registration_method.AddCommand(sitk.sitkStartEvent, lambda: print("Starting Registration"))
        registration_method.AddCommand(sitk.sitkEndEvent, lambda: print("Registration Finished"))
        registration_method.AddCommand(sitk.sitkIterationEvent,
                                       lambda: print("\r{0:.2f}".format(
                                           registration_method.GetMetricValue()), end=""))

        final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                      sitk.Cast(moving_image, sitk.sitkFloat32))

        # Apply transformation to moving slice
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(final_transform)

        registered_slice = resampler.Execute(moving_image)

        registered_atlas_slices.append(sitk.GetArrayFromImage(registered_slice))

    return registered_atlas_slices

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust these values based on your data characteristics
])

# Setup
root_dir = 'E:\\zenodo_upload_v2'
dataset = MRIDataset(root_dir, transform=transform)

# DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the sample image
sample_image = dataset[0].numpy()

# Path to the directory containing the dataset
root_dir = 'E:\\zenodo'

# Directory for the atlas
subject_dir = os.path.join(root_dir, 'atlas')

# Load the atlas image
atlas_image = load_image(subject_dir, 't1_reference_subject_06mm.nii')

if atlas_image is not None:
    # Print the dimensions of the image
    print(f'Dimensions of atlas image: {atlas_image.shape}')

    # Define the range of slices for registration
    slice_range = (19, 30)

    # Perform atlas registration
    registered_atlas_slices = register_atlas_to_image(sample_image, atlas_image, slice_range)

    # Display registered atlas slices
    for idx, registered_slice in enumerate(registered_atlas_slices):
        plt.imshow(registered_slice, cmap='gray', origin='lower')
        plt.title(f'Registered Atlas Slice {idx+slice_range[0]}')
        plt.axis('on')
        plt.show()
