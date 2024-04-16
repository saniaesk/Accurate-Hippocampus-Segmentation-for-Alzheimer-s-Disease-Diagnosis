import os
import nibabel as nib
import matplotlib.pyplot as plt

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

def display_image(image, slice_index):
    # Select a slice of the image
    image_slice = image[:, :, slice_index]

    # Display the image
    plt.imshow(image_slice.T, cmap='gray', origin='lower')  # Transpose the image slice for correct orientation
    plt.title(f'Slice Index: {slice_index}')  # Add a title to the plot with the slice index
    plt.axis('on')  # Show axes
    plt.show()

# Path to the directory containing the dataset
root_dir = 'E:\\zenodo'

# Directory for the atlas
subject_dir = os.path.join(root_dir, 'atlas')

# Load the image 1
#image = load_image(subject_dir, 't1_reference_subject_06mm.nii.gz')
image = load_image(subject_dir, 't1_reference_subject_06mm.nii.gz')

if image is not None:
    # Print the dimensions of the image
    print(f'Dimensions of image: {image.shape}')

    # Select a slice based on the image size
    #slice_index = image.shape[2] // 2  # Change this line to select a different slice along the third axis

    # Display the image
    display_image(image, 26)
