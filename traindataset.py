import os
import shutil

# Set the path to the directory containing your images
source_dir = 'imageformangotree\MangoLeafBD Dataset'  # Change to your actual directory
# Set the path to the base dataset directory
base_dir = 'D:/opencv/dataset'

# Create train and validation directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# List of disease classes
disease_classes = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 
                    'Gall Midge', 'Powdery Mildew', 'Sooty Mould', 'Healthy']  # Add more as needed

# Create class directories under train and validation
for cls in disease_classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, cls), exist_ok=True)

# Function to move images to the respective class folder
def organize_images():
    for cls in disease_classes:
        # Create the path for class folder in the source directory
        class_source_dir = os.path.join(source_dir, cls)

        # Check if the class source directory exists
        if os.path.exists(class_source_dir):
            # Get all images in the class source directory
            images = os.listdir(class_source_dir)

            # Split images into train and validation sets (80% train, 20% validation)
            train_images = images[:int(0.8 * len(images))]
            validation_images = images[int(0.8 * len(images)):]

            # Move images to the train directory
            for img in train_images:
                shutil.move(os.path.join(class_source_dir, img), os.path.join(train_dir, cls, img))

            # Move images to the validation directory
            for img in validation_images:
                shutil.move(os.path.join(class_source_dir, img), os.path.join(validation_dir, cls, img))

            print(f"Organized {len(images)} images for class '{cls}'.")

        else:
            print(f"Source directory for class '{cls}' does not exist.")

# Run the organization function
organize_images()
