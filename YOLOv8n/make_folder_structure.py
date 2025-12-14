import os
import random
import shutil

# Set random seed for reproducibility
random.seed(42)

# Define the source directory containing images and annotations
source_dir = 'All images'  # Replace with the path to your folder containing all images & annotations
train_dir = 'dataset/images/train'
train_labels_dir = 'dataset/labels/train'
val_dir = 'dataset/images/val'
val_labels_dir = 'dataset/labels/val'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# List all images in the source directory
all_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
random.shuffle(all_files)  # Shuffle the list to ensure randomness

# Split ratio (e.g., 80% train, 20% val)
split_ratio = 0.8
split_idx = int(len(all_files) * split_ratio)

# Split into training and validation
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

# Function to move files safely
def move_files(file_list, image_dir, label_dir):
    for file_name in file_list:
        # Move the image
        src_image_path = os.path.join(source_dir, file_name)
        dst_image_path = os.path.join(image_dir, file_name)
        shutil.move(src_image_path, dst_image_path)

        # Move the corresponding annotation file
        # Assumes the annotation file has the same name but ends in .txt
        annotation_file = file_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        src_annotation_path = os.path.join(source_dir, annotation_file)
        if os.path.exists(src_annotation_path):
            dst_annotation_path = os.path.join(label_dir, annotation_file)
            shutil.move(src_annotation_path, dst_annotation_path)
        else:
            print(f"Warning: Annotation file {annotation_file} not found for image {file_name}")

# Move the training and validation files
print("Moving training files...")
move_files(train_files, train_dir, train_labels_dir)
print("Moving validation files...")
move_files(val_files, val_dir, val_labels_dir)

print("Dataset split complete!")
print(f"Training images and labels moved to: {train_dir}, {train_labels_dir}")
print(f"Validation images and labels moved to: {val_dir}, {val_labels_dir}")
