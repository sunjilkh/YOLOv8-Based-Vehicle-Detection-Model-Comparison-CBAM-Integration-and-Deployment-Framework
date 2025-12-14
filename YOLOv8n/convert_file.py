import os
import xml.etree.ElementTree as ET

# Define class mapping
class_map = {
    "bicycle": 0,
    "bus": 1,
    "car": 2,
    "cng": 3,
    "auto": 4,
    "bike": 5,
    "Multi-Class": 6,
    "rickshaw": 7,
    "truck": 8,
    "van": 9
}

def convert_voc_to_yolo(xml_file, output_dir, class_map):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Image dimensions
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    print(f"Processing file: {xml_file}, Width: {width}, Height: {height}")

    output_file = os.path.join(output_dir, os.path.splitext(root.find("filename").text)[0] + ".txt")
    with open(output_file, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            print(f"Found object: {class_name}")

            if class_name not in class_map:
                print(f"Skipping unknown class: {class_name}")
                continue  # Skip if class is not in the mapping

            class_id = class_map[class_name]

            # Bounding box coordinates
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            xmax = int(bbox.find("xmax").text)
            ymin = int(bbox.find("ymin").text)
            ymax = int(bbox.find("ymax").text)

            print(f"BBox: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            bbox_width = (xmax - xmin) / width
            bbox_height = (ymax - ymin) / height

            print(f"YOLO Format: {class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

            # Write to YOLO format
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

def process_all_folders(base_dir, output_dir, class_map):
    """Process all subfolders in a directory and convert XML annotations."""
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if not exists

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.isdir(folder_path):  # Process only folders
            print(f"Processing folder: {folder_name}")
            
            # Create a subfolder in the output directory
            output_subfolder = os.path.join(output_dir, folder_name)
            os.makedirs(output_subfolder, exist_ok=True)
            
            # Process each XML file in the folder
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".xml"):
                    print("ok")
                    xml_file = os.path.join(folder_path, file_name)
                    convert_voc_to_yolo(xml_file, output_subfolder, class_map)

# Directories
base_dir = "Before Augmentation"  # Replace with the directory containing the folders
output_dir = "Output"  # Replace with the directory to save YOLO annotations

# Run the conversion
process_all_folders(base_dir, output_dir, class_map)
# convert_voc_to_yolo('./hello.xml','./',class_map)
print("Conversion completed!")
