import os
import urllib.request
import zipfile
import shutil

def download_and_prepare_tiny_imagenet(save_dir="data/tiny-imagenet"):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(save_dir, "tiny-imagenet-200.zip")
    extracted_dir = os.path.join(save_dir, "tiny-imagenet-200")

    os.makedirs(save_dir, exist_ok=True)

    # Step 1: Download
    if not os.path.exists(zip_path):
        print("📥 Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete.")
    else:
        print("📦 Zip file already exists, skipping download.")

    # Step 2: Extract
    if not os.path.exists(extracted_dir):
        print("📂 Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(save_dir)
        print("✅ Extract complete.")
    else:
        print("🗂 Already extracted, skipping.")

    # Step 3: Restructure for ImageFolder compatibility
    print("🔧 Rearranging files into ImageFolder format...")

    def move_images(src_folder, target_folder, label_map):
        for label in os.listdir(src_folder):
            class_dir = os.path.join(src_folder, label, "images")
            if not os.path.isdir(class_dir): continue
            target_class_dir = os.path.join(target_folder, label_map[label])
            os.makedirs(target_class_dir, exist_ok=True)
            for img_file in os.listdir(class_dir):
                shutil.copy(os.path.join(class_dir, img_file), os.path.join(target_class_dir, img_file))

    # Train set
    train_src = os.path.join(extracted_dir, "train")
    train_dst = os.path.join(save_dir, "train")
    train_label_map = {x: x for x in os.listdir(train_src)}
    move_images(train_src, train_dst, train_label_map)

    # Validation set
    val_src = os.path.join(extracted_dir, "val")
    val_dst = os.path.join(save_dir, "val")

    # Read val annotations
    val_annotations_path = os.path.join(val_src, "val_annotations.txt")
    val_img_dir = os.path.join(val_src, "images")
    val_label_map = {}
    with open(val_annotations_path, "r") as f:
        for line in f.readlines():
            fname, label = line.strip().split("\t")[:2]
            val_label_map[fname] = label

    # Copy validation images to class folders
    for fname, label in val_label_map.items():
        target_dir = os.path.join(val_dst, label)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(os.path.join(val_img_dir, fname), os.path.join(target_dir, fname))

    print("✅ Tiny ImageNet is ready at:", save_dir)

if __name__ == "__main__":
    download_and_prepare_tiny_imagenet()