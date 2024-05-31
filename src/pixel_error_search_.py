from PIL import Image
import os

def check_image_integrity(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # verify that it is, in fact, an image
    except (IOError, SyntaxError) as e:
        print(f"Corrupted file: {file_path}")

def main():
    directory = '/home/mmc/disk2/duck/cap/data/noodle/train/images/'
    corrupted_files = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add any file extensions of interest
           # print(filename)
            file_path = os.path.join(directory, filename)
            try:
                check_image_integrity(file_path)
            except Exception as e:
                corrupted_files.append(filename)
    
    if corrupted_files:
        print("List of corrupted files:")
        for file in corrupted_files:
            print(file)
    else:
        print("No corrupted files found")
if __name__ == "__main__":
    main()