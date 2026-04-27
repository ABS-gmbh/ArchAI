import os
import zipfile

def unzip_all(directory):
    """
    Recursively finds all .zip files in the directory and unzips them
    in the same location as the zip file.
    """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.zip'):
                zip_path = os.path.join(root, filename)
                # Unzip in the same directory as the zip file
                print(f"Unzipping {zip_path}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    print(f"Done unzipping {zip_path}")
                except zipfile.BadZipFile:
                    print(f"Warning: {zip_path} is not a valid zip file, skipping...")
                except Exception as e:
                    print(f"Error unzipping {zip_path}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python unzip.py <directory>")
    else:
        unzip_all(sys.argv[1])
