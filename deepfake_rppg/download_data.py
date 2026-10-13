import kagglehub
import shutil
import os
import glob

def download_and_setup_data():
    print("Downloading dataset 'xdxd003/ff-c23' using kagglehub...")
    print("This may take a while depending on your connection speed...")
    
    # Download dataset
    path = kagglehub.dataset_download("xdxd003/ff-c23")
    print(f"Dataset downloaded to: {path}")
    
    target_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Inspecting downloaded files in {path}...")
    # List files to understand structure
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        # Limit file print
        for f in files[:5]:
            print('{}{}'.format(subindent, f))
        if len(files) > 5:
            print('{}(... {} more files)'.format(subindent, len(files)-5))
            
    # Attempt to copy or symlink
    # Since we don't know the exact structure yet, let's keep it in the cache 
    # and just print the path for the user to use in main.py, OR return it.
    
    return path

if __name__ == "__main__":
    download_and_setup_data()
