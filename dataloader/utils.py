# dataloader/utils.py

import time
import os
import requests
import tarfile
import zipfile
import numpy as np
from tqdm import tqdm

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' took {time.time() - start_time:.2f}s to complete.")
        return result
    return wrapper


def extract_nested_tar(tar_path, extract_dir):
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(path=extract_dir)
    
    # Check for nested tar files and extract them
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(('.tar', '.tar.gz', '.tgz')):
                nested_tar_path = os.path.join(root, file)
                nested_extract_dir = os.path.splitext(nested_tar_path)[0]
                extract_nested_tar(nested_tar_path, nested_extract_dir)
                os.remove(nested_tar_path)  # Remove the nested tar file after extraction

def download_file(url, dest_path):
    # Extract the file name from the URL
    file_name = os.path.basename(url)
    
    # Combine the folder path with the file name to get the full path
    dest_path = os.path.join(dest_path, file_name)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        file_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)

        # Open the file in the folder and write content
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                size = file.write(data)
                progress_bar.update(size)
        progress_bar.close()

        print(f"File downloaded to {dest_path}")
    
    except PermissionError as pe:
        print(f"Permission error: {pe}. Check if you have write access to {dest_path}.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    # Extract or process the file based on its format
    file_name = os.path.basename(dest_path)
    extract_dir = os.path.dirname(dest_path)

    if file_name.endswith(('.tar.gz', '.tgz', '.tar')):
        extract_nested_tar(dest_path, extract_dir)        
    elif file_name.endswith('.zip'):
        with zipfile.ZipFile(dest_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        os.remove(dest_path)
    elif file_name.endswith('.npz'):
        # For .npz files, we'll load the data and save it in a more accessible format
        with np.load(dest_path) as data:
            for key in data.files:
                np.save(os.path.join(extract_dir, f"{key}.npy"), data[key])
        os.remove(dest_path)
        print(f"NPZ file unpacked. Individual .npy files saved in {extract_dir}")
    else:
        print(f"Unknown file format: {file_name}. File will be kept as is.")

    print(f"File downloaded and processed (if applicable) in {extract_dir}")