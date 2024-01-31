import zipfile
import os
import pandas as pd
import wget


import subprocess

#Replace 'cities/titanic123' with the actual dataset you want to download
dataset_name = 'cities/titanic123'

# Construct the Kaggle command
kaggle_command = f'kaggle datasets download -d {dataset_name}'

# Run the Kaggle command using subprocess
subprocess.run(kaggle_command, shell=True)
destination_folder = r"D:\\VED\\model_tracking"

zip_file_path = 'titanic123.zip'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents to the destination folder
    zip_ref.extractall(destination_folder)
print(f"Successfully extracted files to")