"""
Created on 2025/8/8 
Author: Hao Chen (chen960216@gmail.com)
"""
import os
import shutil
from pathlib import Path

# some path for working directories
workdir = Path(__file__).parent
workdir_d3qn = workdir.joinpath("rl_frame", "d3qn")

def delete_all_files_in_directory(folder_path):
    folder_path = str(folder_path)
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))