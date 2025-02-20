#!/usr/bin/env python3

import os
import shutil

def reorganize_sequences(root_dir):
    """
    For each folder named 'sequenceXXXX' inside root_dir:
      - Create 'dynamic' and 'static' subfolders.
      - Move .mp4 files from ims/*/ into 'dynamic'.
      - Move .png files from ims/*/ into 'static'.
    """
    # Iterate over all items in the root directory
    for seq_name in os.listdir(root_dir):
        print(f"Processing {seq_name}")
        seq_path = os.path.join(root_dir, seq_name)
        
        # Only process directories that match 'sequence...'
        if not os.path.isdir(seq_path) or not seq_name.startswith('sequence'):
            continue
        
        ims_path = os.path.join(seq_path, 'ims')
        if not os.path.isdir(ims_path):
            continue  # No ims folder, skip
        
        # Create 'dynamic' and 'static' folders (if they don't exist yet)
        dynamic_path = os.path.join(seq_path, 'dynamic')
        static_path = os.path.join(seq_path, 'static')
        os.makedirs(dynamic_path, exist_ok=True)
        os.makedirs(static_path, exist_ok=True)
        
        # Go through each subfolder in ims (e.g., cam_000_video, cam_001_image, etc.)
        for folder_name in os.listdir(ims_path):
            folder_path = os.path.join(ims_path, folder_name)
            
            if os.path.isdir(folder_path):
                # Check the contents of this folder
                files = os.listdir(folder_path)
                
                # Expecting only one file (either .mp4 or .png)
                if len(files) == 1:
                    file_name = files[0]
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Move the file based on extension
                    if file_name.lower().endswith('.mp4'):
                        shutil.move(file_path, dynamic_path)
                    elif file_name.lower().endswith('.png'):
                        shutil.move(file_path, static_path)
                    else:
                        print(f"Skipping unknown file type: {file_path}")
                else:
                    print(f"Warning: multiple or no files found in {folder_path}, skipping.")
            else:
                print(f"Skipping non-directory item: {folder_path}")

if __name__ == "__main__":
    # Update this to your actual root directory path
    root_directory = "/work/williamb/datasets/bouncing_spheres_1000seq_200ts_video"
    reorganize_sequences(root_directory)
