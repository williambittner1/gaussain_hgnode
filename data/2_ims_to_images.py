import os
import shutil
from pathlib import Path

def collect_first_images():
    
    dataset_name = "100_ts_spheres_and_floor_25_cams_1k_res"

    output_dir = Path(f'data/{dataset_name}/images')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all subdirectories in data/ims
    ims_dir = Path(f'data/{dataset_name}/ims')
    subdirs = sorted([d for d in ims_dir.iterdir() if d.is_dir()])  # Sort to ensure consistent ordering
    
    # Process each subdirectory
    for idx, subdir in enumerate(subdirs):
        # Get all image files (assuming they're jpg or png)
        image_files = sorted(list(subdir.glob('*.jpg')) + list(subdir.glob('*.png')))
        
        if image_files:
            # Get the first image
            first_image = image_files[0]
            
            # Create output filename in format cam_000, cam_001, etc.
            output_filename = f"cam_{idx:03d}{first_image.suffix}"  # Keep original extension
            output_path = output_dir / output_filename
            
            # Copy the file
            print(f"Copying {first_image} to {output_path}")
            shutil.copy2(first_image, output_path)
        else:
            print(f"No images found in {subdir}")

if __name__ == "__main__":
    collect_first_images()