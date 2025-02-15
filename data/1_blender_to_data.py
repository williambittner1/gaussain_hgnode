import os
import sys

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import os
from glob import glob
import shutil
from utils_data import utils_data_making, utils_colmap
import numpy as np
import json
from PIL import Image as PIL_Image

def get_intrinsics(camera_info):
    """Get intrisics matrix from camera info"""
    fx = camera_info['fx']
    fy = camera_info['fy']
    cx = camera_info['width'] / 2
    cy = camera_info['height'] / 2
    position = camera_info['position']
    k = [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]
    
    return k      

def main(args):
    print('Converting Blender data to Dynamic 3D Gaussians data...')

    # Copy images from bleder folder to data. Images need to be .JPEG.
    img_source_folder = os.path.join(args.data_path, 'ims')
    target_folder = ims_folder = os.path.join(args.output_path, 'ims')
    seg_folder = os.path.join(args.output_path, 'seg')
    
    # img_source_files = glob(os.path.join(img_source_folder, '*', '*.jpg'))

    # for img_source_file in img_source_files:
    #     num_cam = img_source_file.split('/')[-2]
    #     target_folder = os.path.join(ims_folder, num_cam)
    #     if not os.path.exists(target_folder):
    #         os.makedirs(target_folder)

    #     shutil.copy(img_source_file, os.path.join(target_folder, 'render.jpg'))

    # # Check if the target folder exists; create it if not
    # if not os.path.exists(target_folder):
    #     os.makedirs(target_folder)

    # # Check if the target folder exists; create it if not
    # if not os.path.exists(seg_folder):
    #     os.makedirs(seg_folder)

    # # Copy each item from source to target
    # for item in os.listdir(img_source_folder):
    #     for file in os.listdir(os.path.join(img_source_folder, item)):

    #         source_item = os.path.join(img_source_folder, item, file)
    #         target_item = os.path.join(target_folder, item, file)
    #         seg_target_item = os.path.join(seg_folder, item, file)

    #         if not os.path.exists(os.path.join(target_folder, item)):
    #             os.makedirs(os.path.join(target_folder, item))

    #         # Copy files and directories
    #         if os.path.isdir(source_item):
    #             shutil.copytree(source_item, target_item)
    #         else:
    #             shutil.copy2(source_item, target_item)


    #     # Generate black images as everything is static currently. Images have the same size as the original images
    #     # Segmentation imaged need to be .PNG
    #     # TODO: change this to generate real segmentation images

    #     utils_data_making.create_white_seg_images(target_folder, seg_folder, new_extension='.png')
    #     # utils_data_making.generate_seg_images(args, target_folder, num_cam, img_name='render.jpg')

    # Copy point cloud from Blender folder to repository
    # pc_source_folder = os.path.join(args.data_path, 'init_pt_cld.npz')
    # shutil.copy(pc_source_folder, os.path.join(args.output_path, 'init_pt_cld.npz'))

    # Get intrinsics and extrinsics values from Blender data
    data = dict()
    cameras_info_path = os.path.join(args.data_path, 'cameras_gt.json')

    # read cameras info
    with open(cameras_info_path, 'r') as f:
        cameras_info = json.load(f)  # list of dictionaries, each dict is a camera

    data['w'] = cameras_info[0]['width']
    data['h'] = cameras_info[0]['height']
    
    w2c, c2w, k, cam_id, fn = [], [], [], [], []    
    
    total_timesteps = cameras_info[0]['total_timesteps']
    total_cameras = cameras_info[0]['total_cameras']


    for t in range(total_timesteps):
        k_inner = []
        w2c_inner = []
        c2w_inner = []
        cam_id_inner = []
        fn_inner = []
        for c in range(total_cameras):
            # select camera_info, where camera_info['id'] == c and camera_info['t'] == t
            curr_camera_info = [camera_info for camera_info in cameras_info if camera_info['id'] == c and camera_info['t'] == t][0]
            k_inner.append(get_intrinsics(curr_camera_info))
            w2c_inner.append(curr_camera_info['w2c'])
            c2w_inner.append(curr_camera_info['c2w'])
            cam_id_inner.append(str(curr_camera_info['id']))
            fn_inner += [f"{args.output_path}/ims/{cam_id_inner[c]}/{curr_camera_info['img_name']}.png"]

# "fn": [["1/000000.jpg", "2/000000.jpg", "3/000000.jpg", "4/000000.jpg", "5/000000.jpg", "6/000000.jpg", "7/000000.jpg", 
#       "8/000000.jpg", "9/000000.jpg", "11/000000.jpg", "12/000000.jpg", "13/000000.jpg", "14/000000.jpg", "16/000000.jpg", 
#       "17/000000.jpg", "18/000000.jpg", "19/000000.jpg", "20/000000.jpg", "21/000000.jpg", "22/000000.jpg", "23/000000.jpg", 
#       "24/000000.jpg", "25/000000.jpg", "26/000000.jpg", "27/000000.jpg", "28/000000.jpg", "29/000000.jpg"],

        k.append(k_inner)
        w2c.append(w2c_inner)
        cam_id.append(cam_id_inner)
        fn.append(fn_inner)
        c2w.append(c2w_inner)
        # # "fn": [["1/000000.jpg", "2/000000.jpg", "3/000000.jpg", "4/000000.jpg", "5/000000.jpg", "6/000000.jpg", "7/000000.jpg", 
#       "8/000000.jpg", "9/000000.jpg", "11/000000.jpg", "12/000000.jpg", "13/000000.jpg", "14/000000.jpg", "16/000000.jpg", 
#       "17/000000.jpg", "18/000000.jpg", "19/000000.jpg", "20/000000.jpg", "21/000000.jpg", "22/000000.jpg", "23/000000.jpg", 
#       "24/000000.jpg", "25/000000.jpg", "26/000000.jpg", "27/000000.jpg", "28/000000.jpg", "29/000000.jpg"],



    # IMPOTANT! Change this when moving from static to dynamic
    data['w2c'] = w2c
    data['k'] = k
    data['cam_id'] = cam_id
    data['fn'] = fn
    data['c2w'] = c2w
    # Save data as a json file
    with open(os.path.join(args.output_path, 'train_meta.json'), 'w') as f:
        json.dump(data, f)


if __name__=='__main__':
    dataset_name = "25_cams_1k_res"
    args = argparse.ArgumentParser()
    args.add_argument('--data_path', type=str, default=f'data/{dataset_name}', help='Path to the Blender data.')
    args.add_argument('--output_path', type=str, default=f'data/{dataset_name}', help='Path to the output data.')
    args.add_argument('--dataset_name', type=str, default=dataset_name, help='Name of the dataset.')
    args = args.parse_args()

    main(args)