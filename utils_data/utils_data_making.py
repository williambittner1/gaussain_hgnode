from PIL import Image as PIL_Image
import os
import shutil

def generate_seg_images(args, target_folder, num_cam, img_name):
    """Generate black images PNG as everything is static currently. Images have the same size as the original images.
    TODO: change this to generate segmentation images when doing dynamic scenes.
    Parameters:
        target_folder (str): Path to the folder where the original images are.
        num_cam (str): Number of the camera.
    """
    original_image = PIL_Image.open(os.path.join(target_folder, img_name))
    width, height = original_image.size
    black_image = PIL_Image.new('L', (width, height), 1)
    seg_folder = os.path.join(args.output_path, args.dataset_name, 'seg')
    target_seg_folder = os.path.join(seg_folder, num_cam)
    if not os.path.exists(target_seg_folder):
        os.makedirs(target_seg_folder)
    black_image.save(os.path.join(target_seg_folder,'render.png'))

def generate_seg(args):
    """ 
    Generate segmentation images for all cameras in the dataset. The images should be plain 
    white and have the same size as the rgb images. They should be populated in the seg folder 
    with the same structure as the ims folder.
    """
    # Check if the target folder exists; create it if not
    seg_folder = os.path.join(args.output_path, 'seg')
    if not os.path.exists(seg_folder):
        os.makedirs(seg_folder)

    # Copy each item from source to target
    for item in os.listdir(os.path.join(args.output_path, 'ims')):
        source_item = os.path.join(args.output_path, 'ims', item)
        target_item = os.path.join(seg_folder, item)

        # Copy files and directories
        if os.path.isdir(source_item):
            shutil.copytree(source_item, target_item)
        else:
            shutil.copy2(source_item, target_item)

        # the copied item should be changed to a white image with the same size as the original image
        
def create_white_seg_images(input_folder, output_folder, new_extension='.png'):
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each subfolder in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            new_subfolder_path = os.path.join(output_folder, subfolder)

            # Create corresponding subfolder in the output folder
            if not os.path.exists(new_subfolder_path):
                os.makedirs(new_subfolder_path)

            # Iterate through each file in the subfolder
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)

                # Check if it's a file
                if os.path.isfile(file_path):
                    img = PIL_Image.open(file_path)
                    width, height = img.size

                    # Create a white image
                    white_image = PIL_Image.new('RGB', (width, height), 'white')

                    # Save the white image with the same name but different extension
                    new_file_name = os.path.splitext(file_name)[0] + new_extension
                    new_file_path = os.path.join(new_subfolder_path, new_file_name)
                    white_image.save(new_file_path)