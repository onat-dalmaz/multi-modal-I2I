import os
import glob
from PIL import Image
import numpy as np
help_msg = """
The dataset can be downloaded from https://cityscapes-dataset.com.
Please download the datasets [gtFine_trainvaltest.zip] and [leftImg8bit_trainvaltest.zip] and unzip them.
gtFine contains the semantics segmentations. Use --gtFine_dir to specify the path to the unzipped gtFine_trainvaltest directory. 
leftImg8bit contains the dashcam photographs. Use --leftImg8bit_dir to specify the path to the unzipped leftImg8bit_trainvaltest directory. 
The processed images will be placed at --output_dir.

Example usage:

python3 prepare_cityscapes_dataset.py --gtFine_dir ./gtFine/ --leftImg8bit_dir ./leftImg8bit --output_dir /auto/data2/odalmaz/CVproject/pytorch-CycleGAN-and-pix2pix/datasets/datasets/
"""

def load_resized_img(path,segmap=False):
    if segmap:
        return Image.open(path).resize((256, 256))
    else:
        return Image.open(path).convert('RGB').resize((256, 256))

def check_matching_pair(segmap_path, photo_path):
    segmap_identifier = os.path.basename(segmap_path).replace('_gtFine_labelIds', '')
    photo_identifier = os.path.basename(photo_path).replace('_leftImg8bit', '')

    assert segmap_identifier == photo_identifier, \
        "[%s] and [%s] don't seem to be matching. Aborting." % (segmap_path, photo_path)

def boundary(raw_input):#, save_path, save_name):
    """
    calculate boundary mask & save
    :param raw_input: *instanceIds image
    :param save_path: city name
    :param save_name: boundary mask name
    :return:
    """
    # process instance mask
    instance_mask = Image.open(raw_input)#.resize((256, 256))
    width = instance_mask.size[0]
    height = instance_mask.size[1]
    mask_array = np.array(instance_mask)
    # print(width,height)
    # define the boundary mask
    boundary_mask = np.zeros((height, width), dtype=np.uint8)  # 0-255

    # perform boundary calculate: the center pixel_id is differ from the 4-nearest pixels_id
    for i in range(1, height-1):
        for j in range(1, width-1):
            if mask_array[i, j] != mask_array[i - 1, j] \
                    or mask_array[i, j] != mask_array[i + 1, j] \
                    or mask_array[i, j] != mask_array[i, j - 1] \
                    or mask_array[i, j] != mask_array[i, j + 1]:
                boundary_mask[i, j] = 255
    boundary_image = Image.fromarray(np.uint8(boundary_mask))
    return boundary_image.resize((256, 256))

def process_cityscapes(gtFine_dir, leftImg8bit_dir, output_dir, phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir, exist_ok=True)
    # os.makedirs(savedir + 'A', exist_ok=True)
    # os.makedirs(savedir + 'B', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)
        # print(savedir_boundary)
    #txt files
    # filename = "val_domain.txt" if phase == 'val' else "train_domain.txt"
    # outF = open(filename, "w")
    # textList = ["One", "Two", "Three", "Four", "Five"]
    # for line in textList:
    #     print(line, file=outF)


    segmap_expr = os.path.join(gtFine_dir, phase) + "/*/*_labelIds.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    instance_expr = os.path.join(gtFine_dir, phase) + "/*/*_instanceIds.png"
    instance_paths = glob.glob(instance_expr)
    instance_paths = sorted(instance_paths)

    photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*/*_leftImg8bit.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    assert len(segmap_paths) == len(photo_paths), \
        "%d images that match [%s], and %d images that match [%s]. Aborting." % (len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)
    num = 2975 if phase == 'val' else 0
    for i, (segmap_path,instance_path, photo_path) in enumerate(zip(segmap_paths,instance_paths, photo_paths)):
        no = i+num
        check_matching_pair(segmap_path, photo_path)
        # segmap = load_resized_img(segmap_path,True)
        boundary_image = boundary(instance_path)
        photo = load_resized_img(photo_path)
        # boundary_image.save('deneme1.jpg',format='JPEG')
        # empty_channel = Image.new('L', (256, 256), color=0)
        #
        # segmap_boundary = Image.merge('RGB', ( segmap,boundary_image, empty_channel))
        #data for pix2pix where the two images are placed side-by-side
        sidebyside = Image.new('RGB', (512, 256))
        sidebyside.paste(boundary_image, (256, 0))
        sidebyside.paste(photo, (0, 0))
        savepath = os.path.join(savedir, "%d.jpg" % i)
        sidebyside.save(savepath, format='JPEG', subsampling=0, quality=100)
        # print("{domain}/{domain}_%d.jpg" % no, file=outF)

        # savepath_boundary = os.path.join(savedir_boundary, "boundary_map_%d.jpg" % no)
        # boundary_image.save(savepath_boundary, format='PNG', subsampling=0, quality=100)
        #
        # savepath_image = os.path.join(savedir_image, "image_%d.jpg" % no)
        # photo.save(savepath_image, format='PNG', subsampling=0, quality=100)
        #
        # savepath_semantic = os.path.join(savedir_semantic, "semantic_map_%d.jpg" % no)
        # segmap.save(savepath_semantic, format='PNG', subsampling=0, quality=100)
        # # data for cyclegan where the two images are stored at two distinct directories
        # savepath = os.path.join(savedir + 'A', "%d_A.jpg" % i)
        # photo.save(savepath, format='JPEG', subsampling=0, quality=100)
        # savepath = os.path.join(savedir + 'B', "%d_B.jpg" % i)
        # segmap.save(savepath, format='JPEG', subsampling=0, quality=100)

        if i % (len(segmap_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (no, len(segmap_paths)+num, output_dir))
    # outF.close()








if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gtFine_dir', type=str, required=True,
                        help='Path to the Cityscapes gtFine directory.')
    parser.add_argument('--leftImg8bit_dir', type=str, required=True,
                        help='Path to the Cityscapes leftImg8bit_trainvaltest directory.')
    parser.add_argument('--output_dir', type=str, required=True,
                        default='./datasets/cityscapes',
                        help='Directory the output images will be written to.')
    opt = parser.parse_args()

    print(help_msg)
    #
    output_dir ="boundary_image"
    print('Preparing Cityscapes Dataset for train phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, output_dir, "train")
    print('Preparing Cityscapes Dataset for val phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, output_dir, "val")

    print('Done')



