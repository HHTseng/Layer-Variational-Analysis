import glob, os, random
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d
from utils import *
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

patch_size = (33, 33)
max_patches = 20

# image_path = './T91/'
image_path = '/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/CUFED/train_CUFED/ref/'          # ground truth (high res) image folder
image_files = os.path.join(image_path, '*.png')
patch_save_dir = '/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/CUFED/CUFED_blurred_patches/'
save_selected_img = '/home/htseng/Downloads/Transfer_Learning_datasets/preprocessing_SR_images/CUFED/CUFED_blurred_images/'

Path(patch_save_dir).mkdir(parents=True, exist_ok=True)

files = glob.glob(image_files)
random.shuffle(files)
files = files[:2100]
for _ in files:
    print('processing:', _)

train_paths, val_paths = train_test_split(files, test_size=100, random_state=999)

def images_to_blur_patches(paths, blur_scale=[1, 3, 4], mode='train'):
    blur_patches = [[] for _ in range(len(blur_scale))]  # patches of various blurring scales

    for path in paths:
        img = Image.open(path).convert('RGB')
        image = img.resize(((img.width // 12) * 12, (img.height // 12) * 12), resample=Image.BICUBIC)

        # save selected images
        save_folder = os.path.join(save_selected_img, f'{mode}/')
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        image.save(path.replace(image_path, save_folder))

        # convert RGB to YCbCr images
        imgs = np.stack([np.asarray(RGB2cbcr(image, blur_scale[s])) for s in range(len(blur_scale))], axis=-1)

        # crop into patches
        img_patches = extract_patches_2d(imgs, patch_size, max_patches=max_patches, random_state=None)

        for s in range(len(blur_scale)):
            blur_patches[s].append(img_patches[..., s])

    for s in range(len(blur_scale)):
        blur_patches[s] = np.vstack(blur_patches[s])

        # save patches
        save_patches = os.path.join(patch_save_dir, f'blur_scale_{blur_scales[s]}/')
        check_path(save_patches)
        np.save(os.path.join(save_patches, f'X_{mode}.npy'), blur_patches[s])

    return blur_patches


# select blurring scales
blur_scales = [1, 3, 4, 6]

# processing images to blur patches
train_blur_patches = images_to_blur_patches(train_paths, blur_scales, 'train')  # train set
test_blur_patches = images_to_blur_patches(val_paths, blur_scales, 'test')      # test set