from pathlib import Path
import random

import torch
from args_default import Config
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import traceback
import torch.nn.functional as Fs
import datetime

def seperate_data(root, train_ratio, val_ratio, test_ratio, shuffle=True):
    failure = Config.FAILURE
    success = Config.SUCCESS

    # check the ratio
    sum_ratio = train_ratio + val_ratio + test_ratio
    if str(sum_ratio) != "1.0" and str(sum_ratio) != "1":
        return failure
    
    # get the number of all files
    root_folder = Path(root)
    annotated_folder = root_folder / "annotated"
    raw_folder = root_folder / "raw"

    if not root_folder.exists() or not annotated_folder.exists() or not raw_folder.exists():
        print("necessary folders don't exist")
        return failure
    
    raw_files = sorted(list(raw_folder.glob("*")))
    num_files = len(raw_files)
    indices = np.arange(num_files)

    # get the numbers for each dataset
    num_train = round(num_files * train_ratio)
    num_val = round(num_files * val_ratio)
    num_test = num_files - num_train - num_val
    print(f"numbers of file to make: train=[{num_train}], val=[{num_val}], test=[{num_test}]")

    # check the sanity
    if train_ratio > 0 and num_train < 1:
        print(f"ratio is greater than 0[{train_ratio = }],  but that fraction didn't make any number to select in files[{num_train = }]")
        return failure
    elif val_ratio > 0 and num_val < 1:
        print(f"ratio is greater than 0[{val_ratio = }],  but that fraction didn't make any number to select in files[{num_val = }]")
        return failure
    elif test_ratio > 0 and num_test < 1:
        print(f"ratio is greater than 0[{test_ratio = }],  but that fraction didn't make any number to select in files[{num_test = }]")
        return failure

    # prepare the directories for copies
    train_folder = root_folder / "train"
    val_folder = root_folder / "val"
    test_folder = root_folder / "test"

    # if seperated before
    if train_folder.exists():
        print("seperated data exist already!")
        return success

    annotated_files = sorted(list(annotated_folder.glob("*.png")) + list(annotated_folder.glob("*.jpg")))
    labelmap_txt = list(annotated_folder.glob("labelmap.txt"))[0]

    # make commonly necessary folders
    if num_train > 0:
        train_folder.mkdir()
        (train_folder / "annotated").mkdir()
        (train_folder / "raw").mkdir()

    if num_val > 0:
        val_folder.mkdir()
        (val_folder / "annotated").mkdir()
        (val_folder / "raw").mkdir()

    if num_test > 0:
        test_folder.mkdir()
        (test_folder / "annotated").mkdir()
        (test_folder / "raw").mkdir()

    # shuffle
    if shuffle:
        random.shuffle(indices)
    
    # split data
    indices_train = indices[:num_train]
    indices_val = indices[num_train:num_train+num_val]
    indices_test = indices[num_train+num_val:num_train+num_val+num_test]

    # --- train
    count = 0
    for index in indices_train:
        path_raw = raw_files[index]
        path_annotated = annotated_files[index]

        copy_path_raw = train_folder / "raw" / path_raw.name
        copy_path_annotated = train_folder / "annotated" / path_annotated.name
        copy_path_labelmap_txt = train_folder / "annotated" / labelmap_txt.name

        copy_path_raw.write_bytes(path_raw.read_bytes())
        copy_path_annotated.write_bytes(path_annotated.read_bytes())
        copy_path_labelmap_txt.write_bytes(labelmap_txt.read_bytes())

        count += 1
    print(f"train saved [{count}] files, {num_train = }")

    # --- val
    count = 0
    for index in indices_val:
        path_raw = raw_files[index]
        path_annotated = annotated_files[index]

        copy_path_raw = val_folder / "raw" / path_raw.name
        copy_path_annotated = val_folder / "annotated" / path_annotated.name
        copy_path_labelmap_txt = val_folder / "annotated" / labelmap_txt.name

        copy_path_raw.write_bytes(path_raw.read_bytes())
        copy_path_annotated.write_bytes(path_annotated.read_bytes())
        copy_path_labelmap_txt.write_bytes(labelmap_txt.read_bytes())

        count += 1
    print(f"val saved [{count}] files, {num_val = }")

    # --- test
    count = 0
    for index in indices_test:
        path_raw = raw_files[index]
        path_annotated = annotated_files[index]

        copy_path_raw = test_folder / "raw" / path_raw.name
        copy_path_annotated = test_folder / "annotated" / path_annotated.name
        copy_path_labelmap_txt = test_folder / "annotated" / labelmap_txt.name

        copy_path_raw.write_bytes(path_raw.read_bytes())
        copy_path_annotated.write_bytes(path_annotated.read_bytes())
        copy_path_labelmap_txt.write_bytes(labelmap_txt.read_bytes())

        count += 1
    print(f"test saved [{count}] files, {num_test = }")
    return success

def erase_coloured_text_and_lines(img_path):
    # 1. 하늘색 HSV : 186, 98%, 95%
    # 2. 녹색 HSV : 118, 98%, 95%
    #               118, 98%, 65%
    #               118, 98%, 35%
    # 3. 빨강 HSV : 0, 98%, 65%
    if isinstance(img_path, str):
        img = cv2.imread(img_path)
        image = img.copy()
    elif isinstance(img_path, np.ndarray):
        image = img_path
    else:
        image = np.array(img_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 위의 기준보다 좀 더 널럴하게 Saturation 맞췄음. 아닐 경우 잘 인지 안 됨
    s_max = 255
    v_max = 255
    lower = np.array([0, s_max*0.60, v_max*0.30])
    upper = np.array([200, s_max*0.98, v_max*0.98])
    # lower = np.array([0, 128, 128])
    # upper = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((9, 9), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)  #// make dilation image

    dst = cv2.inpaint(image, dilated_mask, 5, cv2.INPAINT_NS)

    return dst

def get_transforms(is_train):
    def _lambda_contrast_fn(image, **kwargs):
        image = 255 - image
        image = image.astype(np.uint8)
        return image
    
    def _lambda_minmax_fn(image, **kwards):
        image = image.astype(np.float32)
        image = image / 255.0
        return image
    
    if is_train:
        transforms = A.Compose([
            # A.Lambda(image=_lambda_contrast_fn),
            A.RGBShift(r_shift_limit=(-50, 50), g_shift_limit=(-50, 50), b_shift_limit=(-50, 50)),
            A.ColorJitter(brightness=(0.8, 1), contrast=(0.8, 1), saturation=(0.5, 1), hue=(-0.5, 0.5)),
            A.RandomCrop(height=Config.CROP_HEIGHT, width=Config.CROP_HEIGHT, p=0.5),
            A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH, interpolation=0),
            A.HorizontalFlip(),
            A.GridDistortion(),            
            A.Blur(),
            # A.Lambda(image=_lambda_minmax_fn),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(transpose_mask=True),
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=Config.RESIZED_HEIGHT, width=Config.RESIZED_WIDTH, interpolation=0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.Lambda(image=_lambda_minmax_fn),
            ToTensorV2(transpose_mask=True),
        ])
    return transforms 


def get_label_info(labeltxt_path):
    try:
        with open(labeltxt_path, "r") as f:
            label_txt = f.readlines()[1:]
    except Exception as e:
        print(e) 
        traceback.print_exc()
        label_txt = None
        
    label_to_rgb = {}
    if label_txt is None:
        label_to_rgb[0] = [0, 0, 0] # background
        label_to_rgb[1] = [255, 0, 0] # lower
        label_to_rgb[2] = [255, 255, 0] # middle
        label_to_rgb[3] = [0, 255, 0] # rivet
        label_to_rgb[4] = [0, 0, 255] # upper
        
    else:
        for txt_idx, txt in enumerate(label_txt):
            divider_1 = txt.find(":")
            divider_2 = txt.find("::")

            label_name = txt[:divider_1]
            label_value = txt[divider_1+1:divider_2]
            rgb_values = list(map(int, label_value.split(",")))

            label_to_rgb[txt_idx] = rgb_values

    return label_to_rgb

def adjust_ratio_and_convert_to_numpy(img:Image.Image) -> Image.Image:
    target_ratio = Config.TARGET_IMAGE_RATIO
    h = img.height
    w = img.width
    ratio = h / w
    new_ratio = 0.00

    if ratio == target_ratio:
        return np.array(img)
    
    elif ratio < target_ratio:
        # print(f"img_path: [{img_path}] image's height is shorter than the standard")
        new_h = int(w * target_ratio)
        half_new_h = (new_h - h) // 2
        if 2 * half_new_h != new_h:
            another_half = new_h - half_new_h - h
        else:
            another_half = half_new_h  
        
        img_np = np.array(img)
        img_new = np.pad(img_np, ((half_new_h, another_half), (0, 0), (0, 0)), "reflect")
        # print(f"new ratio: [{img_new.shape[0] / img_new.shape[1]}], new_height: [{img_new.shape[0]}]")
        
    else:
        # print(f"img_path: [{img_path}] image's width is shorter than the standard")
        new_w = int(h / target_ratio)
        half_new_w = (new_w - w) // 2
        if 2 * half_new_w != new_w:
            another_half = new_w - half_new_w - w
        else:
            another_half = half_new_w
        
        img_np = np.array(img)
        img_new = np.pad(img_np, ((0, 0), (half_new_w, another_half), (0, 0)), "reflect")
        # print(f"new ratio: [{img_new.shape[0] / img_new.shape[1]}], new_width: [{img_new.shape[1]}]")

    new_ratio = img_new.shape[0] / img_new.shape[1]
    img_new = np.array(Image.fromarray(img_new).resize((Config.TARGET_WIDTH, Config.TARGET_HEIGHT)))
    assert f"{target_ratio :.2f}" == f"{new_ratio :.2f}", f"{target_ratio =}, {new_ratio =}"
    assert img_new.shape[0] == Config.TARGET_HEIGHT, f"{img_new.shape = }"
    return img_new

def post_process(output: torch.Tensor, names: list[str], labelmap_path: str) -> None:
    
    output = F.softmax(output, dim=1) # N, C, H, W
    output = output.argmax(1) # N, H, W
    output = output.cpu().numpy().astype(np.uint8) # N, H, W
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for idx, (out, name) in enumerate(zip(output, names)):
        # save_image(colour_image(out, labelmap_path), timestamp, str(idx).zfill(4))
        save_image(colour_image(out, labelmap_path), timestamp, name)

def save_image(image: np.ndarray, timestamp: str, file_name: str, to_w: int = None, to_h: int = None) -> None:
    image_pil = Image.fromarray(image)
    if to_w is not None and to_h is not None:
        image_pil = image_pil.resize((to_w, to_h))

    save_folder = Path(f"./inference_result_{timestamp}")
    if not save_folder.exists():
        save_folder.mkdir()
        print(f"[{str(save_folder)} is made...]")

    save_path = save_folder / f"{f'{file_name}.png'}"
    image_pil.save(save_path)
    print(f"[{file_name}.png] saved...!")

def colour_image(image: np.ndarray, labelmap_path) -> np.ndarray:
    label_info = get_label_info(labelmap_path)
    canvas = np.zeros((image.shape[0], image.shape[1], 3))
    for class_idx, rgb in label_info.items():
        x, y = np.where(image == class_idx)
        canvas[x, y, :] = rgb
        
    return canvas.astype(np.uint8)
    