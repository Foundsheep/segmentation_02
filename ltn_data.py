import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import lightning as L
import random

from utils import erase_coloured_text_and_lines, get_transforms, adjust_ratio_and_convert_to_numpy
from args_default import Config
import json
import shutil

class SPRDataset(Dataset):
    """
    1. root 폴더 아래에 'annotated', 'raw' 폴더가 있는 것을 가정한다.
    2. 'annotated' 폴더 안에 'labelmap.txt' 파일로서 segmentation mask 이미지의 값과 라벨을
       설명하는 텍스트 파일이 있는 것을 가정한다.
    3. 'raw' 폴더 내 파일명과 'annotated' 폴더 내 파일명이 동일하다고 가정한다.(확장자 제외)
       -> cvat을 이용한 마스크 이미지의 확장자는 항상 png여서 'raw' 폴더 내 이미지
          확장자가 jpg인 경우 확장자만 바꾼 파일명을 사용하여 탐색한다.
    """
    def __init__(
        self, 
        ds_root,
        is_train=True, 
        transforms=None
        ):
        super().__init__()
        self.ds_root = Path(ds_root)
        self.folder_preprocessed = self.ds_root / "preprocessed"
        self.folder_annotated = self.ds_root / "annotated"
        self.is_train = is_train
        self.transforms = transforms
        self.image_list, self.label_list, self.label_txt = self._read_paths()
        self.num_classes = len(self.label_txt)
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = self.label_list[idx]

        # read image
        img = Image.open(img_path)
        img = adjust_ratio_and_convert_to_numpy(img)
        label_rgb = Image.open(label_path)
        label_rgb = adjust_ratio_and_convert_to_numpy(label_rgb)

        # # ------------ when to make masks in N, C, H, W
        # label_mask = np.zeros((label_rgb.shape[0], label_rgb.shape[1], self.num_classes))

        # # convert label to multi dimensional [0, 1] valued image
        # for txt_idx, txt in enumerate(self.label_txt):
        #     divider_1 = txt.find(":")
        #     divider_2 = txt.find("::")

        #     label_name = txt[:divider_1]
        #     label_value = txt[divider_1+1:divider_2]
            
        #     rgb_values = list(map(int, label_value.split(",")))

        #     x, y = np.where(
        #                 (
        #                     (label_rgb[:, :, 0] == rgb_values[0]) & 
        #                     (label_rgb[:, :, 1] == rgb_values[1]) & 
        #                     (label_rgb[:, :, 2] == rgb_values[2])
        #                 )
        #             )
        #     label_mask[x, y, txt_idx] = 1
        # # ---------------------------------------------
        
        label_mask = np.zeros((label_rgb.shape[0], label_rgb.shape[1]))
        
        # convert label to integer class numbers
        for txt_idx, txt in enumerate(self.label_txt):
            divider_1 = txt.find(":")
            divider_2 = txt.find("::")

            label_name = txt[:divider_1]
            label_value = txt[divider_1+1:divider_2]
            
            rgb_values = list(map(int, label_value.split(",")))

            x, y = np.where(
                        (
                            (label_rgb[:, :, 0] == rgb_values[0]) & 
                            (label_rgb[:, :, 1] == rgb_values[1]) & 
                            (label_rgb[:, :, 2] == rgb_values[2])
                        )
                    )
            
            label_mask[x, y] = txt_idx

        if self.transforms:
            augmented = self.transforms(image=img, mask=label_mask)
            img = augmented["image"]
            label_mask = augmented["mask"]
            
        # if idx % 10 == 0:
        #     Image.fromarray(img.numpy().transpose(1, 2, 0).astype(np.uint8)).save(f"temp_images/img_{str(idx).zfill(5)}.png")
            
        #     label_restored = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        #     for label_idx, tmp_txt in enumerate(self.label_txt):
        #         divider_1 = txt.find(":")
        #         divider_2 = txt.find("::")

        #         label_name = txt[:divider_1]
        #         label_value = txt[divider_1+1:divider_2]
                
        #         rgb_values = list(map(int, label_value.split(",")))

        #         x, y = np.where(label_mask.numpy() == label_idx)
        #         label_restored[x, y, :] = rgb_values
            
        #     Image.fromarray(label_restored.astype(np.uint8)).save(f"temp_images/lab_{str(idx).zfill(5)}.png")
        return img, label_mask
    
    def _read_paths(self):
        # image_list = list(self.folder_preprocessed.glob("*.png")) + list(self.folder_preprocessed.glob("*.jpg"))
        image_list = list(self.folder_preprocessed.glob("*"))

        # if processed files have .png extension, keep it, else convert it to match labels' original format
        label_list = [
            (
                self.folder_annotated
                / Path(
                    p.parts[-1]
                    if p.parts[-1].endswith(".png")
                    else f"{p.parts[-1][:-4]}.png"
                )
            )
            for p in image_list
        ]

        assert len(image_list) == len(label_list), f"{len(image_list) = }, {len(label_list)}"
        # to convert label image to multi dimensional [0,1] valued image
        label_map_path = self.folder_annotated / Path("labelmap.txt")
        with open(str(label_map_path), "r") as f:

            # first line is asuumed to have title
            label_txt = f.readlines()[1:]

        return image_list, label_list, label_txt
    

        

class SPRDataModule(L.LightningDataModule):
    def __init__(self, root, batch_size, shuffle, train_num_workers, labeltxt_path, data_split=True, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_num_workers = train_num_workers
        self.data_split = data_split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.folder_root = Path(self.root).absolute()
        self.folder_raw = self.folder_root / "raw"
        self.folder_preprocessed = self.folder_root / "preprocessed"
        self.folder_annotated = self.folder_root / "annotated"
        
        with open(str(labeltxt_path), "r") as f:
            # first line is asuumed to have title
            self.labelmap_txt = f.readlines()[1:]

        if not self.folder_annotated.exists():
            self.gather_data()
    def prepare_data(self):
        """
        this part prepares data that are processed only once on CPU
        and don't need to be done more than that
        """
        self._pre_process_image()
        self._make_mapping_dict()
        
        if self.data_split:
            split_result = self._split_data(
                root=self.root,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                shuffle=self.shuffle
            )
            if split_result == Config.FAILURE:
                raise Exception("Data split and storage didn't succeed")

    def _pre_process_image(self):
        """
        preprocess image to get rid of potential unnecessary marks on it
        and save the preprocessed images in a dedicated folder
        """
        # if annotated folder doesn't exist, it doesn't proceed
        if not self.folder_annotated.exists():
            print(f"annotated folder doesn't exist, so returning None\nannotated folder set to [{str(self.folder_annotated.absolute())}]")
            return None, None, None

        # if pre-processed data already exist
        if self.folder_preprocessed.exists() and len(list(self.folder_preprocessed.glob("*"))) > 0:
            print(f"preprocessed folder exists")
        
        else:
            # TODO: * is used here. Might be better with .jpg and .png selectively
            glob_raw = self.folder_raw.glob("*")

            if not self.folder_preprocessed.exists():
                self.folder_preprocessed.mkdir()
                print(f"preprocesed folder: [{str(self.folder_preprocessed)}] is made")
            
            cnt = 0
            for raw_file_name in glob_raw:
                img_preprocessed = erase_coloured_text_and_lines(str(raw_file_name))
                img_name = raw_file_name.parts[-1]
                path_preprocessed = self.folder_preprocessed / Path(img_name)
                cv2.imwrite(str(path_preprocessed), img_preprocessed)
                cnt += 1

            print(f"[{cnt}] number of preprocessed files made")

    def _split_data(self, root, train_ratio, val_ratio, test_ratio, shuffle=True):
        """
        split data physically in order to make dataset load data
        through file paths. Perhaps, data split requires to be done manually,
        and if so, this function doesn't go processed
        """
        failure = Config.FAILURE
        success = Config.SUCCESS

        # check the ratio
        sum_ratio = train_ratio + val_ratio + test_ratio
        if str(sum_ratio) != "1.0" and str(sum_ratio) != "1":
            return failure

        # get the number of all files
        if not self.folder_root.exists() or \
            not self.folder_annotated.exists() or \
            not self.folder_preprocessed.exists():
            print("necessary folders don't exist")
            return failure
        
        files_preprocessed = sorted(list(self.folder_preprocessed.glob("*")))
        num_files = len(files_preprocessed)
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
        train_folder = self.folder_root / "train"
        val_folder = self.folder_root / "val"
        test_folder = self.folder_root / "test"

        # if seperated before
        if train_folder.exists():
            print("seperated data exist already!")
            return success

        files_annotated = sorted(list(self.folder_annotated.glob("*.png")) + list(self.folder_annotated.glob("*.jpg")))

        # make commonly necessary folders
        if num_train > 0:
            train_folder.mkdir()
            (train_folder / "annotated").mkdir()
            (train_folder / "preprocessed").mkdir()

        if num_val > 0:
            val_folder.mkdir()
            (val_folder / "annotated").mkdir()
            (val_folder / "preprocessed").mkdir()

        if num_test > 0:
            test_folder.mkdir()
            (test_folder / "annotated").mkdir()
            (test_folder / "preprocessed").mkdir()

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
            path_preprocessed = files_preprocessed[index]
            path_annotated = files_annotated[index]

            copy_path_annotated = train_folder / "annotated" / path_annotated.name
            copy_path_preprocessed = train_folder / "preprocessed" / path_preprocessed.name
            copy_path_labelmap_txt = train_folder / "annotated" / self.labelmap_txt.name

            copy_path_annotated.write_bytes(path_annotated.read_bytes())
            copy_path_preprocessed.write_bytes(path_preprocessed.read_bytes())
            copy_path_labelmap_txt.write_bytes(self.labelmap_txt.read_bytes())

            count += 1
        print(f"train saved [{count}] files, {num_train = }")

        # --- val
        count = 0
        for index in indices_val:
            path_preprocessed = files_preprocessed[index]
            path_annotated = files_annotated[index]

            copy_path_annotated = val_folder / "annotated" / path_annotated.name
            copy_path_preprocessed = val_folder / "preprocessed" / path_preprocessed.name
            copy_path_labelmap_txt = val_folder / "annotated" / self.labelmap_txt.name

            copy_path_annotated.write_bytes(path_annotated.read_bytes())
            copy_path_preprocessed.write_bytes(path_preprocessed.read_bytes())
            copy_path_labelmap_txt.write_bytes(self.labelmap_txt.read_bytes())

            count += 1
        print(f"val saved [{count}] files, {num_val = }")

        # --- test
        count = 0
        for index in indices_test:
            path_preprocessed = files_preprocessed[index]
            path_annotated = files_annotated[index]

            copy_path_annotated = test_folder / "annotated" / path_annotated.name
            copy_path_preprocessed = test_folder / "preprocessed" / path_preprocessed.name
            copy_path_labelmap_txt = test_folder / "annotated" / self.labelmap_txt.name

            copy_path_annotated.write_bytes(path_annotated.read_bytes())
            copy_path_preprocessed.write_bytes(path_preprocessed.read_bytes())
            copy_path_labelmap_txt.write_bytes(self.labelmap_txt.read_bytes())

            count += 1
        print(f"test saved [{count}] files, {num_test = }")
        return success

    def _make_mapping_dict(self):
        """
        make dictionaries that map label to name and name to label
        so that they could be later used in inference to show the labels' correspondent name
        """
        root_as_p = Path(self.root)                
        label_to_name_map = {}
        name_to_label_map = {}
        for txt_idx, txt in enumerate(self.labelmap_txt):
            divider_1 = txt.find(":")
            label_name = txt[:divider_1]

            label_to_name_map[txt_idx] = label_name
            name_to_label_map[label_name] = txt_idx

        # save to json files for inference
        save_path_1 = root_as_p / "label_to_name.json"
        save_path_2 = root_as_p / "name_to_label.json"
        
        if not save_path_1.exists() and not save_path_2.exists():
            with open(str(save_path_1), "w") as outfile:
                json.dump(label_to_name_map, outfile)
            with open(str(save_path_2), "w") as outfile:
                json.dump(name_to_label_map, outfile)                
            print("json files saved for inference!")
        else:
            print("either of json files exists, so didn't make them all")   

    def gather_data(self):
        # create the folders first
        if not self.folder_raw.exists():
            self.folder_raw.mkdir()
        if not self.folder_annotated.exists():
            self.folder_annotated.mkdir()
        
        files_raw = self.folder_root.glob("*/raw/*.jpg")
        files_annotated = self.folder_root.glob("*/annotated/*.png")

        count = 0
        for p in files_raw:
            destination = self.folder_raw / p.name
            destination.write_bytes(p.read_bytes())
            count += 1
        print(f"[{count}] raw files are gathered")
        
        count = 0
        for p in files_annotated:
            destination = self.folder_annotated / p.name
            destination.write_bytes(p.read_bytes())
            count += 1
        print(f"[{count}] annotated files are gathered")

    def setup(self, stage: str):
        if stage == "fit":
            self.ds_train = SPRDataset(
                ds_root=self.root + "/train",
                is_train=True,
                transforms=get_transforms(True)
            )
            self.ds_val = SPRDataset(
                ds_root=self.root + "/val",
                is_train=False,
                transforms=get_transforms(False)
            )
        elif stage == "validate":
            self.ds_val = SPRDataset(
                ds_root=self.root + "/val",
                is_train=False,
                transforms=get_transforms(False)
            )
        elif stage == "test":
            self.ds_test = SPRDataset(
                ds_root=self.root + "/test",
                is_train=False,
                transforms=get_transforms(False)
            )
        elif stage == "predict":
            self.ds_predict = SPRDataset(
                ds_root=self.root + "/predict",
                is_train=False,
                transforms=get_transforms(False)
            )
        else:
            self.ds_train = SPRDataset(
                ds_root=self.root + "/train",
                is_train=True,
                transforms=get_transforms(True)
            )
    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=self.batch_size,
            shuffle=self.shuffle, 
            num_workers=self.train_num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=self.batch_size, 
            num_workers=self.train_num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test, 
            batch_size=self.batch_size, 
            num_workers=self.train_num_workers,
            persistent_workers=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.ds_predict, 
            batch_size=self.batch_size, 
            num_workers=self.train_num_workers,
            persistent_workers=True
        )