import torch
from pathlib import Path

class Config():
    CROP_HEIGHT = 1248
    CROP_WIDTH = 1664
    TARGET_IMAGE_RATIO = 0.75
    TARGET_HEIGHT = 1920
    TARGET_WIDTH = 2560

    # train loop
    ROOT = "./datasets/2nd"
    TRAIN_BATCH_SIZE = 8
    TRAIN_NUM_WORKERS = 2 # TODO: change later to adapt to the local circumstance
    TRAIN_NUM_GPUS = 2
    LOG_EVERY_N_STEPS = 1
    TRAIN_LOG_FOLDER = str(Path(__file__).absolute().parent)
    CHECKPOINT_PATH = ""
    MODEL_NAME = "SegFormer"
    LOSS_NAME = "DiceLoss"
    OPTIMIZER_NAME = "Adam"
    LR = 0.001
    SHUFFLE = True
    RESIZED_HEIGHT = 480 # height and width should be divisible by 32
    RESIZED_WIDTH = 640
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_EPOCHS = 1
    MIN_EPOCHS = 1
    NUM_CLASSES = 5

    LABELMAP_TXT_PATH = "./labelmap.txt"

    # code
    SUCCESS = 1
    FAILURE = 0