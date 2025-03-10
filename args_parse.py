import argparse
from args_default import Config

def get_args():
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument("--root", type=str, default=Config.ROOT)
    parser.add_argument("--train_batch_size", type=int, default=Config.TRAIN_BATCH_SIZE)
    parser.add_argument("--train_num_workers", type=int, default=Config.TRAIN_NUM_WORKERS)
    parser.add_argument("--train_num_gpus", type=int, default=Config.TRAIN_NUM_GPUS)
    parser.add_argument("--log_every_n_steps", type=int, default=Config.LOG_EVERY_N_STEPS)
    parser.add_argument("--train_log_folder", type=str, default=Config.TRAIN_LOG_FOLDER)
    parser.add_argument("--checkpoint_path", type=str, default=Config.CHECKPOINT_PATH)
    parser.add_argument("--model_name", type=str, default=Config.MODEL_NAME)
    parser.add_argument("--backbone_name", type=str, default="")
    parser.add_argument("--loss_name", type=str, default=Config.LOSS_NAME)
    parser.add_argument("--optimizer_name", type=str, default=Config.OPTIMIZER_NAME)
    parser.add_argument("--lr", type=float, default=Config.LR)
    parser.add_argument("--use_early_stop", type=bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--shuffle", type=bool, default=Config.SHUFFLE)
    parser.add_argument("--hparams_path", type=str, default="")
    parser.add_argument("--resized_height", type=int, default=Config.RESIZED_HEIGHT)
    parser.add_argument("--resized_width", type=int, default=Config.RESIZED_WIDTH)
    parser.add_argument("--device", type=str, default=Config.DEVICE)
    parser.add_argument("--max_epochs", type=int, default=Config.MAX_EPOCHS)
    parser.add_argument("--min_epochs", type=int, default=Config.MIN_EPOCHS)
    parser.add_argument("--num_classes", type=int, default=Config.NUM_CLASSES)

    # train test split
    parser.add_argument("--data_split", type=bool, default=True)
    parser.add_argument("--split_root", type=str, default="")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)

    # inference
    parser.add_argument("--img_path", type=str, default="")
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--labelmap_txt_path", type=str, default=Config.LABELMAP_TXT_PATH)


    parser.add_argument("--predict", action=argparse.BooleanOptionalAction)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction)
    parser.add_argument("--resume_training", action=argparse.BooleanOptionalAction)
    parser.add_argument("--fast_dev_run", action=argparse.BooleanOptionalAction)


    args = parser.parse_args()
    return args