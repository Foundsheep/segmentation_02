import torch
import lightning as L

import datetime
import yaml
import io
from args_parse import get_args
from ltn_model import SPRSegmentModel
from ltn_data import SPRDataModule
from utils import post_process, adjust_ratio_and_convert_to_numpy, get_transforms
from PIL import Image
from pathlib import Path


import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)


def train(args):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')

    dm = SPRDataModule(
        root=args.root,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        train_num_workers=args.train_num_workers,
        labeltxt_path=args.labelmap_txt_path,
        data_split=args.data_split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    if args.backbone_name:
        default_root_dir = f"{args.train_log_folder}/{timestamp}_{args.model_name}_{args.backbone_name}_{args.loss_name}_batch{args.train_batch_size}_epochs{args.max_epochs}_lr{args.lr}"
    else:
        default_root_dir = f"{args.train_log_folder}/{timestamp}_{args.model_name}_{args.loss_name}_batch{args.train_batch_size}_epochs{args.max_epochs}_lr{args.lr}"

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.train_num_gpus,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=default_root_dir,
        fast_dev_run=args.fast_dev_run,
        # strategy="ddp_find_unused_parameters_true",
        # strategy="ddp",
        check_val_every_n_epoch=1 # TODO: change        
    )

    if args.resume_training:
        print("*************** TRAINING RESUMED ***************")
        with open(args.hparams_path) as stream:
            hp = yaml.safe_load(stream)            
        model = SPRSegmentModel(
                model_name=hp["model_name"],
                backbone_name=hp["backbone_name"],
                loss_name=hp["loss_name"],
                lr=hp["lr"],
                optimizer_name=hp["optimizer_name"],
                use_early_stop=hp["use_early_stop"],
                momentum=hp["momentum"],
                weight_decay=hp["weight_decay"],
        )
        trainer.fit(model=model, datamodule=dm, ckpt_path=args.checkpoint_path)
    else:
        print("*************** TRAINING STARTS ***************")
        model = SPRSegmentModel(
            model_name=args.model_name,
            backbone_name=args.backbone_name,
            loss_name=args.loss_name,
            optimizer_name=args.optimizer_name,
            lr=args.lr,
            use_early_stop=args.use_early_stop,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        trainer.fit(model=model, datamodule=dm)
    print("*************** TRAINING DONE ***************")
    print("*********************************************")

    if not args.fast_dev_run:
        trainer.test(model=model, datamodule=dm)

        example_input = torch.randn(4, 3, args.resized_height, args.resized_width)
        # script_model = model.to_torchscript(method="trace", example_inputs=example_input, strict=False)    
        # torch.jit.save(script_model, f"{default_root_dir}/script_model.pt")
        # print("Torch script model has been saved!")

        model.to_onnx(
            file_path=f"{default_root_dir}/model.onnx",
            input_sample=example_input,
            export_params=True
        )
    print("**************** ONNX SAVED *****************")
    print("*********************************************")

def predict(args):
    print("************* PREDICTION START **************")
    print("*********************************************")
    model = SPRSegmentModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        backbone_name=args.backbone_name,
        loss_name=args.loss_name,
        optimizer_name=args.optimizer_name,
        lr=args.lr,
        use_early_stop=args.use_early_stop,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    transforms = get_transforms(is_train=False)

    if args.folder_to_predict:
        if args.data_to_predict:
            raise "Either 'folder_to_predict' or 'data_to_predict' is allowed in prediction arguments"
        iteration_list = (
            list(Path(args.folder_to_predict).glob("*.png")) +
            list(Path(args.folder_to_predict).glob("*.jpg")) + 
            list(Path(args.folder_to_predict).glob("*.JPG"))
        )
    
    names = []
    test_data = []
    for elem in iteration_list:
        tmp_img = Image.open(elem if args.folder_to_predict else io.BytesIO(elem))
        tmp_img = adjust_ratio_and_convert_to_numpy(tmp_img)
        tmp_img = transforms(image=tmp_img)["image"]
        test_data.append(tmp_img)
        names.append(elem.name)
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
    post_process(outputs, names, args.labelmap_txt_path)
    print("************* PREDICTION DONE ***************")
    print("*********************************************")
    

def tune_func(args):    
    # GPU performance increases!
    torch.set_float32_matmul_precision('medium')

    dm = SPRDataModule(
        root=args["root"],
        batch_size=args["train_batch_size"],
        shuffle=args["shuffle"],
        dl_num_workers=args["train_num_workers"],
        labeltxt_path=args["labelmap_txt_path"],
        data_split=args["data_split"],
        train_ratio=args["train_ratio"],
        val_ratio=args["val_ratio"],
        test_ratio=args["test_ratio"]
    )
    
    model = SPRSegmentModel(
        model_name=args["model_name"],
        backbone_name=args["backbone_name"],
        loss_name=args["loss_name"],
        optimizer_name=args["optimizer_name"],
        lr=args["lr"],
        use_early_stop=args["use_early_stop"],
        momentum=args["momentum"],
        weight_decay=args["weight_decay"]
    )

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

def tune(args):
    # reset_dir_for_ray_session = str(Path(__file__).parent.parent.parent.absolute())
    # ray.init(_temp_dir=reset_dir_for_ray_session)

    search_space = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([4, 6, 8]),
    }

    search_space.update(vars(args))    

    # scaling_config = ScalingConfig(
    #     num_workers=2, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    # )

    scaling_config = ScalingConfig(
        use_gpu=True
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )

    # Number of sampls from parameter space
    num_samples = 10
    
    scheduler = ASHAScheduler(max_t=args.max_epochs, grace_period=1, reduction_factor=2)

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        tune_func,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

if __name__ == "__main__":
    args = get_args()
    
    print("**********************************************")
    print("Args provided as:")
    print(dict(vars(args)))
    print("**********************************************")

    if args.train and not args.predict:
        train(args)
    elif not args.train and args.predict:
        predict(args)
    else:
        raise ValueError("either '--predict' or 'train' should be declared")