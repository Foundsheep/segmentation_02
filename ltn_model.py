from torch import optim
import lightning as L
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import datetime
import traceback

from torch.nn.functional import softmax

# TODO:
# 1. get model files in local host

class SPRSegmentModel(L.LightningModule):
    def __init__(
        self, 
        model_name,
        loss_name,
        optimizer_name, 
        lr, 
        num_classes=5,
        backbone_name=None, 
        use_early_stop=False, 
        momentum=0., 
        weight_decay=0.
        ):
        super().__init__()
        self.model_name = model_name
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.num_classes = num_classes
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.backbone_name = backbone_name
        self.use_early_stop = use_early_stop
        self.model = self._load_model(self.model_name, self.backbone_name)

        if loss_name == "DiceLoss":
            self.loss_fn = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        elif loss_name == "FocalLoss":
            self.loss_fn = smp.losses.FocalLoss(mode="multiclass")
        elif loss_name == "TverskyLoss":
            self.loss_fn = smp.losses.TverskyLoss(mode="multiclass", from_logits=True)
        elif loss_name == "JaccardLoss":
            self.loss_fn = smp.losses.JaccardLoss(mode="multiclass", from_logits=True)
        elif loss_name == "LovaszLoss":
            self.loss_fn = smp.losses.LovaszLoss(mode="multiclass", from_logits=True)
        else:
            print(f"Provided loss name is wrong. {loss_name = }")


        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif optimizer_name == "AdamW":
            if weight_decay == 0:
                self.weight_decay = 0.01
            self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            print(f"Provided optimizer name is wrong. {optimizer_name = }")
            
        # self.save_hyperparameters(ignore=["loss_fn", "optimizer", "use_early_stop"])
        self.save_hyperparameters()

    def _load_model(self, model_name, backbone_name):
        model = None
        try:
            if model_name == "UnetPlusPlus":
                if backbone_name is None or backbone_name == "":
                    model = smp.UnetPlusPlus(
                        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                        classes=self.num_classes,     # model output channels (number of classes in your dataset)
                    )
                else:
                    model = smp.UnetPlusPlus(
                        encoder_name=backbone_name,
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=self.num_classes,
                    )
            elif model_name == "DeepLabV3Plus":
                if backbone_name is None or backbone_name == "":
                    model = smp.DeepLabV3Plus(
                        encoder_name="resnet18",
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=self.num_classes,
                    )
                else:
                    model = smp.DeepLabV3Plus(
                        encoder_name=backbone_name,
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=self.num_classes,
                    )
            elif model_name == "SegFormer":
                if backbone_name is None or backbone_name == "":
                    model = smp.Unet(
                        encoder_name="mit_b1",
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=self.num_classes,
                    )
                else:
                    model = smp.Unet(
                        encoder_name=backbone_name,
                        encoder_weights="imagenet",
                        in_channels=3,
                        classes=self.num_classes,
                    )
            else:
                print(f"Provided model name is wrong. {model_name = }")
        except Exception as e:
            print(e)
            traceback.print_exc()
        return model
    
    def _count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def forward(self, x):
        x = self._preprocess(x)
        x = x.to(self.device)
        self.model.to(self.device)
        return self.model(x)

    def _preprocess(self, images):
        def _to_torch_tensor(img):
            if isinstance(img, Image.Image):
                x = np.array(img)
                x = torch.from_numpy(x).permute(2, 0, 1)
            elif isinstance(img, np.ndarray):
                x = torch.from_numpy(img).permute(2, 0, 1)
            elif isinstance(img, torch.Tensor):
                x = img
            else:
                raise TypeError(f"image should be one of the PIL.Image.Image or numpy.ndarray or torch.Tensor. Input type is {type(img)}")
            return x

        xs = []
        if np.array(images).size == 3:
            xs.append(_to_torch_tensor(images))
        else:
            for img in images:
                xs.append(_to_torch_tensor(img))

        return torch.stack(xs, dim=0).to(dtype=torch.float, device=self.device)
    
    def shared_step(self, batch, stage):
        x, y = batch
        preds = self.model(x)
        loss = self.loss_fn(preds, y.long())

        preds_softmax = softmax(preds, dim=1)
        preds_argmax = preds_softmax.argmax(dim=1)
        # y_argmax = y.argmax(dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            preds_argmax.long(), 
            y.long(),
            mode="multiclass",
            num_classes=self.num_classes
        )
        
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro-imagewise")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro-imagewise")

        self.log(f"{stage}_IoU", iou, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log(f"{stage}_accuracy", accuracy, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log(f"{stage}_f1_score", f1_score, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=False) 
        
        
        return {"loss": loss, "iou": iou, "accuracy":accuracy, "f1_score": f1_score}

    def on_fit_start(self):
        tb = self.logger.experiment
        
        layout = {
            "training_result": {
                "loss": ["Multiline", ["train_loss", "val_loss"]],
                "accuracy": ["Multiline", ["train_accuracy", "val_accuracy"]],
                "IoU": ["Multiline", ["train_IoU", "val_IoU"]],
                "f1_score": ["Multiline", ["train_f1_score", "val_f1_score"]],
            }
        }

        tb.add_custom_scalars(layout)
        
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        return self.optimizer
    
    def configure_callbacks(self):        
        callbacks = []
        if self.use_early_stop:
            early_stop = EarlyStopping(
                monitor="val_IoU",
                patience=4,
                mode="max",
                verbose=True
            )
            callbacks.append(early_stop)

        checkpoint_save_last = ModelCheckpoint(
            save_last=True,
            filename="{epoch}-{step}-{train_loss:.4f}_last"
        )
        callbacks.append(checkpoint_save_last)

        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="{epoch}-{step}-{val_loss:.3f}-{val_iou:.3f}"
        )
        callbacks.append(checkpoint)
        return callbacks