# Based on PyTorch Lightning Tutorial 13 -
# SSL : https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
# Modified by Fares Abawi (@fabawi).
import logging
import os
import argparse

try:
    import comet_ml
except ImportError:
    comet_ml = None
try:
    import wandb
except ImportError:
    wandb = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    logging.warning("Matplotlib not installed. This is not needed if you run this script as --headless")

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torchvision import transforms

from models import imagebind_model
from models import lora as LoRA
from models.imagebind_model import ModalityType, load_module, save_module

logging.basicConfig(level=logging.INFO, force=True)

# Logging settings
LOG_ON_STEP = True
LOG_ON_EPOCH = True


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class ImageBindTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42, 
                 self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), 
                 lora=False, lora_rank=4, lora_checkpoint_dir="./.checkpoints/lora",
                 lora_layer_idxs=None, lora_modality_names=None,
                 linear_probing=False
                 ):
        super().__init__()
        assert not (linear_probing and lora), \
            "Linear probing is a subset of LoRA training procedure for ImageBind. " \
            "Cannot set both linear_probing=True and lora=True. " \
            "Linear probing stores params in lora_checkpoint_dir"
        self.save_hyperparameters()

        # Load full pretrained ImageBind model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        if lora:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
                
            self.model.modality_trunks.update(LoRA.apply_lora_modality_trunks(self.model.modality_trunks, rank=lora_rank,
                                                                              layer_idxs=lora_layer_idxs,
                                                                              modality_names=lora_modality_names))
            LoRA.load_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=lora_checkpoint_dir)

            # Load postprocessors & heads
            load_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=lora_checkpoint_dir)
            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
        elif linear_probing:
            for modality_preprocessor in self.model.modality_preprocessors.children():
                modality_preprocessor.requires_grad_(False)
            for modality_trunk in self.model.modality_trunks.children():
                modality_trunk.requires_grad_(False)
            for modality_postprocessor in self.model.modality_postprocessors.children():
                modality_postprocessor.requires_grad_(False)

            load_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=lora_checkpoint_dir)
            for modality_head in self.model.modality_heads.children():
                modality_head.requires_grad_(False)
                final_layer = list(modality_head.children())[-1]
                final_layer.requires_grad_(True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, 
                                betas=self.hparams.momentum_betas)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        data_a, class_a, data_b, class_b = batch

        # class_a is always "vision" according to ImageBind
        feats_a = [self.model({class_a[0]: data_a_i}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)
        # class_b could be any modality
        feats_b = [self.model({class_b[idx]: data_b_i}) for idx, data_b_i in enumerate(data_b)]
        feats_b_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0)

        if self.hparams.self_contrast:
            feats_a_b_tensor = torch.cat([feats_a_tensor.chunk(2)[0], feats_b_tensor], dim=0)
            feats_tensors = [feats_a_tensor, feats_a_b_tensor]
            temperatures = [1, self.hparams.temperature]
            contrast = ["self", "cross"]
        else:
            feats_a_b_tensor = torch.cat([feats_a_tensor, feats_b_tensor], dim=0)
            feats_tensors = [feats_a_b_tensor]
            temperatures = [self.hparams.temperature]
            contrast = ["cross"]

        # Accumulate self-contrastive loss for image and its augmentation, and modailty with image
        dual_nll = False
        for feats_idx, feats_tensor in enumerate(feats_tensors):
            # Calculate cosine similarity
            cos_sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Mask out same-modality comparisons (assuming first half is images, second half is another modality or perturbed image in case of "self")
            half_batch = cos_sim.shape[0] // 2
            same_modality_mask = torch.cat([torch.zeros(half_batch, half_batch), torch.ones(half_batch, half_batch)], dim=1).bool().to(cos_sim.device)
            cos_sim.masked_fill_(~same_modality_mask, -9e15)

            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=half_batch, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / temperatures[feats_idx]
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            nll = nll.mean()
            if not dual_nll:
                dual_nll = nll
            else:
                dual_nll += nll
                dual_nll /= 2
            # Logging loss
            self.log(mode + "_loss_" + contrast[feats_idx], nll, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            # Get ranking position of positive example
            comb_sim = torch.cat(
                [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
                dim=-1,
            )
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            # Logging ranking metrics
            self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)

        self.log(mode + "_loss", dual_nll, prog_bar=True,
                 on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
        return dual_nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")

    def on_validation_epoch_end(self):
        if self.hparams.lora:
            # Save LoRA checkpoint
            LoRA.save_lora_modality_trunks(self.model.modality_trunks, checkpoint_dir=self.hparams.lora_checkpoint_dir)
            # Save postprocessors & heads
            save_module(self.model.modality_postprocessors, module_name="postprocessors",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)
        elif self.hparams.linear_probing:
            # Save postprocessors & heads
            save_module(self.model.modality_heads, module_name="heads",
                        checkpoint_dir=self.hparams.lora_checkpoint_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ImageBind model with PyTorch Lightning and LoRA.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training ('cpu' or 'cuda')")
    parser.add_argument("--datasets_dir", type=str, default="./.datasets",
                        help="Directory containing the datasets")
    parser.add_argument("--datasets", type=str, nargs="+", default=["dreambooth"], choices=["dreambooth"],
                        help="Datasets to use for training and validation")
    parser.add_argument("--full_model_checkpoint_dir", type=str, default="./.checkpoints/full",
                        help="Directory to save the full model checkpoints")
    parser.add_argument("--full_model_checkpointing", action="store_true", help="Save full model checkpoints")
    parser.add_argument("--loggers", type=str, nargs="+", choices=["tensorboard", "wandb", "comet", "mlflow"],
                        help="Loggers to use for logging")
    parser.add_argument("--loggers_dir", type=str, default="./.logs", help="Directory to save the logs")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (Don't plot samples on start)")

    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--momentum_betas", nargs=2, type=float, default=[0.9, 0.95],
                        help="Momentum beta 1 and 2 for Adam optimizer")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for InfoNCE loss")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--self_contrast", action="store_true", help="Use self-contrast on the image modality")

    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_rank", type=int, default=4, help="Rank of LoRA layers")
    parser.add_argument("--lora_checkpoint_dir", type=str, default="./.checkpoints/lora",
                        help="Directory to save LoRA checkpoint")
    parser.add_argument("--lora_modality_names", nargs="+", type=str, default=["vision", "text"],
                        choices=["vision", "text", "audio", "thermal", "depth", "imu"],
                        help="Modality names to apply LoRA")
    parser.add_argument("--lora_layer_idxs", nargs="+", type=int,
                        help="Layer indices to apply LoRA")
    parser.add_argument("--lora_layer_idxs_vision", nargs="+", type=int,
                        help="Layer indices to apply LoRA for vision modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_text", nargs="+", type=int,
                        help="Layer indices to apply LoRA for text modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_audio", nargs="+", type=int,
                        help="Layer indices to apply LoRA for audio modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_thermal", nargs="+", type=int,
                        help="Layer indices to apply LoRA for thermal modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_depth", nargs="+", type=int,
                        help="Layer indices to apply LoRA for depth modality. Overrides lora_layer_idxs if specified")
    parser.add_argument("--lora_layer_idxs_imu", nargs="+", type=int,
                        help="Layer indices to apply LoRA for imu modality. Overrides lora_layer_idxs if specified")

    parser.add_argument("--linear_probing", action="store_true",
                        help="Freeze model and train the last layers of the head for each modality.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Create loggers
    loggers = []
    for logger in args.loggers if args.loggers is not None else []:
        if logger == "wandb":
            wandb.init(project="imagebind", config=args)
            wandb_logger = pl_loggers.WandbLogger(
                save_dir=args.loggers_dir,
                name="imagebind")
            loggers.append(wandb_logger)
        elif logger == "tensorboard":
            tensorboard_logger = pl_loggers.TensorBoardLogger(
                save_dir=args.loggers_dir,
                name="imagebind")
            loggers.append(tensorboard_logger)
        elif logger == "comet":
            comet_logger = pl_loggers.CometLogger(
                save_dir=args.loggers_dir,
                api_key=os.environ["COMET_API_KEY"],
                workspace=os.environ["COMET_WORKSPACE"],
                project_name=os.environ["COMET_PROJECT_NAME"],
                experiment_name=os.environ.get("COMET_EXPERIMENT_NAME", None),
            )
            loggers.append(comet_logger)
        elif logger == "mlflow":
            mlflow_logger = pl_loggers.MLFlowLogger(
                save_dir=args.loggers_dir,
                experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
                tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
                run_name="imagebind"
            )
            loggers.append(mlflow_logger)
        else:
            raise ValueError(f"Unknown logger: {logger}")

    # Set experiment properties
    seed_everything(args.seed, workers=True)
    torch.backends.cudnn.determinstic = True
    device_name = args.device  # "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    contrast_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)],
                                   p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    train_datasets = []
    test_datasets = []

    # Load datasets
    if "dreambooth" in args.datasets:
        from datasets.dreambooth import DreamBoothDataset
        train_datasets.append(DreamBoothDataset(
            root_dir=os.path.join(args.datasets_dir, "dreambooth", "dataset"), split="train",
            transform=ContrastiveTransformations(contrast_transforms,
                                                 n_views=2 if args.self_contrast else 1)))
        test_datasets.append(DreamBoothDataset(
            root_dir=os.path.join(args.datasets_dir, "dreambooth", "dataset"), split="test",
            transform=ContrastiveTransformations(contrast_transforms,
                                                 n_views=2 if args.self_contrast else 1)))

    if len(args.datasets) == 1:
        train_dataset = train_datasets[0]
        test_dataset = test_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        num_workers=args.num_workers,
    )

    # Visualize some examples
    if not args.headless:
        NUM_IMAGES = args.batch_size
        imgs = [torch.stack(train_dataset[idx][0], dim=0) for idx in range(NUM_IMAGES)]
        imgs = torch.stack(imgs, dim=0)
        img_grid = torchvision.utils.make_grid(imgs.reshape(-1, *imgs.shape[2:]), nrow=6, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        plt.figure(figsize=(10, 5))
        plt.title(f"Augmented image examples of the available datasets: {args.datasets}")
        plt.imshow(img_grid.cpu())
        plt.axis("off")
        plt.show()
        plt.close()

    # Parse indices of layers to apply LoRA
    lora_layer_idxs = {}
    lora_modality_names = []
    modalities = ["vision", "text", "audio", "thermal", "depth", "imu"]
    for modality_name in args.lora_modality_names:
        if modality_name in modalities:
            modality_type = getattr(ModalityType, modality_name.upper())
            lora_layer_idxs[modality_type] = getattr(args, f'lora_layer_idxs_{modality_name}', None)
            if not lora_layer_idxs[modality_type]:
                lora_layer_idxs[modality_type] = None
            lora_modality_names.append(modality_type)
        else:
            raise ValueError(f"Unknown modality name: {modality_name}")

    # Train dataset
    model = ImageBindTrain(max_epochs=args.max_epochs, batch_size=args.batch_size, lr=args.lr,
                           weight_decay=args.weight_decay, momentum_betas=args.momentum_betas,
                           temperature=args.temperature,
                           num_workers=args.num_workers, self_contrast=args.self_contrast,
                           lora=args.lora, lora_rank=args.lora_rank, lora_checkpoint_dir=args.lora_checkpoint_dir,
                           lora_layer_idxs=lora_layer_idxs if lora_layer_idxs else None,
                           lora_modality_names=lora_modality_names if lora_modality_names else None,
                           linear_probing=args.linear_probing)

    if args.full_model_checkpointing:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,
                         "callbacks": [ModelCheckpoint(monitor="val_loss", dirpath=args.full_model_checkpoint_dir,
                                                        filename="imagebind-{epoch:02d}-{val_loss:.2f}",
                                                        save_last=True, mode="min")]}
    else:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,}

    trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=1 if ":" not in device_name else [int(device_name.split(":")[1])], deterministic=True,
                      max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
                      logger=loggers if loggers else None, **checkpointing)

    trainer.fit(model, train_loader, val_loader)

