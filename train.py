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
from torcheval.metrics import MulticlassPrecisionRecallCurve, MulticlassAccuracy, MulticlassF1Score
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


def unique_with_index(class_idx):
    sorted_indices = torch.argsort(class_idx)
    sorted_input = class_idx[sorted_indices]
    first_occurrence = torch.cat((torch.tensor([True], device=class_idx.device), sorted_input[:-1] != sorted_input[1:]))
    return sorted_input[first_occurrence], sorted_indices[first_occurrence]


class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]


class ImageBindTrain(L.LightningModule):
    def __init__(self, lr=5e-4, weight_decay=1e-4, max_epochs=500, batch_size=32, num_workers=4, seed=42, 
                 class_masking=False, self_contrast=False, temperature=0.07,  momentum_betas=(0.9, 0.95), 
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
        
    def _calculate_and_log_class_metrics(self, sim, class_idx, mode, name, unique=False):
        num_classes = int(max(class_idx))+1
        accm = MulticlassAccuracy(average="micro", num_classes=num_classes, device=self.device)
        accm.update(input=sim, target=class_idx)
        accuracy = accm.compute()
        # accuracy = sync_and_compute(accm)
        self.log(f"{mode}/{name}_acc", accuracy, on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH,
                 prog_bar=True, batch_size=self.hparams.batch_size)
        if unique:
            # Precision & Recall
            prc = MulticlassPrecisionRecallCurve(num_classes=num_classes, device=self.device)
            prc.update(input=sim, target=class_idx)
            precision, recall, thresholds = prc.compute()
            # F1Score
            f1s = MulticlassF1Score(average=None, num_classes=num_classes, device=self.device)
            f1s.update(input=sim, target=class_idx)
            f1 = f1s.compute()
            # Accuracy
            accm = MulticlassAccuracy(average=None, num_classes=num_classes, device=self.device)
            accm.update(input=sim, target=class_idx)
            accuracy = accm.compute()
            # Calculate and log metrics for each unique class
            for i, class_idx in enumerate(class_idx.unique()):
                self.log(f"{mode}/{name}_class_{class_idx}_precision", precision[i].mean(),
                         on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, prog_bar=False, batch_size=self.hparams.batch_size)
                self.log(f"{mode}/{name}_class_{class_idx}_recall", recall[i].mean(),
                         on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, prog_bar=False, batch_size=self.hparams.batch_size)
                self.log(f"{mode}/{name}_class_{class_idx}_acc", accuracy[i].mean(),
                         on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, prog_bar=False, batch_size=self.hparams.batch_size)
                self.log(f"{mode}/{name}_class_{class_idx}_f1", f1[i].mean(),
                         on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, prog_bar=False, batch_size=self.hparams.batch_size)
                
    def info_nce_loss(self, batch, mode="train"):

        data_a, modal_a, data_b, modal_b, class_idx = batch

        feats_a = [self.model({modal_a[0]: data_a_i}) for data_a_i in data_a]
        feats_a_tensor = torch.cat([list(dict_.values())[0] for dict_ in feats_a], dim=0)

        if isinstance(data_b, list):
            feats_b_list = []
            for feats_b_idx in range(len(data_b)):
                feats_b = [self.model({modal_b[feats_b_idx][idx]: data_b_i}) for idx, data_b_i in
                           enumerate(data_b[feats_b_idx])]
                feats_b_list.append(torch.cat([list(dict_.values())[0] for dict_ in feats_b], dim=0))
            # feats_b_tensor is the mean of feats_b_list between modalities
            feats_b_tensor = torch.stack(feats_b_list, dim=0).mean(dim=0)
        else:
            feats_b = [self.model({modal_b[idx]: data_b_i}) for idx, data_b_i in enumerate(data_b)]
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

        dual_nll = False
        for feats_idx, feats_tensor in enumerate(feats_tensors):
            sim = F.cosine_similarity(feats_tensor[:, None, :], feats_tensor[None, :, :], dim=-1)
            self_mask = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
            sim.masked_fill_(self_mask, -9e15)

            # If class_masking is set to True, mask out similar classes
            if self.hparams.class_masking:
                # Create a mask for the same classes, where True means the pair is of the same class
                class_mask = class_idx[:, None] == class_idx[None, :]
                class_mask = class_mask.repeat(2, 2)
                class_mask[self_mask.roll(shifts=sim.shape[0] // 2, dims=0)] = False  # Exclude diagonal elements
                # Mask out similarities of same class pairs
                sim.masked_fill_(class_mask, -9e15)

            pos_mask = self_mask.roll(shifts=sim.shape[0] // 2, dims=0)
            sim = sim / temperatures[feats_idx]
            nll = -sim[pos_mask] + torch.logsumexp(sim, dim=-1)
            nll = nll.mean()

            if not dual_nll:
                dual_nll = nll
            else:
                dual_nll += nll
                dual_nll /= 2

            self.log(f"{mode}/loss_" + contrast[feats_idx], nll, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)

            if self.hparams.class_masking:
                comb_sim = torch.cat([sim[pos_mask][:, None], sim.masked_fill(class_mask + pos_mask, -9e15)], dim=-1)

                ############################################ S. TEST #################################################
                # Split the class indices, similarity matrix, and masks into two halves
                half = sim.size(0) // 2
                sim_a, sim_b = sim[:half, half:], sim[half:, :half]

                # For the first half, calculate metrics for the image to x_modality comparisons
                unique_class, unique_indices = unique_with_index(class_idx)
                _, class_in_unique = torch.max((unique_class[None, :] == class_idx[:, None]).long(), dim=-1)

                sim_ab_unique = sim_a[:, unique_indices]
                sim_ab_unique[torch.arange(sim_ab_unique.shape[0]), class_in_unique] = torch.diag(sim_a)
                sim_ab_unique_padded = torch.zeros(sim_a.shape[0], max(unique_class) + 1, device=self.device)
                sim_ab_unique_padded.index_copy_(1, unique_class, sim_ab_unique)

                # Calculate and log metrics for the first half
                self._calculate_and_log_class_metrics(sim_ab_unique_padded, class_idx, mode, "first_half", unique=True)

                # For the second half, calculate metrics for the x_modality to image comparisons
                sim_ba_unique = sim_b[:, unique_indices]
                sim_ba_unique[torch.arange(sim_ba_unique.shape[0]), class_in_unique] = torch.diag(sim_b)
                sim_ba_unique_padded = torch.zeros(sim_b.shape[0], max(unique_class) + 1, device=self.device)
                sim_ba_unique_padded.index_copy_(1, unique_class, sim_ba_unique)

                # Calculate and log metrics for the second half
                self._calculate_and_log_class_metrics(sim_ba_unique_padded, class_idx, mode, "second_half", unique=True)
                ############################################ E. TEST #################################################
            else:
                comb_sim = torch.cat([sim[pos_mask][:, None], sim.masked_fill(pos_mask, -9e15)], dim=-1)
            sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
            acc_top1 = (sim_argsort == 0).float().mean()
            acc_top5 = (sim_argsort < 5).float().mean()
            self.log(f"{mode}/acc_mean_pos", 1 + sim_argsort.float().mean(), prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)

            self.log(f"{mode}/acc_top1", acc_top1, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)
            self.log(f"{mode}/acc_top5", acc_top5, prog_bar=True,
                     on_step=LOG_ON_STEP, on_epoch=LOG_ON_EPOCH, batch_size=self.hparams.batch_size)

        self.log(f"{mode}/loss", dual_nll, prog_bar=True,
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
    parser.add_argument("--class_masking", action="store_true", help="Mask classes with the same id within the batch "
                                                                     "to avoid selecting them as negatives")

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
                           num_workers=args.num_workers, self_contrast=args.self_contrast, class_masking=args.class_masking,
                           lora=args.lora, lora_rank=args.lora_rank, lora_checkpoint_dir=args.lora_checkpoint_dir,
                           lora_layer_idxs=lora_layer_idxs if lora_layer_idxs else None,
                           lora_modality_names=lora_modality_names if lora_modality_names else None,
                           linear_probing=args.linear_probing)

    if args.full_model_checkpointing:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,
                         "callbacks": [ModelCheckpoint(monitor="val/acc_top1_epoch", dirpath=args.full_model_checkpoint_dir,
                                                        filename="imagebind-{epoch:02d}-{val_loss:.2f}",
                                                        save_last=True, mode="max")]}
    else:
        checkpointing = {"enable_checkpointing": args.full_model_checkpointing,}

    trainer = Trainer(accelerator="gpu" if "cuda" in device_name else "cpu",
                      devices=1 if ":" not in device_name else [int(device_name.split(":")[1])], deterministic=True,
                      max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val,
                      logger=loggers if loggers else None, **checkpointing)

    trainer.fit(model, train_loader, val_loader)

