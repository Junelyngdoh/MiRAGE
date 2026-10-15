import os
import random
import csv
import numpy as np
from PIL import Image, ImageFile
import logging
import argparse
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# Robust loading logic used in UCMerced configurations
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import RandAugment

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import CLIPModel, CLIPVisionConfig, CLIPProcessor
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------
# Determinism
# -------------------------
def provide_determinism(seed: Optional[int] = 23):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
        logging.info(f"No seed provided. Generated random seed: {seed}")
    else:
        logging.info(f"Using provided seed for reproducibility: {seed}")
    
    # 1. Enforce strict deterministic algorithms for CUDA
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
           
    # Force deterministic PyTorch operations. Crash if a deterministic implementation is missing.
    torch.use_deterministic_algorithms(True, warn_only=False)

def worker_seed_set(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_rng_state(train_loader):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader_gen": train_loader.generator.get_state(),
    }

def set_rng_state(state, train_loader):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
    train_loader.generator.set_state(state["loader_gen"].cpu())

# -------------------------
# Transforms
# -------------------------
clip_mean =[0.48145466, 0.4578275, 0.40821073]
clip_std  =[0.26862954, 0.26130258, 0.27577711]

train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((336, 336)),
    transforms.RandomHorizontalFlip(p=0.5),
    RandAugment(num_ops=2, magnitude=4),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=clip_mean, std=clip_std)
])

test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((336, 336)),
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std)
])

# -------------------------
# Dataset wrapper for split
# -------------------------
class CustomSubsetDataset(Dataset):
    def __init__(self, samples, transform=None, classes=None, safe_load: bool = False, robust_retry: bool = False):
        self.samples = samples
        self.transform = transform
        self.classes = classes
        self.safe_load = safe_load
        self.robust_retry = robust_retry

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.robust_retry:
            last_err = None
            for attempt in range(10):
                try:
                    with Image.open(path) as im:
                        img = im.convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    return img, label
                except Exception as e:
                    last_err = e
                    time.sleep(0.1) # Slight delay before retrying
                    continue
            # Return dummy instead of skipping to next idx to preserve batch determinism
            logging.warning(f"Failed to load {path} after 10 attempts. Using dummy image. Error: {last_err}")
            img = Image.new("RGB", (336, 336), (128, 128, 128))
            if self.transform:
                img = self.transform(img)
            return img, label
        else:
            if self.safe_load:
                try:
                    image = Image.open(path).convert('RGB')
                except Exception as e:
                    logging.warning(f"Failed to open image: {path} ({e}). Using dummy image.")
                    image = Image.new("RGB", (336, 336), (128, 128, 128))
            else:
                image = Image.open(path).convert('RGB')

            if self.transform:
                image = self.transform(image)
            return image, label

# -------------------------
# CutMix helpers
# -------------------------
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    return x, y_a, y_b, lam

# -------------------------
# MoR modules
# -------------------------
@dataclass
class MoRConfig:
    t_max: int = 4
    router_hidden: int = 256
    mask_sharpness: float = 10.0
    t_min: float = 1.0
    compute_reg_weight: float = 0.0

class SharedResidualMLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class NeuralRouter(nn.Module):
    def __init__(self, dim: int, cfg: MoRConfig):
        super().__init__()
        self.cfg = cfg
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, cfg.router_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.router_hidden),
            nn.Linear(cfg.router_hidden, 1),
        )

    def forward(self, cls_feat: torch.Tensor, patch_feat: torch.Tensor) -> torch.Tensor:
        patch_pool = patch_feat.mean(dim=1) 
        x = torch.cat([patch_pool, cls_feat], dim=1) 
        s = torch.sigmoid(self.mlp(x)).squeeze(-1) 
        t_soft = self.cfg.t_min + (self.cfg.t_max - self.cfg.t_min) * s
        return t_soft

    def step_mask(self, t_soft: torch.Tensor, t_idx: int, hard: bool = False) -> torch.Tensor:
        if hard:
            t_hard = torch.round(t_soft).clamp(self.cfg.t_min, self.cfg.t_max)
            return (t_idx <= t_hard).float()
        sharp = self.cfg.mask_sharpness
        return torch.sigmoid(sharp * (t_soft - float(t_idx)))

class RecursiveGlobalHead(nn.Module):
    def __init__(self, num_classes: int, dim: int, cfg: MoRConfig):
        super().__init__()
        self.cfg = cfg
        self.update_block = SharedResidualMLP(dim=dim, hidden=512, dropout=0.1)
        self.visual_proj = nn.Linear(dim, 768, bias=False)
        
        self.classifier = nn.Linear(768, num_classes, bias=False)        

    def forward(self, cls_feat: torch.Tensor, t_soft: torch.Tensor, router: NeuralRouter, hard_mask: bool = False):
        h = cls_feat
        # 1. MoR Routing Loop (Unchanged)
        for t in range(1, self.cfg.t_max + 1):
            m = router.step_mask(t_soft, t, hard=hard_mask).unsqueeze(-1)
            h_new = self.update_block(h)
            h = h + m * (h_new - h)
            
        
        # 2. Project visual features from 1024-dim to 768-dim CLIP space
        h_proj = self.visual_proj(h)
        
        # 3. L2 Normalize the visual features (Crucial for CLIP math)
        h_proj = h_proj / h_proj.norm(dim=-1, keepdim=True)
        
        # 4. L2 Normalize the semantic weights
        w = self.classifier.weight
        w = w / w.norm(dim=-1, keepdim=True)
        
        # 5. Compute Cosine Similarity and scale by temperature (100.0 is standard CLIP temp)
        logits = 100.0 * F.linear(h_proj, w)
        
        return logits, h

class RecursiveSpatialHead(nn.Module):
    def __init__(self, num_classes: int, dim: int, cfg: MoRConfig, num_heads: int = 4, qk_dim: int = 128):
        super().__init__()
        self.cfg = cfg
        self.num_heads = num_heads
        self.qk_dim = qk_dim
        self.scale = (qk_dim ** -0.5)

        self.k_proj = nn.Linear(dim, num_heads * qk_dim)
        self.v_proj = nn.Linear(dim, num_heads * qk_dim)
        self.q_proj = nn.Linear(dim, num_heads * qk_dim)
        self.ctx_proj = nn.Linear(num_heads * qk_dim, dim)

        self.update_block = SharedResidualMLP(dim=dim, hidden=512, dropout=0.1)
        self.visual_proj = nn.Linear(dim, 768, bias=False)
        
        self.classifier = nn.Linear(768, num_classes, bias=False)

    def forward(self, patch_feat: torch.Tensor, t_soft: torch.Tensor, router: NeuralRouter, hard_mask: bool = False):
        B, N, D = patch_feat.shape
        K = self.k_proj(patch_feat).reshape(B, N, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)
        V = self.v_proj(patch_feat).reshape(B, N, self.num_heads, self.qk_dim).permute(0, 2, 1, 3)

        h = patch_feat.mean(dim=1) 
        # 1. MoR Routing Loop (Unchanged)
        for t in range(1, self.cfg.t_max + 1):
            m = router.step_mask(t_soft, t, hard=hard_mask).unsqueeze(-1)
            Q = self.q_proj(h).reshape(B, self.num_heads, 1, self.qk_dim)
            attn = (Q @ K.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            ctx = (attn @ V).reshape(B, self.num_heads * self.qk_dim)

            c = self.ctx_proj(ctx)
            h_new = self.update_block(h + c)
            h = h + m * (h_new - h)

        # ?? HERE IS WHERE visual_proj IS CALLED ??
        # 2. Project visual features from 1024-dim to 768-dim CLIP space
        h_proj = self.visual_proj(h)
        
        # 3. L2 Normalize the visual features (Crucial for CLIP math)
        h_proj = h_proj / h_proj.norm(dim=-1, keepdim=True)
        
        # 4. L2 Normalize the semantic weights
        w = self.classifier.weight
        w = w / w.norm(dim=-1, keepdim=True)
        
        # 5. Compute Cosine Similarity and scale by temperature
        logits = 100.0 * F.linear(h_proj, w)
        
        return logits, h


# -------------------------
# Loss, Eval & Parameters
# -------------------------
def inter_classifier_consistency_loss(l1, l2):
    p1, p2 = F.softmax(l1, dim=1), F.softmax(l2, dim=1)
    lp1, lp2 = F.log_softmax(l1, dim=1), F.log_softmax(l2, dim=1)
    return 0.5 * (
        F.kl_div(lp2, p1, reduction="batchmean", log_target=False)
        + F.kl_div(lp1, p2, reduction="batchmean", log_target=False)
    )

def evaluate_model(clip_model, router, C1, C2, loader, name, device, hard_routing: bool = True, use_amp: bool = False):
    clip_model.eval(); router.eval(); C1.eval(); C2.eval()
    c1_correct, c2_correct, avg_correct, total = 0, 0, 0, 0
    mse1_sum, mse2_sum, mse_avg_sum = 0.0, 0.0, 0.0
    running_t_soft = 0.0
    nb = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # Use deterministic AMP flag
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                vo = clip_model.vision_model(pixel_values=imgs)
                cls_feat = vo.pooler_output
                patch_feat = vo.last_hidden_state[:, 1:, :]
                t_soft = router(cls_feat, patch_feat)
                l1, _ = C1(cls_feat, t_soft, router, hard_mask=hard_routing)
                l2, _ = C2(patch_feat, t_soft, router, hard_mask=hard_routing)
                avg_l = (l1 + l2) / 2

            running_t_soft += float(t_soft.mean().item())
            nb += 1

            # Accuracy
            c1_correct += (l1.argmax(dim=1) == labels).sum().item()
            c2_correct += (l2.argmax(dim=1) == labels).sum().item()
            avg_correct += (avg_l.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

            # MSE Calculation (Brier Score approach)
            num_classes = l1.size(1)
            labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()
            
            p1 = F.softmax(l1, dim=1)
            p2 = F.softmax(l2, dim=1)
            p_avg = F.softmax(avg_l, dim=1)
            
            mse1_sum += F.mse_loss(p1, labels_one_hot, reduction='sum').item()
            mse2_sum += F.mse_loss(p2, labels_one_hot, reduction='sum').item()
            mse_avg_sum += F.mse_loss(p_avg, labels_one_hot, reduction='sum').item()

    acc1 = 100.0 * c1_correct / total if total > 0 else 0.0
    acc2 = 100.0 * c2_correct / total if total > 0 else 0.0
    acc_avg = 100.0 * avg_correct / total if total > 0 else 0.0
    avg_t_soft = (running_t_soft / nb) if nb > 0 else 0.0

    # Calculate average MSE
    total_elements = total * l1.size(1) if total > 0 else 1
    mse1 = mse1_sum / total_elements
    mse2 = mse2_sum / total_elements
    mse_avg = mse_avg_sum / total_elements

    logging.info(f"Eval - {name} - Final Acc (Ensemble): {acc_avg:.2f}%, C1 Acc: {acc1:.2f}%, C2 Acc: {acc2:.2f}%, Avg T_soft: {avg_t_soft:.3f} | C1 MSE: {mse1:.4f}, C2 MSE: {mse2:.4f}")

    #logging.info(f"Eval - {name} - C1 Acc: {acc1:.2f}%, C2 Acc: {acc2:.2f}%, Avg T_soft: {avg_t_soft:.3f} | C1 MSE: {mse1:.4f}, C2 MSE: {mse2:.4f}")
    return acc1, acc2, acc_avg, avg_t_soft, mse1, mse2, mse_avg

def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())

# -------------------------
# Scheduler & Checkpoints
# -------------------------
class CustomLRScheduler:
    def __init__(self, optimizer, warmup_iterations, step_epochs, decay_factor, iters_per_epoch):
        self.optimizer = optimizer
        self.warmup_iterations = warmup_iterations
        self.step_epochs = step_epochs
        self.decay_factor = decay_factor
        self.current_iter = 0
        self.iters_per_epoch = iters_per_epoch
        self.initial_lrs =[group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        for i, group in enumerate(self.optimizer.param_groups):
            initial_lr = self.initial_lrs[i]
            if self.current_iter < self.warmup_iterations:
                lr = initial_lr * (self.current_iter + 1) / self.warmup_iterations
            else:
                epoch_offset_iters = self.current_iter - self.warmup_iterations
                current_epoch_in_decay_phase = epoch_offset_iters // self.iters_per_epoch if self.iters_per_epoch > 0 else 0
                num_decays = current_epoch_in_decay_phase // self.step_epochs
                lr = initial_lr * (self.decay_factor ** num_decays)
            group['lr'] = lr
        self.current_iter += 1

def save_checkpoint(epoch, peft_model, router, c1, c2, optimizer, scheduler, filepath, scaler=None, train_loader=None):
    state = {
        'epoch': epoch,
        'peft_model_state_dict': get_peft_model_state_dict(peft_model),
        'router_state_dict': router.state_dict(),
        'c1_state_dict': c1.state_dict(),
        'c2_state_dict': c2.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_current_iter': scheduler.current_iter,
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'rng_state': get_rng_state(train_loader) if train_loader is not None else None,
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath, peft_model, router, c1, c2, optimizer=None, scheduler=None, device='cpu'):
    if not filepath or (not os.path.exists(filepath)):
        logging.warning(f"Checkpoint file not found: {filepath}. Starting from scratch.")
        return 0, None, None

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filepath, map_location=device)

    try:
        set_peft_model_state_dict(peft_model, checkpoint['peft_model_state_dict'])
    except Exception:
        peft_model.load_state_dict(checkpoint['peft_model_state_dict'], strict=False)

    router.load_state_dict(checkpoint['router_state_dict'])
    c1.load_state_dict(checkpoint['c1_state_dict'])
    c2.load_state_dict(checkpoint['c2_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_current_iter' in checkpoint:
        scheduler.current_iter = checkpoint['scheduler_current_iter']

    logging.info(f"Loaded checkpoint from {filepath}. Resuming from epoch {start_epoch}.")
    return start_epoch, checkpoint.get('scaler_state_dict'), checkpoint.get('rng_state')


# -------------------------
# Training Loop 
# -------------------------
def train_model(
    clip_model, router, C1, C2, train_loader, criterion, optimizer, scheduler, num_epochs,
    device, checkpoint_dir, save_every_n_epochs, test_loader_for_logging=None, start_epoch=0,
    csv_writer=None, csv_file_handle=None, loss_weight_c1=0.5, loss_weight_c2=0.5,
    consistency_loss_weight=0.1, use_cutmix=True, cutmix_prob=0.5, cutmix_alpha=1.0,
    compute_reg_weight: float = 0.0, hard_routing_eval: bool = True, use_amp: bool = False, 
    eval_every_n_epochs: int = 1, scaler_state=None
):
    scaler = torch.amp.GradScaler(enabled=use_amp)
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    for epoch in range(start_epoch, num_epochs):
        clip_model.train(); router.train(); C1.train(); C2.train()

        running_total, running_c1, running_c2, running_cons, running_compute, running_tsoft = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        processed_batches = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            apply_cutmix = use_cutmix and (np.random.rand() < cutmix_prob)
            if apply_cutmix:
                imgs, labels_a, labels_b, lam = cutmix_data(imgs, labels, alpha=cutmix_alpha)

            # Use deterministic AMP flag
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                vo = clip_model.vision_model(pixel_values=imgs)
                cls_feat = vo.pooler_output
                patch_feat = vo.last_hidden_state[:, 1:, :]

                t_soft = router(cls_feat, patch_feat)
                running_tsoft += float(t_soft.mean().detach().item())

                logits_c1, _ = C1(cls_feat, t_soft, router, hard_mask=False)
                logits_c2, _ = C2(patch_feat, t_soft, router, hard_mask=False)

                if apply_cutmix:
                    loss_c1 = criterion(logits_c1, labels_a) * lam + criterion(logits_c1, labels_b) * (1.0 - lam)
                    loss_c2 = criterion(logits_c2, labels_a) * lam + criterion(logits_c2, labels_b) * (1.0 - lam)
                else:
                    loss_c1 = criterion(logits_c1, labels)
                    loss_c2 = criterion(logits_c2, labels)

                loss_cons = inter_classifier_consistency_loss(logits_c1, logits_c2) if consistency_loss_weight > 0 else 0.0
                loss_compute = t_soft.mean() if compute_reg_weight > 0 else 0.0

                total_loss = (loss_weight_c1 * loss_c1) + (loss_weight_c2 * loss_c2) + \
                             (consistency_loss_weight * loss_cons) + (compute_reg_weight * loss_compute)

            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(clip_model.parameters()) +
                list(router.parameters()) +
                list(C1.parameters()) +
                list(C2.parameters()),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_total += float(total_loss.item())
            running_c1 += float(loss_c1.item())
            running_c2 += float(loss_c2.item())
            if consistency_loss_weight > 0:
                running_cons += float(loss_cons) if isinstance(loss_cons, float) else float(loss_cons.item())
            if compute_reg_weight > 0:
                running_compute += float(loss_compute) if isinstance(loss_compute, float) else float(loss_compute.item())

            processed_batches += 1

        div = max(processed_batches, 1)
        avg_total = running_total / div
        avg_c1 = running_c1 / div
        avg_c2 = running_c2 / div
        avg_cons = running_cons / div
        avg_compute = running_compute / div
        avg_tsoft = running_tsoft / div

        logging.info(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_total:.4f}, "
            f"C1: {avg_c1:.4f}, C2: {avg_c2:.4f}, Cons: {avg_cons:.4f}, "
            f"AvgT: {avg_compute:.4f}, AvgTsoft: {avg_tsoft:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        acc1, acc2, acc_avg, val_t, mse1, mse2, mse_avg = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        is_eval_epoch = ((epoch + 1) % eval_every_n_epochs == 0) or ((epoch + 1) == num_epochs)
        
        if test_loader_for_logging is not None and is_eval_epoch:
            acc1, acc2, acc_avg, val_t, mse1, mse2, mse_avg = evaluate_model(
                clip_model, router, C1, C2, test_loader_for_logging, "Epoch Log", device, 
                hard_routing=hard_routing_eval, use_amp=use_amp
            )
            
        if csv_writer is not None:
            csv_writer.writerow([
                epoch + 1, f"{avg_total:.4f}", f"{avg_c1:.4f}", f"{avg_c2:.4f}", f"{avg_cons:.4f}",
                f"{avg_compute:.4f}", f"{avg_tsoft:.4f}", f"{acc1:.2f}", f"{acc2:.2f}", f"{acc_avg:.2f}", f"{val_t:.4f}",
                f"{mse1:.4f}", f"{mse2:.4f}", f"{mse_avg:.4f}"
            ])
            if csv_file_handle is not None:
                csv_file_handle.flush()

        if checkpoint_dir and save_every_n_epochs > 0 and ((epoch + 1) % save_every_n_epochs == 0):
            save_checkpoint(
                epoch + 1, clip_model, router, C1, C2, optimizer, scheduler,
                os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"),
                scaler=scaler, train_loader=train_loader
            )

    if checkpoint_dir:
        save_checkpoint(
            num_epochs, clip_model, router, C1, C2, optimizer, scheduler,
            os.path.join(checkpoint_dir, f"final_model_epoch_{num_epochs}.pth"),
            scaler=scaler, train_loader=train_loader
        )
        
        
# -------------------------
# Semantic Imprinting
# -------------------------
def get_semantic_weights(clip_model: CLIPModel, class_names: list, proj_dim: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
    logging.info(f"Computing dual semantic imprinting for {len(class_names)} classes...")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    # Prompt 1: Global context for C1
    prompts_c1 =[f"a wide-angle aerial or satellite image showing the overall scene, global layout, and geographic context of a {name.replace('_', ' ')}" for name in class_names]
    #prompts_c1 =[f"a satellite image of an entire {name.replace('_', ' ')}, showing the overall scene and layout" for name in class_names]
    
    # Prompt 2: Local/Spatial context for C2
    prompts_c2 =[ f"a zoomed-in aerial or satellite view focusing on the local textures, structural patterns, and individual components typical of a {name.replace('_', ' ')}" for name in class_names ]
    #prompts_c2 =[f"high-resolution satellite details showing individual objects, textures, and local structures of a {name.replace('_', ' ')}" for name in class_names]


    inputs_c1 = processor(text=prompts_c1, return_tensors="pt", padding=True)
    inputs_c2 = processor(text=prompts_c2, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features_c1 = clip_model.get_text_features(**inputs_c1).cpu()
        text_features_c2 = clip_model.get_text_features(**inputs_c2).cpu()

    # Verify dimensions
    actual_dim = text_features_c1.shape[-1]
    if actual_dim != proj_dim:
        raise ValueError(
            f"CLIP text projection output dim {actual_dim} != expected proj_dim {proj_dim}."
        )

    # Normalize both sets of features
    text_features_c1 = text_features_c1 / text_features_c1.norm(dim=-1, keepdim=True)
    text_features_c2 = text_features_c2 / text_features_c2.norm(dim=-1, keepdim=True)

    logging.info(f"Dual semantic weights ready. C1 shape: {list(text_features_c1.shape)}, C2 shape: {list(text_features_c2.shape)}")
    
    return text_features_c1.float(), text_features_c2.float()


def run_experiment(base_args, current_seed):
    import gc
    import copy
    
    # Create a fresh copy of args for this specific seed run
    args = copy.deepcopy(base_args)
    args.seed = current_seed

    # --- Apply Preset Overrides (Dataset Paths & Quirks) ---
    if args.preset == 'nwpu':
        args.data_root = args.data_root or "./data/NWPU-RESISC45"
        args.checkpoint_dir = f"./checkpoints_nwpu_mor_highacc_seed_{current_seed}"
        csv_filename = f"training_log_nwpu_mor_seed_{current_seed}.csv"

    elif args.preset == 'aid':
        args.data_root = args.data_root or "./data/AID"
        args.checkpoint_dir = f"./checkpoints_aid_mor_highacc_seed_{current_seed}"
        csv_filename = f"training_log_aid_mor_seed_{current_seed}.csv"

    elif args.preset == 'ucmerced':
        args.data_root = args.data_root or "./data/UCMerced_LandUse/Images"
        args.checkpoint_dir = f"./checkpoints_ucmerced_mor_highacc_seed_{current_seed}"
        args.robust_retry = True
        csv_filename = f"training_log_ucmerced_mor_seed_{current_seed}.csv"
    else:
        args.checkpoint_dir = f"{args.checkpoint_dir}_seed_{current_seed}"
        csv_filename = f"training_log_unified_seed_{current_seed}.csv"

    if not args.data_root:
        raise ValueError("Please specify --data_root or utilize a --preset that defines it.")

    # ==========================================
    logging.info("-" * 50)
    logging.info("EXPERIMENT PARAMETERS & ARGUMENTS:")
    for key, value in sorted(vars(args).items()):
        logging.info(f"  {key:25s}: {value}")
    logging.info("-" * 50)
    # ==========================================

    # APPLY STRICT DETERMINISM FIRST
    provide_determinism(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not os.path.isdir(args.data_root):
        raise FileNotFoundError(f"Data directory not found: {args.data_root}. Exiting.")

    logging.info(f"Loading full dataset from {args.data_root}...")
    full_ds = datasets.ImageFolder(args.data_root, transform=None)
    if len(full_ds) == 0:
        raise RuntimeError("Dataset is empty. Please check directory structure (ImageFolder).")

    y = np.array(full_ds.targets)
    
    # ------------------- Stratified Split Resolution -------------------
    if args.train_ratio is not None:
        split_test_size = 1.0 - float(args.train_ratio)
    else:
        split_test_size = float(args.test_size)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=split_test_size,
        stratify=y,
        random_state=args.seed,
        shuffle=True,
    )

    train_ds = CustomSubsetDataset([full_ds.samples[i] for i in train_idx], transform=train_transforms, classes=full_ds.classes, safe_load=args.safe_load, robust_retry=args.robust_retry)
    test_ds = CustomSubsetDataset([full_ds.samples[i] for i in test_idx], transform=test_transforms, classes=full_ds.classes, safe_load=args.safe_load, robust_retry=args.robust_retry)

    num_classes = len(full_ds.classes)
    logging.info(f"Total images: {len(full_ds)}. Classes: {num_classes}")
    logging.info(f"Split results -> Training: {len(train_ds)}, Testing: {len(test_ds)}")

    # ------------------- Model Setup & Semantic Imprinting -------------------
    model_name = "openai/clip-vit-large-patch14-336"
    clip_config = CLIPVisionConfig.from_pretrained(model_name)
    
    logging.info("Loading full CLIPModel on CPU for semantic imprinting...")
    clip_full = CLIPModel.from_pretrained(model_name)
    clip_full.eval()

    if args.use_semantic_imprinting:
        semantic_weights_c1, semantic_weights_c2 = get_semantic_weights(clip_full, full_ds.classes, proj_dim=768)
    else:
        logging.info("Semantic imprinting is disabled. Keeping random initialization for classifiers.")


    # ------------------- Memory Cleanup -------------------
    logging.info("Deleting CLIP text components to free memory before pushing to GPU...")
    del clip_full.text_model
    del clip_full.text_projection
    gc.collect()

    # ------------------- Backbone + LoRA -------------------
    peft_conf = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    clip_peft = get_peft_model(clip_full, peft_conf).to(device)

    # ------------------- MoR Modules -------------------
    mor_cfg = MoRConfig(
        t_max=args.t_max,
        router_hidden=args.router_hidden,
        mask_sharpness=args.mask_sharpness,
        t_min=args.t_min,
        compute_reg_weight=args.compute_reg_weight,
    )

    router = NeuralRouter(dim=clip_config.hidden_size, cfg=mor_cfg).to(device)
    C1 = RecursiveGlobalHead(num_classes=num_classes, dim=clip_config.hidden_size, cfg=mor_cfg).to(device)
    C2 = RecursiveSpatialHead(num_classes=num_classes, dim=clip_config.hidden_size, cfg=mor_cfg, num_heads=4, qk_dim=128).to(device)

    # ------------------- Inject Semantic Weights -------------------
    if args.use_semantic_imprinting:
        logging.info("Injecting specialized semantic weights into C1 (Global) and C2 (Spatial)...")
        with torch.no_grad():
            # Apply to C1
            C1.classifier.weight.copy_(semantic_weights_c1)
            if C1.classifier.bias is not None:
                C1.classifier.bias.zero_()
            
            # Apply to C2
            C2.classifier.weight.copy_(semantic_weights_c2)
            if C2.classifier.bias is not None:
                C2.classifier.bias.zero_()
    

    
    # ------------------- Setup Seeded Generators for Dataloaders -------------------
    g_train = torch.Generator()
    g_train.manual_seed(args.seed)

    g_test = torch.Generator()
    g_test.manual_seed(args.seed + 1)

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, worker_init_fn=worker_seed_set,
        generator=g_test
    ) if len(test_ds) > 0 else None

    # Evaluation Mode
    if args.eval:
        if not args.checkpoint_path:
            raise ValueError("--eval requires --checkpoint_path")
        load_checkpoint(args.checkpoint_path, clip_peft, router, C1, C2, device=device)
        evaluate_model(
            clip_peft, router, C1, C2, test_loader, "Final Evaluation", device, 
            hard_routing=args.hard_routing_eval, use_amp=args.use_amp
        )
        return

    # ------------------- Training Dataloader & Optim -------------------
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=worker_seed_set,
        generator=g_train, persistent_workers=True
    )

    optimizer = optim.AdamW([
            {"params":[p for p in clip_peft.parameters() if p.requires_grad], "lr": args.lr_peft},
            {"params": list(router.parameters()) + list(C1.parameters()) + list(C2.parameters()), "lr": args.lr_classifier},
        ],
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    iters_per_epoch = len(train_loader)
    
    scheduler = CustomLRScheduler(
        optimizer,
        warmup_iterations=5 * iters_per_epoch,
        step_epochs=15,
        decay_factor=0.3,
        iters_per_epoch=iters_per_epoch
    )

    start_epoch, scaler_state, rng_state = 0, None, None
    if args.checkpoint_path:
        logging.info("--- Resuming Training ---")
        start_epoch, scaler_state, rng_state = load_checkpoint(
            args.checkpoint_path, clip_peft, router, C1, C2, 
            optimizer=optimizer, scheduler=scheduler, device=device
        )
        if start_epoch > 0:
            if rng_state is None:
                logging.warning("Checkpoint has no RNG state: resumed run will not match an uninterrupted run.")
            else:
                logging.info(f"Replaying data pipeline for {start_epoch} finished epoch(s)...")
                for _ in range(start_epoch):
                    for _ in train_loader:      
                        pass
                set_rng_state(rng_state, train_loader)


    csv_path = os.path.join(args.checkpoint_dir, csv_filename)
    file_mode = "a" if start_epoch > 0 and os.path.exists(csv_path) else "w"

    with open(csv_path, file_mode, newline="") as csv_file:
        csv_writer_main = csv.writer(csv_file)
        if file_mode == "w":
            csv_writer_main.writerow([
                "epoch", "loss", "c1_loss", "c2_loss", "consist_loss",
                "avg_T", "avg_tsoft_train", "test_acc_c1", "test_acc_c2", "test_acc_avg_all", "val_t_soft",
                "test_mse_c1", "test_mse_c2", "test_mse_avg"
            ])

        logging.info("Starting training...")
        train_model(
            clip_model=clip_peft, router=router, C1=C1, C2=C2, train_loader=train_loader,
            criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=args.num_epochs,
            device=device, checkpoint_dir=args.checkpoint_dir, save_every_n_epochs=args.save_every_n_epochs,
            test_loader_for_logging=test_loader, start_epoch=start_epoch, csv_writer=csv_writer_main,
            csv_file_handle=csv_file, loss_weight_c1=args.loss_weight_c1, loss_weight_c2=args.loss_weight_c2,
            consistency_loss_weight=args.consistency_loss_weight, use_cutmix=args.use_cutmix,
            cutmix_prob=args.cutmix_prob, cutmix_alpha=args.cutmix_alpha, compute_reg_weight=args.compute_reg_weight,
            hard_routing_eval=args.hard_routing_eval, use_amp=args.use_amp, eval_every_n_epochs=args.eval_every_n_epochs,
            scaler_state=scaler_state
        )

    # Optional Parameter Count Profile
    if args.preset == 'ucmerced':
        total_params = count_parameters(clip_peft) + count_parameters(router) + count_parameters(C1) + count_parameters(C2)
        trainable_params = count_parameters(clip_peft, True) + count_parameters(router, True) + count_parameters(C1, True) + count_parameters(C2, True)
        logging.info(f"Total params (backbone+router+heads): {total_params:,}")
        logging.info(f"Trainable params (LoRA+router+heads): {trainable_params:,}")
        
    logging.info("Training finished. Checkpoints saved in checkpoint_dir.")


    # VRAM MEMORY CLEANUP FOR MULTIPLE RUNS
    logging.info("Cleaning up GPU Memory for next seed run...")
    del clip_peft
    del router
    del C1
    del C2
    del optimizer
    del scheduler
    del train_loader
    if test_loader:
        del test_loader
    gc.collect()
    #torch.cuda.empty_cache()


# -------------------------
# Main
# -------------------------
def main():
    import gc 

    parser = argparse.ArgumentParser(description="Unified script supporting NWPU, AID, UCMerced with dynamic train/test splits.")

    parser.add_argument("--preset", type=str, default=None, choices=['nwpu', 'aid', 'ucmerced'],
                        help="Select a dataset preset to auto-configure paths and specific data loading quirks.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1, help="Number of multi-seed runs to execute")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_unified")
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_peft", type=float, default=1e-5)
    parser.add_argument("--lr_classifier", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--save_every_n_epochs", type=int, default=80)
    parser.add_argument("--eval_every_n_epochs", type=int, default=80, 
                        help="Evaluate every k epochs and at the last epoch.")
    
    # SyDHA / MoR Args
    parser.add_argument("--consistency_loss_weight", type=float, default=0.1)
    parser.add_argument("--loss_weight_c1", type=float, default=0.5)
    parser.add_argument("--loss_weight_c2", type=float, default=0.5)
    parser.add_argument("--t_max", type=int, default=4)
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--router_hidden", type=int, default=256)
    parser.add_argument("--mask_sharpness", type=float, default=10.0)
    parser.add_argument("--compute_reg_weight", type=float, default=0.0)

    # Splitting Rules 
    parser.add_argument("--test_size", type=float, default=0.8, help="Test split ratio (default 0.8 for 20/80 train/test split).")
    parser.add_argument("--train_ratio", type=float, default=None, help="Train split ratio. If provided, overrides test_size.")

    # Cutmix
    parser.add_argument("--no-cutmix", dest="use_cutmix", action="store_false")
    parser.set_defaults(use_cutmix=True)
    parser.add_argument("--cutmix_prob", type=float, default=0.5)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)

    # Semantic Imprinting
    parser.add_argument("--no-semantic-imprinting", dest="use_semantic_imprinting", action="store_false", help="Disable semantic imprinting")
    parser.set_defaults(use_semantic_imprinting=True)

    # Misc Exec Params
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--hard-routing-eval", action="store_true")
    parser.set_defaults(hard_routing_eval=True)
    parser.add_argument("--safe_load", action="store_true")
    parser.add_argument("--robust_retry", action="store_true")
    
    parser.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True, 
                    help="Enable Automatic Mixed Precision. WARNING: Can break strict mathematical reproducibility.")
    args = parser.parse_args()

    for i in range(args.num_runs):
        # Starts at args.seed, then steps by 2 (e.g., if seed=0 -> 0, 2, 4...)
        base_seed = args.seed + (i * 2)
        logging.info(f"=== STARTING RUN {i+1}/{args.num_runs} with Train Seed {base_seed} & Test Seed {base_seed + 1} ===")
        
        run_experiment(args, base_seed)
        args.checkpoint_path = None
        
    logging.info("ALL SEEDS COMPLETED SUCCESSFULLY!")
    

if __name__ == "__main__":
    main()
