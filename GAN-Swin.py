import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import math
import torch.nn.functional as F
import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global random seed fixed to: {seed}")


# ========================== Data-related parameters ==========================
class Config:
    input_dim = 616
    data_path = "C:\\Users\\30321\\PycharmProjects\\pythonProject1\\数据集\\Po4-new.csv"
    embed_dim = 64
    window_size = 14
    num_heads = 4
    use_adaptive_window = False


    attention_dropout = 0.1
    mlp_dropout = 0.1
    mlp_output_dropout = 0.1

    batch_size = 64
    n_fold = 5
    num_epochs = 20
    patience = 5
    lr = 0.0007
    weight_decay = 0.004
    warmup_epochs = 1
    label_smoothing_eps = 0.1

    cosine_annealing_T_max = num_epochs - warmup_epochs
    cosine_annealing_eta_min = 1e-6

    output_dir_prefix = "1d_transformer_with_window_attention_cosine_annealing_patchmerging"


# ========================== Core tool functions and modules ==========================
def window_partition_1d(x, window_size):
    B, L, C = x.shape
    assert L % window_size == 0, f"Sequence length ({L}) must be divisible by window size ({window_size})"
    num_windows = L // window_size
    x = x.view(B, num_windows, window_size, C)
    windows = x.contiguous().view(-1, window_size, C)
    return windows, num_windows


def window_reverse_1d(windows, window_size, L, num_windows):
    B = int(windows.shape[0] / num_windows)
    x = windows.view(B, num_windows, window_size, -1)
    x = x.contiguous().view(B, L, -1)
    return x


class WindowAttention1D(nn.Module):

    def __init__(self, dim, window_size, num_heads, attention_dropout):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == self.dim, "dim must be divisible by num_heads"

        self.relative_pos_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1, num_heads))
        )
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_pos_index", relative_coords)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x):
        B_win, N, C = x.shape
        assert N == self.window_size, f"Window length must be {self.window_size}, current is {N}"

        qkv = self.qkv(x).reshape(B_win, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        relative_pos_bias = self.relative_pos_bias_table[self.relative_pos_index.view(-1)]
        relative_pos_bias = relative_pos_bias.view(self.window_size, self.window_size, self.num_heads)
        relative_pos_bias = relative_pos_bias.permute(2, 0, 1).contiguous().view(1, self.num_heads, N, N)
        attn = attn + relative_pos_bias

        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_win, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class SwinTransformerBlock1D(nn.Module):

    def __init__(self, dim, window_size, num_heads, shift_size=0,
                 attention_dropout=0.2, mlp_dropout=0.2, mlp_output_dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in the range [0, window_size)"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(
            dim, window_size, num_heads, attention_dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(mlp_output_dropout)
        )

    def forward(self, x):
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
            mask = torch.zeros((B, L), device=x.device, dtype=torch.bool)
            mask[:, :self.shift_size] = True
            mask[:, -self.shift_size:] = True
        else:
            mask = None

        windows, num_windows = window_partition_1d(x, self.window_size)
        attn_windows = self.attn(windows)
        if mask is not None:
            mask_windows = mask.view(B, num_windows, self.window_size).contiguous().view(-1, self.window_size)
            attn_windows = attn_windows.masked_fill(mask_windows.unsqueeze(-1), 0.0)
        x = window_reverse_1d(attn_windows, self.window_size, L, num_windows)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


# ========================== 4.  PatchMerging1D  ==========================
class PatchMerging1D(nn.Module):
    def __init__(self, in_channels, out_channels=None, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.conv = nn.Conv1d(
            in_channels=in_channels * patch_size,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.norm = nn.LayerNorm(self.out_channels)

    def forward(self, x):
        B, L, C = x.shape
        assert L % self.patch_size == 0, f"Sequence length {L} must be divisible by PatchSize {self.patch_size}"
        x = x.view(B, L // self.patch_size, self.patch_size, C)
        x = x.permute(0, 1, 3, 2).contiguous()  #
        x = x.view(B, L // self.patch_size, C * self.patch_size)

        x = x.permute(0, 2, 1).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)

        return x


# ========================== Model ==========================
class TransformerWithWindowAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        self.window_size = config.window_size

        assert self.input_dim % self.window_size == 0, \
            f"Initial input dimension ({self.input_dim}) must be divisible by window size ({self.window_size})"

        print(f"Model input: {self.input_dim} dimensional features → Each feature as 1 Token → Dimension after embedding: {self.embed_dim}")
        print(f"Window size: {self.window_size} (fixed setting, no adjustment)")
        print(f"Using PatchMerging1D: Sequence length halved + information fusion (no additional Padding required)")

        self.feature_embedding = nn.Sequential(
            nn.Linear(1, self.embed_dim),
            nn.GELU()
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.input_dim, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # stage 1
        self.stage1 = nn.Sequential(
            SwinTransformerBlock1D(
                dim=self.embed_dim,
                window_size=self.window_size,
                num_heads=self.num_heads,
                shift_size=0,
                attention_dropout=config.attention_dropout,
                mlp_dropout=config.mlp_dropout,
                mlp_output_dropout=config.mlp_output_dropout
            ),
            SwinTransformerBlock1D(
                dim=self.embed_dim,
                window_size=self.window_size,
                num_heads=self.num_heads,
                shift_size=self.window_size // 2,
                attention_dropout=config.attention_dropout,
                mlp_dropout=config.mlp_dropout,
                mlp_output_dropout=config.mlp_output_dropout
            ),
            # SwinTransformerBlock1D(
            #     dim=self.embed_dim,
            #     window_size=self.window_size,
            #     num_heads=self.num_heads,
            #     shift_size=0,
            #     attention_dropout=config.attention_dropout,
            #     mlp_dropout=config.mlp_dropout,
            #     mlp_output_dropout=config.mlp_output_dropout
            # ),
            # SwinTransformerBlock1D(
            #     dim=self.embed_dim,
            #     window_size=self.window_size,
            #     num_heads=self.num_heads,
            #     shift_size=self.window_size // 2,
            #     attention_dropout=config.attention_dropout,
            #     mlp_dropout=config.mlp_dropout,
            #     mlp_output_dropout=config.mlp_output_dropout
            # )
        )


        self.patch_merge1 = PatchMerging1D(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim
        )
        #stage 2
        self.stage2 = nn.Sequential(
            SwinTransformerBlock1D(
                dim=self.embed_dim,
                window_size=self.window_size,
                num_heads=self.num_heads,
                shift_size=0,
                attention_dropout=config.attention_dropout,
                mlp_dropout=config.mlp_dropout,
                mlp_output_dropout=config.mlp_output_dropout
            ),
            SwinTransformerBlock1D(
                dim=self.embed_dim,
                window_size=self.window_size,
                num_heads=self.num_heads,
                shift_size=self.window_size // 2,
                attention_dropout=config.attention_dropout,
                mlp_dropout=config.mlp_dropout,
                mlp_output_dropout=config.mlp_output_dropout
            ),
            # SwinTransformerBlock1D(
            #     dim=self.embed_dim,
            #     window_size=self.window_size,
            #     num_heads=self.num_heads,
            #     shift_size=0,
            #     attention_dropout=config.attention_dropout,
            #     mlp_dropout=config.mlp_dropout,
            #     mlp_output_dropout=config.mlp_output_dropout
            # ),
            # SwinTransformerBlock1D(
            #     dim=self.embed_dim,
            #     window_size=self.window_size,
            #     num_heads=self.num_heads,
            #     shift_size=self.window_size // 2,
            #     attention_dropout=config.attention_dropout,
            #     mlp_dropout=config.mlp_dropout,
            #     mlp_output_dropout=config.mlp_output_dropout
            # )
        )

        self.patch_merge2 = PatchMerging1D(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim
        )
        # stage 3
        self.stage3 = nn.Sequential(
            SwinTransformerBlock1D(
                dim=self.embed_dim,
                window_size=self.window_size,
                num_heads=self.num_heads,
                shift_size=0,
                attention_dropout=config.attention_dropout,
                mlp_dropout=config.mlp_dropout,
                mlp_output_dropout=config.mlp_output_dropout
            ),
            SwinTransformerBlock1D(
                dim=self.embed_dim,
                window_size=self.window_size,
                num_heads=self.num_heads,
                shift_size=self.window_size // 2,
                attention_dropout=config.attention_dropout,
                mlp_dropout=config.mlp_dropout,
                mlp_output_dropout=config.mlp_output_dropout
            )
        )


        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


    def forward(self, x):
        B = x.shape[0]
        x = x.squeeze(1).unsqueeze(-1)
        x = self.feature_embedding(x)
        x = x + self.pos_embed

        # ----------------------PatchMerging----------------------
        x = self.stage1(x)
        x = self.patch_merge1(x)
        x = self.stage2(x)
        x = self.patch_merge2(x)
        assert x.shape[1] % self.window_size == 0, \
            f"Stage3 input length ({x.shape[1]}) must be divisible by window size ({self.window_size})"
        x = self.stage3(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)

        return logits

    def forward(self, x):
        B = x.shape[0]

        x = x.squeeze(1).unsqueeze(-1)
        x = self.feature_embedding(x)
        x = x + self.pos_embed

    # Stage1 → PatchMerge1
        x = self.stage1(x)
        x = self.patch_merge1(x)

    # Stage2 → PatchMerge2
        x = self.stage2(x)
        x = self.patch_merge2(x)

    # ---------------------- Dynamic Padding ----------------------
        L = x.shape[1]
        remainder = L % self.window_size
        if remainder != 0:
            pad_total = self.window_size - remainder
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left

            x = x.permute(0, 2, 1)
            x = nn.ConstantPad1d((pad_left, pad_right), 0.0)(x)
            x = x.permute(0, 2, 1)

            if self.training and not hasattr(self, "_patch_padding_logged"):
                print(f"Padding after Patch: Sequence length {L}→{x.shape[1]} (total padding {pad_total}, left {pad_left}, right {pad_right})")
                setattr(self, "_patch_padding_logged", True)


        assert x.shape[1] % self.window_size == 0, \
            f"Stage3 input length ({x.shape[1]}) must be divisible by window size ({self.window_size})"
        x = self.stage3(x)

        x = x.mean(dim=1)
        logits = self.classifier(x)

        return logits

def plot_training_curves(fold_history, output_dir):
    plt.figure(figsize=(12, 6))
    for fold, history in enumerate(fold_history, 1):
        plt.plot(history['train_loss'], label=f'Fold {fold} Train', alpha=0.7)
        plt.plot(history['val_loss'], label=f'Fold {fold} Val', alpha=0.7)
    plt.title('Training & Validation Loss Across Folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()


def plot_metrics(result_df, output_dir):
    metrics = ['accuracy', 'auc', 'f1', 'mcc', 'sensitivity', 'specificity']
    plt.figure(figsize=(15, 15))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 2, i)
        plt.boxplot(result_df[metric])
        plt.title(f'{metric.upper()} Distribution')
        plt.ylabel('Score')
        plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_boxplot.png'))
    plt.close()


def augment_1d_sequence(x, max_crop_ratio=0, noise_std=0):
    x = x.clone()
    seq_len = x.shape[1]

    crop_len = int(seq_len * np.random.uniform(0, max_crop_ratio))
    if crop_len > 0:
        if np.random.random() < 0.5:
            x = F.pad(x[:, crop_len:], (0, crop_len))
        else:
            x = F.pad(x[:, :-crop_len], (crop_len, 0))

    noise = torch.normal(0, noise_std, size=x.shape, device=x.device)
    x = x + noise

    return x


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, is_train=True):
        self.dataset = dataset
        self.is_train = is_train

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.is_train:
            x = augment_1d_sequence(x)
        return x, y

    def __len__(self):
        return len(self.dataset)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        num_classes = logits.shape[-1]
        log_probs = F.log_softmax(logits, dim=-1)

        smooth_targets = torch.full_like(log_probs, self.eps / (num_classes - 1))
        targets = targets.long().unsqueeze(1)
        smooth_targets.scatter_(1, targets, 1 - self.eps)

        loss = -log_probs * smooth_targets
        return loss.sum(dim=-1).mean()


# ========================== 9. Training and Evaluation Main Process ==========================
def train_and_evaluate(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        df = pd.read_csv(config.data_path)
        print(f"\nLoaded data: {len(df)} samples")
        features = df.iloc[:, :-1].values.astype(np.float32)
        labels = df.iloc[:, -1].values
        print(f"Input feature dimension: {features.shape[1]} (must be consistent with config.input_dim={config.input_dim})")

        le = LabelEncoder()
        labels = le.fit_transform(labels)
        print(f"Class distribution: {dict(zip(le.classes_, np.bincount(labels)))}")

        features = features.reshape(-1, 1, config.input_dim)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(config.data_path),
            "results",
            f"{config.output_dir_prefix}_run_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

    except Exception as e:
        print(f"\nData loading failed: {str(e)}")
        return

    skf = StratifiedKFold(n_splits=config.n_fold, shuffle=True, random_state=42)
    results = []
    all_history = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
        print(f"\n=== Fold {fold + 1}/{config.n_fold} ===")
        fold_dir = os.path.join(output_dir, f'fold_{fold + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        train_set = TensorDataset(torch.tensor(features[train_idx]), torch.tensor(labels[train_idx]))
        train_set_aug = AugmentedDataset(train_set, is_train=True)
        train_loader = DataLoader(train_set_aug, batch_size=config.batch_size, shuffle=True, pin_memory=True)

        test_set = TensorDataset(torch.tensor(features[test_idx]), torch.tensor(labels[test_idx]))
        test_loader = DataLoader(test_set, batch_size=config.batch_size, pin_memory=True)

        model = TransformerWithWindowAttention(config).to(device)
        print(f"Total trainable parameters of the model: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3:.1f}k")

        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        criterion = LabelSmoothingCrossEntropy(eps=config.label_smoothing_eps)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.cosine_annealing_T_max,
            eta_min=config.cosine_annealing_eta_min
        )

        best_val_loss = float('inf')
        patience_counter = 0
        fold_history = {'train_loss': [], 'val_loss': []}

        base_lr = config.lr
        current_lr = base_lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        for epoch in range(config.num_epochs):
            if epoch < config.warmup_epochs:
                current_lr = base_lr * (epoch + 1) / config.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            else:
                if epoch == config.warmup_epochs:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = base_lr

            model.train()
            train_loss = 0.0
            for inputs, labels_batch in train_loader:
                inputs, labels_batch = inputs.to(device), labels_batch.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            avg_train_loss = train_loss / len(train_loader.dataset)
            fold_history['train_loss'].append(avg_train_loss)

            model.eval()
            val_loss = 0.0
            all_preds = []
            all_probas = []
            all_labels = []
            with torch.no_grad():
                for inputs, labels_batch in test_loader:
                    inputs, labels_batch = inputs.to(device), labels_batch.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item() * inputs.size(0)

                    probas = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    all_probas.extend(probas)
                    all_preds.extend(preds)
                    all_labels.extend(labels_batch.cpu().numpy())

            avg_val_loss = val_loss / len(test_loader.dataset)
            fold_history['val_loss'].append(avg_val_loss)

            if epoch >= config.warmup_epochs:
                scheduler.step()

            print(f"Epoch {epoch + 1:02d} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(fold_dir, 'best_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping triggered: Validation loss has not decreased for {config.patience} consecutive epochs")
                    break

        model.load_state_dict(torch.load(os.path.join(fold_dir, 'best_model.pth')))
        model.eval()
        all_preds = []
        all_probas = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels_batch in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probas = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_probas.extend(probas)
                all_preds.extend(preds)
                all_labels.extend(labels_batch.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        metrics = {
            'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)),
            'auc': roc_auc_score(all_labels, all_probas) if len(np.unique(all_labels)) > 1 else 0.0,
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'mcc': matthews_corrcoef(all_labels, all_preds),
            'sensitivity': TP / (TP + FN) if (TP + FN) > 0 else 0.0,
            'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0.0
        }
        results.append(metrics)
        all_history.append(fold_history)

        print(f"\nFold {fold + 1} evaluation results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(output_dir, 'cross_validation_results.csv'), index=False)
    print(f"\n=== Final results of 5-fold cross-validation ===")
    print(result_df.describe().loc[['mean', 'std']].round(4))

    plot_training_curves(all_history, output_dir)
    plot_metrics(result_df, output_dir)
    print(f"\nAll results have been saved to: {output_dir}")


# ========================== program entry==========================
if __name__ == "__main__":
    set_seed(seed=42)
    config = Config()
    train_and_evaluate(config)