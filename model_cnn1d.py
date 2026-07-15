"""CNN1D model for IoV IDS (data iov: 31 features, 13 classes).

Thay the hoan toan Transformer bang CNN 1 chieu:
  input (B, 31) -> (B, 1, 31) -> 3 khoi Conv1d -> GAP -> FC -> logits
"""
import torch
import torch.nn as nn

NUM_GLOBAL_CLASSES = 13
INPUT_LEN = 31


class CNN1D_IDS(nn.Module):
    def __init__(self, input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES, dropout=0.15):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (B,1,31) -> (B,32,15)
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2: (B,32,15) -> (B,64,7)
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3: (B,64,7) -> (B,128,1)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (B, 31) float32
        x = x.unsqueeze(1)          # (B, 1, 31)
        x = self.features(x)        # (B, 128, 1)
        x = x.squeeze(-1)           # (B, 128)
        return self.classifier(x)


class FocalLoss(nn.Module):
    """Focal loss voi alpha = sqrt-inverse class frequency (nhu bai bao)."""

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


if __name__ == "__main__":
    m = CNN1D_IDS()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Trainable params: {n:,}")
    out = m(torch.randn(4, INPUT_LEN))
    print("Output shape:", out.shape)
