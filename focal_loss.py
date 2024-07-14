import torch


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.cross_entropy = torch.nn.CrossEntropyLoss(
            weight=self.alpha, reduction="none"
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, target):
        loss = self.cross_entropy(pred, target)
        target = torch.clamp(target, min=0)
        total_weights = torch.sum(self.alpha[target])
        loss = (
            1 - self.softmax(pred)[torch.arange(pred.size(0)), target]
        ) ** self.gamma * loss
        if self.size_average:
            return torch.sum(loss) / total_weights
        else:
            return loss.sum()
