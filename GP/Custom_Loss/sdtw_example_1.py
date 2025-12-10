import torch
import torch.nn as nn
import torch.optim as optim
import math


def softmin(x, gamma):
    """Soft-Min 함수 (SoftDTW에서 사용)"""
    x = torch.where(torch.isinf(x), torch.tensor(1e10, device=x.device), x)  # inf 값을 적당한 큰 값으로 대체
    logsum = -gamma * torch.logsumexp(-x / gamma, dim=-1)
    return logsum


class SoftDTW(nn.Module):
    def __init__(self, gamma=1.0):
        super(SoftDTW, self).__init__()
        self.gamma = gamma

    def forward(self, X, Y):
        batch_size, seq_len, _ = X.shape
        D = torch.cdist(X, Y, p=2)  # 유클리드 거리 계산

        R = torch.full((batch_size, seq_len + 1, seq_len + 1), float('inf'), device=X.device)
        R[:, 0, 0] = 0  # 초기화

        for i in range(1, seq_len + 1):
            for j in range(1, seq_len + 1):
                # Soft-Min 계산
                min_cost = softmin(torch.stack([R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]]), self.gamma)

                # min_cost는 (batch_size,) 형태로 계산되므로 (batch_size, 1, 1)로 확장하여 D와 더할 수 있도록 함
                min_cost_expanded = min_cost.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)로 확장
                R[:, i, j] = D[:, i - 1, j - 1] + min_cost_expanded.squeeze(-1).squeeze(-1)  # Broadcasting을 통해 더하기

        return R[:, -1, -1]  # 마지막 값 리턴


# 예제 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoftDTW(gamma=1.0).to(device)

X = torch.randn(1, 5, 1).to(device)  # (batch_size, sequence_length, feature_dim)
Y = torch.randn(1, 5, 1).to(device)

loss = model(X, Y)
print(f"Final SoftDTW Loss: {loss}")
