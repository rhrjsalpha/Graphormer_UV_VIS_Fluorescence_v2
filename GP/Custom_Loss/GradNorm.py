#from Graphormer.GP.Custom_Loss.sdtw_cuda_loss import SoftDTW
from GP.Custom_Loss.SID_loss import sid_loss
import torch.optim as optim
#from tslearn.metrics import SoftDTWLossPyTorch
from GP.Custom_Loss.soft_dtw_cuda import SoftDTW
import torch
import torch.nn as nn


class GradNorm:
    def __init__(self, num_losses, alpha=0.12):
        self.num_losses = num_losses  # Number of different loss functions
        self.alpha = alpha  # Scaling factor for GradNorm
        self.lambdas = nn.Parameter(torch.ones(self.num_losses, dtype=torch.float32, device="cuda"))  # Initialize loss weights
        self.initial_losses = None  # To store initial loss values

    def compute_weights(self, losses, model):
        """
        Compute dynamic weights for each loss based on gradient norms.

        Args:
            losses (list of torch.Tensor): List containing individual loss values.
            model (torch.nn.Module): The neural network model being trained.

        Returns:
            torch.Tensor: Adjusted weights for each loss.
        """
        #device = self.lambdas.device
        #print("device",device)
        # print("losses",losses)
        # losses: Tensor([L1, L2, ...]) or list of scalars
        if isinstance(losses, list):
            losses = torch.stack([l if torch.is_tensor(l) else torch.tensor(l, device=self.lambdas.device)
                                  for l in losses])

        # 🔧 1) trainable params만 사용
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            # 학습 가능한 파라미터가 없으면 균등가중치로 리턴
            return torch.ones_like(self.lambdas) / self.num_losses

        grads = []
        for i, loss in enumerate(losses):
            # 🔧 2) 해당 loss가 grad 트랙 안 타면 0으로 대체
            if not torch.is_tensor(loss) or not loss.requires_grad:
                grads.append(torch.zeros((), device=self.lambdas.device))
                continue

            g = torch.autograd.grad(
                loss, params,
                retain_graph=True, create_graph=False, allow_unused=True
            )
            g = [gi for gi in g if gi is not None]
            if len(g) == 0:
                grads.append(torch.zeros((), device=self.lambdas.device))
            else:
                grads.append(torch.norm(torch.stack([gi.norm() for gi in g])))

        grads1 = torch.stack(grads).clamp_min(1e-8)

        #print("grads1", grads1.tolist())
        if 0 in grads1.tolist():
            print("0 in grads list ",grads1, grads)

        # Initialize initial losses during the first call
        if self.initial_losses is None:
            self.initial_losses = losses.detach()
        #else:
        #    # 초기 손실보다 클 경우 업데이트
        #    self.initial_losses = torch.where(losses > self.initial_losses, losses.detach(), self.initial_losses)

        # Compute relative losses
        relative_losses = losses / (self.initial_losses + 1e-8)
        r_i = relative_losses / relative_losses.mean()

        # Compute adjusted loss weights
        adjusted_factor = (grads1 / grads1.mean()) * r_i
        loss_weights = self.lambdas * (adjusted_factor ** self.alpha)

        # Normalize weights to maintain the sum to num_losses
        loss_weights = (self.num_losses * loss_weights) / loss_weights.sum()
        loss_weights = torch.clamp(loss_weights, min=1e-3, max=5.0)
        #print("loss_weights",loss_weights)

        #print(loss_weights)
        return loss_weights

    def update_lambdas(self, loss_weights):
        """
        Update the internal lambda parameters with the new loss weights.

        Args:
            loss_weights (torch.Tensor): The newly computed loss weights.
        """
        with torch.no_grad():
            self.lambdas.copy_(loss_weights.detach())
            #self.lambdas = loss_weights.clone().detach()

class GradNorm_new:
    def __init__(self, num_losses, alpha=0.12):
        self.num_losses = num_losses  # 사용할 손실 함수 개수
        self.alpha = alpha  # GradNorm의 스케일링 조정 계수
        self.lambdas = nn.Parameter(torch.ones(num_losses, dtype=torch.float32, device="cuda"))  # 손실 가중치 벡터 (초기값 1)
        self.initial_losses = None  # 초기 손실 저장

    def compute_weights(self, losses, model):
        """
        Gradient Norm 기반 가중치 조정 (첫 번째 epoch은 gradient만 사용)
        """
        if isinstance(losses, list):
            losses = torch.cat([loss.unsqueeze(0) for loss in losses])

        # ** 음수 손실 방지 및 작은 값 보정**
        losses = torch.abs(losses) + 1e-8
        print("losses:", losses)

        # 1. **첫 번째 epoch에서는 `grads`만 고려**
        if self.initial_losses is None:
            self.initial_losses = losses.clone().detach()
            print("🔹 First epoch detected: Using gradients only (no relative loss computation).")

        # 2. **손실 변화량을 고려한 `adjusted_losses` 계산 (첫 번째 epoch 제외)**
        if self.initial_losses is not None:
            adjusted_losses = losses / (torch.abs(losses - self.initial_losses) + 1e-8)
        else:
            adjusted_losses = losses.clone()  # 첫 epoch에서는 조정 없이 사용

        adjusted_losses = torch.clamp(adjusted_losses, min=1e-6, max=1e6)  # NaN 방지
        print("adjusted_losses:", adjusted_losses)

        # 3. 각 손실의 그래디언트 L2 norm 계산
        grads = []
        for loss in losses:
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=False)
            grad_norm = torch.norm(torch.stack([g.norm() if g is not None else torch.tensor(0.0) for g in grad]))
            grads.append(grad_norm)
        grads = torch.stack(grads)
        grads = torch.clamp(grads, min=1e-8, max=1e6)  # NaN 방지
        print("grads_GradNorm:", grads)

        # 4. **첫 번째 epoch에서는 `grads`만 사용**
        if self.initial_losses is None:
            adjusted_factor = grads / (grads.sum() + 1e-8)  # 첫 epoch에서는 gradient 비율만 사용
        else:
            # 5. 상대적 손실 비율 계산 (r_i 수정)
            r_i = adjusted_losses / (adjusted_losses.mean() + 1e-8)
            r_i = torch.clamp(r_i, min=1e-6, max=1e6)  # NaN 방지
            print("relative_losses (r_i):", r_i)

            # 6. GradNorm 가중치 업데이트 공식 적용
            adjusted_factor = (grads / grads.mean() + 1e-8) * r_i  # 첫 epoch 이후에는 Lt / (Lt - L0) 방식 적용
            adjusted_factor = torch.clamp(adjusted_factor, min=1e-6, max=1e6)  # NaN 방지
            print("adjusted_factor:", adjusted_factor)

        # 7. 손실 가중치 업데이트
        loss_weights = self.lambdas * adjusted_factor ** self.alpha

        # 8. Softmax 정규화 적용 (가중치 총합을 1로 유지)
        loss_weights = loss_weights / (loss_weights.sum() + 1e-8)

        # 9. 극단적인 값 방지
        if torch.any(loss_weights < 0.01):
            print("⚠ Warning: Loss weights too small. Adjusting weights.")
            loss_weights = torch.clamp(loss_weights, min=0.01, max=0.99)
            loss_weights = loss_weights / (loss_weights.sum() + 1e-8)

        if torch.any(torch.isnan(loss_weights) | torch.isinf(loss_weights)):
            print("⚠ Warning: Invalid loss weights detected (NaN/Inf). Resetting to equal weights.")
            loss_weights = torch.ones_like(loss_weights) / self.num_losses  # NaN/Inf 발생 시 초기화
            loss_weights = loss_weights / (loss_weights.sum() + 1e-8)

        print("Final loss weights:", loss_weights)
        return loss_weights

    def update_lambdas(self, loss_weights):
        """
        손실 가중치 업데이트
        """
        with torch.no_grad():
            self.lambdas.copy_(loss_weights)


class GradNorm_exp:
    def __init__(self, num_losses, alpha=0.12):
        self.num_losses = num_losses  # 사용할 손실 함수 개수
        self.alpha = alpha  # GradNorm의 스케일링 조정 계수
        self.lambdas = nn.Parameter(torch.ones(num_losses, dtype=torch.float32, device="cuda"))  # 손실 가중치 벡터 (초기값 1)
        self.initial_losses = None  # 초기 손실 저장

    def compute_weights(self, losses, model):
        """
        Gradient Norm 기반 가중치 조정 (e^L 변환 및 상대적 손실 비율 적용)
        """
        for loss in losses:
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True,
                                       allow_unused=True)
            grad_norm = torch.norm(torch.stack([g.norm() if g is not None else torch.tensor(0.0) for g in grad]))
            print("grad_1", grad_norm)

        if isinstance(losses, list):
            #print(type(losses), len(losses))
            #print(losses)
            losses = torch.cat([loss.unsqueeze(0) for loss in losses])

            #print("losses after\n",losses)

        print("losses",losses)
        # 1. e^L 변환 적용 (손실을 항상 양수로 변환)
        exp_losses = torch.exp(losses)
        print("exp_losses", exp_losses)

        # 2. 초기 손실 저장 (첫 번째 epoch에서 설정)
        if self.initial_losses is None:
            self.initial_losses = exp_losses.clone().detach()  # 초기 손실을 저장

        # 3. 각 손실의 그래디언트 L2 norm 계산
        grads = []
        for loss in exp_losses:
            #print(type(loss), loss.shape, loss)
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
            grad_norm = torch.norm(torch.stack([g.norm() if g is not None else torch.tensor(0.0) for g in grad]))  # L2 norm 계산
            grads.append(grad_norm)
        #print("grads_GradNorm",grads)

        # 4. 모든 작업의 평균 그래디언트 크기 계산
        grads = torch.stack(grads)  # 모든 그래디언트 크기를 하나의 텐서로 변환
        mean_grad = grads.mean() + 1e-8  # 0 방지

        # 5. 상대적 손실 비율 계산 (그림의 공식 적용)
        relative_losses = exp_losses / (self.initial_losses + 1e-8)  # 초기 손실 대비 변화율
        r_i = relative_losses / (relative_losses.mean() + 1e-8)  # 상대적 손실 비율

        # 6. GradNorm 가중치 업데이트 공식 적용,
        #print("normal",(grads / mean_grad) * r_i)
        adjusted_factor = torch.log1p((grads / mean_grad) * r_i) # 새로운식 e^loss 해준것 보정
        #print("adj",adjusted_factor)
        loss_weights = self.lambdas * adjusted_factor ** self.alpha
        #loss_weights_1 = self.lambdas * ((grads / mean_grad) * r_i) ** self.alpha  # `r_i` 반영 , 손실 가중치 업데이트 (기존식)

        # 7. Softmax 정규화 적용 (가중치 총합을 1로 유지)
        loss_weights = loss_weights / (loss_weights.sum() + 1e-8)

        if torch.any(loss_weights < 0.01):
            print("⚠ Warning: Invalid loss weights detected (0 values found). Adjusting weights.")
            loss_weights[loss_weights <= 0.01] = 0.01  # 0인 값은 1e-8로 설정
            loss_weights = loss_weights / (loss_weights.sum() + 1e-8)

        elif torch.any(loss_weights > 0.99):
            print("⚠ Warning: Invalid loss weights detected (NaN, inf values found). Adjusting weights.")
            loss_weights[torch.isnan(loss_weights) | torch.isinf(loss_weights)] = 0.99  # NaN 또는 inf인 값은 0.999로 설정
            loss_weights = loss_weights / (loss_weights.sum() + 1e-8)

        return loss_weights

    def update_lambdas(self, loss_weights):
        """
        손실 가중치 업데이트
        """
        with torch.no_grad():
            self.lambdas.copy_(loss_weights)


if __name__ == '__main__':
    # ✅ 간단한 모델 정의 (입력 1개 -> 출력 10개)
    # ✅ 실행 설정
    a = "GradNorm"  # "DWA", "DWAWithNormalization", "Uncertainty", "GradNorm", "GradNormWithNormalization" "FirstLossNormalization"중 선택

    # ✅ 모델 정의
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(1, 10)
            self.activation = nn.Softplus()  # 항상 양수 출력

        def forward(self, x):
            return self.activation(self.fc(x))  # 활성화 함수 적용


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ✅ 손실 함수 정의
    SoftDTWLoss = SoftDTW(use_cuda=True, gamma=0.2, bandwidth=None, normalize=True)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    huber_loss = nn.SmoothL1Loss()


    #def sid_loss_wrapper(model_spectra, target_spectra):
    #    mask = torch.ones_like(model_spectra, dtype=torch.bool).to(device)
    #    SID_LOSS = sid_loss(model_spectra, target_spectra, mask, threshold=1e-6)
    #    return SID_LOSS
    def sid_loss_wrapper(model_spectra, target_spectra):
        mask = torch.ones_like(model_spectra, dtype=torch.bool, device=device)
        # reduction="mean_valid" → 샘플별 유효길이로 나눈 뒤 배치 평균(가변 길이 공정성)
        SID_LOSS = sid_loss(
            model_spectra,
            target_spectra,
            mask,
            eps=1e-6,
            reduction="mean_valid",
        )
        return SID_LOSS


    # ✅ Loss Weighting 설정
    if a == "GradNorm":
        loss_modifier = GradNorm(num_losses=5, alpha=0.12)


    # ✅ 학습 과정
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        x = torch.randn(1, 1).to(device)
        y_true = torch.abs(torch.randn(1, 10).to(device))
        y_pred = model(x)

        losses = [
            mse_loss(y_pred, y_true),
            mae_loss(y_pred, y_true),
            huber_loss(y_pred, y_true),
            sid_loss_wrapper(y_pred, y_true).mean(),
            SoftDTWLoss(y_pred.unsqueeze(-1), y_true.unsqueeze(-1)).mean()
        ]
        #print("losses4", losses[4])

        if a in ["GradNorm", "GradNormWithNormalization"]:
            weights = loss_modifier.compute_weights(losses, model)


        if a in ["GradNorm", "GradNormWithNormalization"]:
            L_new = sum(weights[i] * losses[i] for i in range(len(losses)))
            L_new.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {L_new.item()}, Weights: {weights.cpu().detach().numpy()}")
        elif a == "FirstLossNormalization":
            L_new = loss_modifier.update_weights(losses)
            L_new.sum().backward()
            optimizer.step()