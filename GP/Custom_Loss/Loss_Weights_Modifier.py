import torch
import torch.nn as nn
import torch.optim as optim
#from chemprop.train.loss_functions import sid_loss
from Graphormer.GP5.Custom_Loss.sdtw_cuda_loss import SoftDTW

from Graphormer.GP5.Custom_Loss.sdtw_python.soft_dtw import soft_dtw

def sid_loss(
    model_spectra: torch.tensor,
    target_spectra: torch.tensor,
    mask: torch.tensor,
    threshold: float = None,
) -> torch.tensor:
    """
    Loss function for use with spectra graphormer_data type.

    :param model_spectra: The predicted spectra output from a model with shape (batch_size,spectrum_length).
    :param target_spectra: The target spectra with shape (batch_size,spectrum_length). Values must be normalized so that each spectrum sums to 1.
    :param mask: Tensor with boolean indications of where the spectrum output should not be excluded with shape (batch_size,spectrum_length).
    :param threshold: Loss function requires that values are positive and nonzero. Values below the threshold will be replaced with the threshold value.
    :return: A tensor containing loss values for the batch with shape (batch_size,spectrum_length).
    """
    # Move new tensors to torch device
    torch_device = model_spectra.device

    # Normalize the model spectra before comparison
    zero_sub = torch.zeros_like(model_spectra, device=torch_device)
    one_sub = torch.ones_like(model_spectra, device=torch_device)
    if threshold is not None:
        threshold_sub = torch.full(model_spectra.shape, threshold, device=torch_device)
        model_spectra = torch.where(
            model_spectra < threshold, threshold_sub, model_spectra
        )
    model_spectra = torch.where(mask, model_spectra, zero_sub)

    sum_model_spectra = torch.sum(model_spectra, axis=1, keepdim=True)
    model_spectra = torch.div(model_spectra, sum_model_spectra)

    # Calculate loss value
    target_spectra = torch.where(mask, target_spectra, one_sub)
    model_spectra = torch.where(
        mask, model_spectra, one_sub
    )  # losses in excluded regions will be zero because log(1/1) = 0.

    loss = torch.mul(
        torch.log(torch.div(model_spectra, target_spectra)), model_spectra
    ) + torch.mul(torch.log(torch.div(target_spectra, model_spectra)), target_spectra)
    return loss

class FirstLossNormalization:
    def __init__(self, epsilon=1e-8, device="cuda"):
        #self.num_losses = num_losses
        self.epsilon = epsilon
        self.device = torch.device(device)

        # 첫 번째 손실값을 저장할 변수 (첫 번째 epoch에 저장됨)
        self.first_loss = None

    def update_weights(self, current_losses):
        """
        첫 번째 loss만 기준으로 정규화하여 가중치를 계산.
        """
        current_losses = torch.tensor(current_losses, dtype=torch.float32, device=self.device)

        # 첫 번째 epoch에서 첫 번째 손실 값 저장
        if self.first_loss is None:
            self.first_loss = current_losses.clone().requires_grad_(True) # 첫 번째 손실만 저장

        # 첫 번째 손실(`self.first_loss`)을 기준으로 정규화
        normalized_losses = current_losses / (self.first_loss + self.epsilon)
        #print(normalized_losses.tolist())
        return normalized_losses

import torch
import torch.nn as nn

class DWA:
    def __init__(self, num_losses, window_size=2, temperature=1.0):
        self.num_losses = num_losses
        self.window_size = window_size
        self.temperature = temperature  # 🔹 소프트맥스 온도 매개변수 추가
        self.history = torch.zeros((window_size, num_losses)).to("cuda")  # 최근 loss 값 저장
        self.device = torch.device("cuda")

    def update_weights(self, current_losses, epoch):
        """
        Loss 변화율을 기반으로 새로운 가중치 계산 (소프트맥스 적용).
        """
        current_losses = torch.tensor(current_losses, dtype=torch.float32, device=self.device)

        # 초기 몇 epoch 동안은 균등 가중치 사용
        if epoch < self.window_size:
            return torch.ones(self.num_losses, dtype=torch.float32, device=self.device) / self.num_losses

        # loss 기록 업데이트
        self.history = torch.cat((self.history[1:], current_losses.unsqueeze(0)), dim=0)

        # Loss 변화율 기반 가중치 조정
        ratios = torch.clamp(self.history[-1] / (self.history[-2] + 1e-8), min=0.5, max=2)

        # 🔹 소프트맥스 적용하여 가중치 계산 (온도 매개변수 포함)
        exp_ratios = torch.exp(-ratios / self.temperature)
        weights = torch.softmax(exp_ratios, dim=0)

        return weights


class DWAWithNormalization:
    def __init__(self, num_losses, window_size=2, epsilon=1e-8, temperature=1.0):
        self.num_losses = num_losses
        self.window_size = window_size
        self.epsilon = epsilon
        self.temperature = temperature  # 🔹 소프트맥스 온도 매개변수 추가
        self.history = torch.zeros((window_size, num_losses)).to("cuda")  # 최근 loss 값 저장
        self.weights = nn.Parameter(torch.ones(num_losses, dtype=torch.float32))  # 학습 가능한 가중치
        self.initial_losses = None  # 첫 번째 epoch의 손실 값을 저장할 변수
        self.device = torch.device("cuda")

    def update_weights(self, current_losses, epoch):
        """
        Loss 변화율을 기반으로 새로운 가중치 계산 (초기 손실 기반 정규화 포함 + Softmax 적용).
        """
        current_losses = torch.tensor(current_losses, dtype=torch.float32, device=self.device)

        # 첫 번째 epoch에서 초기 손실 값 저장
        if self.initial_losses is None:
            self.initial_losses = current_losses.clone().detach()

        # 초기 손실을 기준으로 정규화
        normalized_losses = current_losses / (self.initial_losses + self.epsilon)

        # 초기 window_size 동안 손실값 저장 단계
        if epoch < self.window_size:
            self.history[epoch] = normalized_losses.clone().detach()
            return torch.ones(self.num_losses, dtype=torch.float32, device=self.device) / self.num_losses  # 균등 가중치 반환

        # history 업데이트 (이전 값 유지)
        self.history[:-1] = self.history[1:].clone()
        self.history[-1] = normalized_losses.clone().detach()

        # Loss 변화율 기반 가중치 조정
        ratios = torch.clamp(self.history[-1] / (self.history[-2] + self.epsilon), min=0.5, max=2)

        # 🔹 소프트맥스 적용하여 가중치 계산 (온도 매개변수 포함)
        exp_ratios = torch.exp(-torch.abs(ratios) / self.temperature)
        self.weights.data = torch.softmax(exp_ratios, dim=0)

        return self.weights

class ModifiedUncertaintyWeightedLoss(nn.Module):
    def __init__(self, num_losses, gamma=0.02):
        super().__init__()
        self.num_losses = num_losses
        self.log_sigmas = nn.Parameter(torch.zeros(num_losses, dtype=torch.float32, requires_grad=True))
        self.gamma = gamma

    def forward(self, losses, epoch):
        """
        Uncertainty 기반 Loss 조정
        """
        sigmas = torch.exp(self.log_sigmas)
        decay_factor = torch.exp(torch.tensor(-self.gamma * epoch, dtype=torch.float32))

        losses = torch.tensor(losses, dtype=torch.float32, requires_grad=True)
        total_loss = sum((loss / (2 * sigmas[i] ** 2)) * decay_factor for i, loss in enumerate(losses))
        total_loss += torch.sum(self.log_sigmas)  # 정규화 항 추가
        return total_loss


class GradNorm:
    def __init__(self, num_losses, alpha=0.12):
        self.num_losses = num_losses # 사용할 손실 함수 개수
        self.alpha = alpha # GradNorm의 스케일링 조정 계수
        self.lambdas = nn.Parameter(torch.ones(num_losses, dtype=torch.float32, device="cuda")) # 손실 가중치 벡터 (num_losses 크기의 학습 가능한 텐서, 초기값 1)

    def compute_weights(self, losses, model): # 여러 개의 손실 값 리스트 losses # losses 리스트 안에는 여러 loss의 텐서가 들어감 / 예:[SID, SoftDTW]
        """
        Gradient Norm 기반 가중치 조정
        """
        grads = []
        for loss in losses:

            # 1. 각 손실의 그래디언트 L2 norm 계산
            # model.parameters() 가중치와 Bias를 반환
            # grad 리스트에는 model.parameters()(즉, 모델의 가중치)마다 계산된 그래디언트 텐서들이 저장됨
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True) # None 값이 반환될 경우 기본값 0으로 처리, 각 손실에 대한 그래디언트를 계산


            # 1-1. g.norm() → g(각 가중치의 그래디언트 텐서)에 대한 L2 norm을 계산, 만약 g가 None이면 torch.tensor(0.0)을 반환하여 NaN 방지, None 값이 있는 이유는 어떤 손실이 특정 가중치에 영향을 주지 않을 수도 있기 때문
            # 1-2. torch.stack() 모든 노름 값을 하나의 텐서로 변환
            # 1-3. 최종적으로 L2 norm을 계산 torch.norm(torch.stack([...]))
            # L2 Norm = 제곱 합한후 제곱근을 씌우는것
            grad_norm = torch.norm(torch.stack([g.norm() if g is not None else torch.tensor(0.0) for g in grad]))

            grads.append(grad_norm)

        # 2. 모든 작업의 평균 그래디언트 norm 계산
        grads = torch.stack(grads) # 모든 그래디언트 크기를 하나의 텐서로 변환 -> 개별 loss 들의 graident 크기
        mean_grad = grads.mean() + 1e-8 # 0 방지, n개 loss의 graident 평균

        # 3. 상대적 손실 비율 계산
        relative_losses = losses / (losses[0] + 1e-8)  # 초기 손실 대비 변화율 계산
        r_i = relative_losses / (relative_losses.mean() + 1e-8)  # 상대적 손실 비율

        # GradNorm 업데이트 공식 적용
        loss_weights = self.lambdas * (grads / mean_grad) ** self.alpha # GradNorm 공식 적용

        # Softmax 정규화
        loss_weights = loss_weights / (loss_weights.sum() + 1e-8) # 1e-8 -> 0 방지
        return loss_weights

    def update_lambdas(self, loss_weights):
        with torch.no_grad():
            self.lambdas.copy_(loss_weights)

class GradNormWithNormalization:
    def __init__(self, num_losses, alpha=0.12, epsilon=1e-8):
        self.num_losses = num_losses
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambdas = nn.Parameter(torch.ones(num_losses, dtype=torch.float32, device="cuda"))  # 초기 가중치 1로 설정
        self.initial_losses = None  # 초기 손실 저장

    def compute_weights(self, losses, model):
        # 🔹 `losses`가 리스트 형태일 수 있으므로 `stack()`으로 변환
        losses = torch.stack(losses)

        # 🔹 첫 번째 epoch의 손실 저장 (초기 기준값)
        if self.initial_losses is None:
            self.initial_losses = losses.clone().detach()

        # 🔹 손실 정규화 (안정적인 학습을 위해 `epsilon` 추가)
        normalized_losses = losses / (self.initial_losses + self.epsilon)
        print("normalized_losses", normalized_losses.tolist())

        # 🔹 모델의 모든 손실에 대한 그래디언트 크기 계산
        grads = []
        for loss in normalized_losses:
            grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
            grad_norm = torch.norm(torch.stack([g.norm() if g is not None else torch.tensor(0.0) for g in grad]))
            grads.append(grad_norm)

        grads = torch.stack(grads)  # 그래디언트 크기 텐서 변환
        mean_grad = grads.mean() + self.epsilon  # 평균 그래디언트 크기

        # 🔹 GradNorm 공식 적용 (정규화된 손실 & 그래디언트 크기 사용)
        loss_weights = self.lambdas * (grads / mean_grad) ** self.alpha

        # 🔹 Softmax 정규화 적용 (가중치 총합이 1이 되도록)
        loss_weights = loss_weights / (loss_weights.sum() + self.epsilon)
        return loss_weights

if __name__ == '__main__':
    # ✅ 간단한 모델 정의 (입력 1개 -> 출력 10개)
    # ✅ 실행 설정
    a = "DWAWithNormalization"  # "DWA", "DWAWithNormalization", "Uncertainty", "GradNorm", "GradNormWithNormalization" "FirstLossNormalization"중 선택

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
    SoftDTWLoss = SoftDTW(use_cuda=True, gamma=1.0, bandwidth=None)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    huber_loss = nn.SmoothL1Loss()


    def sid_loss_wrapper(model_spectra, target_spectra):
        mask = torch.ones_like(model_spectra, dtype=torch.bool).to(device)
        SID_LOSS = sid_loss(model_spectra, target_spectra, mask, threshold=1e-6)
        return SID_LOSS


    # ✅ Loss Weighting 설정
    if a == "DWA":
        loss_modifier = DWA(num_losses=5, window_size=2)
    elif a == "DWAWithNormalization":
        loss_modifier = DWAWithNormalization(num_losses=5, window_size=10)
    elif a == "Uncertainty":
        loss_modifier = ModifiedUncertaintyWeightedLoss(num_losses=5)
    elif a == "GradNorm":
        loss_modifier = GradNorm(num_losses=5, alpha=0.12)
    elif a == "GradNormWithNormalization":
        loss_modifier = GradNormWithNormalization(num_losses=5, alpha=0.12)
    elif a == "FirstLossNormalization":
        loss_modifier = FirstLossNormalization()

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

        if a in ["DWA", "DWAWithNormalization"]:
            weights = loss_modifier.update_weights([loss.item() for loss in losses], epoch)
        elif a in ["GradNorm", "GradNormWithNormalization"]:
            weights = loss_modifier.compute_weights(losses, model)
        elif a == "Uncertainty":
            weights = loss_modifier(losses, epoch)

        if a in ["DWA", "DWAWithNormalization","GradNorm", "GradNormWithNormalization","Uncertainty"]:
            L_new = sum(weights[i] * losses[i] for i in range(len(losses)))
            L_new.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {L_new.item()}, Weights: {weights.cpu().detach().numpy()}")
        elif a == "FirstLossNormalization":
            L_new = loss_modifier.update_weights(losses)
            L_new.sum().backward()
            optimizer.step()
            #print(f"Epoch {epoch}, Loss: {L_new},")




