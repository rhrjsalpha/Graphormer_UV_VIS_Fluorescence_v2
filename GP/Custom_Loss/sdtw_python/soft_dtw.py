import numpy as np

def soft_dtw(D, gamma):
    """
    Soft-DTW loss 계산.

    Args:
        D (np.ndarray): 거리 행렬.
        gamma (float): 스케일링 파라미터.

    Returns:
        float: Soft-DTW 값.
    """
    m, n = D.shape
    R = np.full((m + 1, n + 1), np.inf)
    R[0, 0] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            R[i, j] = D[i - 1, j - 1] + min(R[i - 1, j], R[i - 1, j - 1], R[i, j - 1])

    return R[m, n]


def soft_dtw_grad(D, gamma):
    """
    Soft-DTW의 그라디언트 계산.

    Args:
        D (np.ndarray): 거리 행렬.
        gamma (float): 스케일링 파라미터.

    Returns:
        np.ndarray: 그라디언트 행렬.
    """
    m, n = D.shape
    E = np.zeros((m + 1, n + 1))
    E[m, n] = 1

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i - 1, j - 1] = E[i, j]  # 역방향 업데이트

    return E[1:, 1:]
