import numpy as np
import matplotlib.pyplot as plt

# 1. 生成训练数据：在 [0, 2π] 上采样
np.random.seed(42)
N = 200        # 样本数
X = np.linspace(0, 2 * np.pi, N).reshape(-1, 1)  # (N, 1)
y = np.sin(X)                                      # (N, 1)

# 2. 定义网络结构
input_size = 1
hidden_size = 20   # 你可以改成 50, 100 试试看
output_size = 1
lr = 1e-2
epochs = 5000

# 3. 参数初始化（高斯随机）
W1 = 0.1 * np.random.randn(hidden_size, input_size)  # (H, 1)
b1 = np.zeros((1, hidden_size))                      # (1, H)
W2 = 0.1 * np.random.randn(output_size, hidden_size) # (1, H)
b2 = np.zeros((1, output_size))                      # (1, 1)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

loss_history = []

# 4. 训练循环
for epoch in range(epochs):
    # ---- 前向传播 ----
    # X: (N, 1)
    z1 = X @ W1.T + b1        # (N, H)
    a1 = relu(z1)             # (N, H)
    y_pred = a1 @ W2.T + b2   # (N, 1)

    # 损失：MSE
    diff = y_pred - y         # (N, 1)
    loss = np.mean(diff ** 2)
    loss_history.append(loss)

    # ---- 反向传播 ----
    # dL/dy_pred
    dL_dy = 2 * diff / N      # (N, 1)

    # 输出层梯度
    # y_pred = a1 @ W2.T + b2
    # dL/dW2 = dL/dy_pred^T @ a1
    dW2 = dL_dy.T @ a1        # (1, H)
    db2 = np.sum(dL_dy, axis=0, keepdims=True)  # (1, 1)

    # 传回到隐藏层
    # a1: (N, H), W2: (1, H)
    da1 = dL_dy @ W2          # (N, H)
    dz1 = da1 * relu_grad(z1) # (N, H)

    # 隐藏层权重梯度
    # z1 = X @ W1.T + b1
    dW1 = dz1.T @ X           # (H, 1)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, H)

    # ---- 参数更新（梯度下降） ----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}, Loss = {loss:.6f}")

# 5. 可视化结果
with np.errstate(invalid='ignore'):
    plt.figure()
    plt.scatter(X, y, s=10, label='True sin(x)')
    plt.scatter(X, y_pred, s=10, label='MLP prediction')
    plt.legend()
    plt.title("MLP approximation to sin(x)")
    plt.tight_layout()
    plt.savefig("relu_mlp_sinx.png", dpi=300)
    plt.close()


# 可以顺便画一下损失下降曲线
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.tight_layout()
plt.savefig("relu_mlp_sinx_loss.png", dpi=300)
plt.close()
