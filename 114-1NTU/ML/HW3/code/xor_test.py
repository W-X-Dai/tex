import numpy as np
from myann import ANN # type:ignore

# XOR dataset
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)

Y = np.array([0,1,1,0], dtype=float)

# 建 ANN 模型
ann = ANN(lr=0.1)
ann.add_linear(2, 3)
ann.add_sigmoid()
ann.add_linear(3, 1)
ann.add_sigmoid()

# 訓練迴圈
epochs = 10000
for epoch in range(epochs):
    total_loss = 0.0
    for i in range(4):
        x = X[i]
        y = Y[i]

        out = ann.forward(x)
        y_pred = float(out[0])
        loss = ann.bce(y, y_pred)
        total_loss += loss

        ann.backward(y, y_pred)
        ann.update()

    if epoch % 500 == 0:
        print(f"epoch {epoch}, loss = {total_loss/4:.6f}")

# 測試輸出
print("\nFinal predictions:")
for i in range(4):
    x = X[i]
    out = ann.forward(x)
    print(f"{x} => {float(out[0]):.6f}")
