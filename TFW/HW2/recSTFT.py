import time
import numpy as np
import matplotlib.pyplot as plt



def recSTFT(x, t, f, B):
    x = np.asarray(x, dtype=complex)
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)

    dt = t[1] - t[0]
    df = f[1] - f[0]
    T, F = len(t), len(f)
    N = int(round(1.0 / (dt * df)))
    Q = int(round(B / dt))
    
    n0 = int(round(t[0] / dt))
    m0 = int(round(f[0] / df))
    n = n0 + np.arange(T)
    m = m0 + np.arange(F)

    xp = np.concatenate([np.zeros(Q, complex), x, np.zeros(Q, complex)])
    seg = np.lib.stride_tricks.sliding_window_view(xp, 2 * Q + 1)
    x1 = np.zeros((T, N), dtype=complex)
    x1[:, :2 * Q + 1] = seg

    X1 = np.fft.fft(x1, axis=1)

    X1m = X1[:, np.mod(m, N)]
    phase = np.exp(1j * 2 * np.pi * np.outer(Q - n, m) / N)
    y = dt * phase * X1m
    return y.T


def recSTFT_direct(x, t, f, B):
    dt = t[1] - t[0]
    Q = int(round(B / dt))
    y = np.zeros((len(f), len(t)), dtype=complex)
    for i in range(len(t)):
        lo, hi = max(0, i - Q), min(len(t) - 1, i + Q)
        tau = t[lo:hi + 1]
        y[:, i] = dt * np.exp(-1j * 2 * np.pi * np.outer(f, tau)) @ x[lo:hi + 1]
    return y


if __name__ == "__main__":
    dt, df = 0.05, 0.05
    t1 = np.arange(0, 10, dt)
    t2 = np.arange(10, 20, dt)
    t3 = np.arange(20, 30 + dt / 2, dt)
    t = np.arange(0, 30 + dt / 2, dt)
    f = np.arange(-5, 5 + df / 2, df)
    x = np.concatenate([np.cos(2 * np.pi * t1), np.cos(6 * np.pi * t2), np.cos(4 * np.pi * t3)])
    B = 1

    recSTFT(x, t, f, B)
    tic = time.perf_counter()
    y = recSTFT(x, t, f, B)
    toc = time.perf_counter()
    print(f"FFT-based rec-STFT running time: {toc - tic:.6f} s")

    tic = time.perf_counter()
    y_d = recSTFT_direct(x, t, f, B)
    toc = time.perf_counter()
    print(f"Direct rec-STFT running time: {toc - tic:.6f} s")
    print("max |FFT-based - direct| =", np.max(np.abs(y - y_d)))

    C = 400
    img = np.abs(y) / np.max(np.abs(y)) * C
    plt.figure(figsize=(8, 4))
    plt.imshow(img, cmap="gray", origin="lower", aspect="auto",
               extent=[t[0], t[-1], f[0], f[-1]])
    plt.xlabel("Time (Sec)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"rec-STFT of x(t), B = {B}")
    plt.tight_layout()
    plt.savefig("recSTFT_result.png", dpi=200)
    plt.show()