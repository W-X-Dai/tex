import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft as sfft
from scipy.io import wavfile


def Gabor(x, tau, t, f, sgm):
    """Scaled Gabor transform (unbalanced form).

    x   : input signal (1-D), sampled at tau
    tau : sampling points of the input on the t-axis (uniform, step dtau)
    t   : sampling points of the output on the t-axis (uniform, step dt)
    f   : sampling points on the f-axis (uniform, step df)
    sgm : scaling parameter sigma
    return y with shape (len(f), len(t))  (row: f, column: t, same as Matlab imagesc)
    """
    x = np.asarray(x).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    t = np.atleast_1d(np.asarray(t, dtype=float))
    f = np.atleast_1d(np.asarray(f, dtype=float))

    dtau = tau[1] - tau[0]
    df = f[1] - f[0] if f.size > 1 else 1.0

    # Constraint (1): dtau*df = 1/N, N must be an integer  (TFW4 p.133)
    N = int(round(1.0 / (dtau * df)))
    if abs(N * dtau * df - 1) > 1e-6:
        raise ValueError("1/(dtau*df) must be an integer")

    # Window width: exp(-sgm*pi*t^2) ~ 0 for |t| > B, B = 1.9/sqrt(sgm)
    B = 1.9 / np.sqrt(sgm)
    Q = int(np.ceil(B / dtau))
    # Constraint (2): N >= 2Q+1
    if N < 2 * Q + 1:
        raise ValueError(f"N = {N} < 2Q+1 = {2*Q+1}; use a smaller df")

    k = np.arange(-Q, Q + 1)                       # k = q - Q
    w = np.exp(-sgm * np.pi * (k * dtau) ** 2)     # window samples w(k*dtau)

    # input index of each window center (p_n = nS in TFW4 p.133)
    pc = np.rint((t - tau[0]) / dtau).astype(np.int64)
    # frequency index m and m1 = mod(m, N)  (TFW4 p.123)
    m = np.rint(f / df).astype(np.int64)
    m1 = np.mod(m, N)

    # zero padding so that x((p_n+k)*dtau) = 0 outside the input
    xpad = np.concatenate([np.zeros(Q, x.dtype), x, np.zeros(Q, x.dtype)])
    real_input = not np.iscomplexobj(x)
    if real_input:
        # real x -> use rFFT; X1(m) = conj(X1(N-m)) for m > N/2
        use_conj = m1 > N // 2
        col = np.where(use_conj, N - m1, m1)

    C = t.size
    y = np.empty((f.size, C), dtype=complex)
    chunk = max(1, int(2e7 // N))                  # limit memory of the FFT output
    for s in range(0, C, chunk):
        e = min(C, s + chunk)
        idx = pc[s:e, None] + k[None, :] + Q       # indices into xpad
        idx = np.clip(idx, 0, xpad.size - 1)       # t outside tau -> zeros (padding)
        out_of_range = (pc[s:e, None] + k[None, :] < 0) | (pc[s:e, None] + k[None, :] >= x.size)
        seg = xpad[idx] * w[None, :]               # x1(q), q = 0..2Q
        seg[out_of_range] = 0
        if real_input:
            X1 = sfft.rfft(seg, n=N, axis=1, workers=-1)[:, col]   # N-point FFT (zero padded)
            X1[:, use_conj] = np.conj(X1[:, use_conj])
        else:
            X1 = sfft.fft(seg, n=N, axis=1, workers=-1)[:, m1]
        # exp(j*2*pi*(Q - p_n)*m/N), integer arithmetic mod N keeps the phase exact
        ph = np.mod((Q - pc[s:e, None]) * m[None, :], N)
        y[:, s:e] = (X1 * np.exp(2j * np.pi * ph / N)).T

    # dtau * sigma^(1/4), and exp(-j*2*pi*f*tau0) if tau does not start at 0
    y *= dtau * sgm ** 0.25 * np.exp(-2j * np.pi * f * tau[0])[:, None]
    return y


def audioread(path):
    """Like Matlab audioread: returns float samples in [-1, 1] and fs."""
    fs, a = wavfile.read(path)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float64) / np.iinfo(a.dtype).max
    else:
        a = a.astype(np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return a, fs


if __name__ == "__main__":
    import sys

    wav = sys.argv[1] if len(sys.argv) > 1 else "Chord.wav"
    a1, fs = audioread(wav)
    x = a1[:, 0]                      # only extract the first channel
    tau = np.arange(x.size) / fs      # dtau = 1/fs (input sampling points)
    dt, df, sgm = 0.01, 1, 200
    t = np.arange(0, tau.max() + 1e-12, dt)
    f = np.arange(20, 1000 + df, df)

    tic = time.perf_counter()
    y = Gabor(x, tau, t, f, sgm)
    toc = time.perf_counter()
    print(f"fs = {fs} Hz, length = {tau.max():.3f} s, y.shape = {y.shape}")
    print(f"Elapsed time is {toc - tic:.6f} seconds.")

    plt.figure(figsize=(8, 4.5))
    plt.imshow(np.abs(y) / np.abs(y).max(), extent=[t[0], t[-1], f[0], f[-1]],
               origin="lower", aspect="auto", cmap="gray", vmin=0, vmax=0.5)
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Scaled Gabor transform of {wav} ($\\sigma$ = {sgm})")
    plt.tight_layout()
    plt.savefig("Gabor_result.png", dpi=150)
    plt.show()