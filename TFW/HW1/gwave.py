import numpy as np
from scipy.io import wavfile


def gwave(a, b, c, T, Fs, filename="gwave.wav"):
    N = int(round(T * Fs))
    t = np.arange(N) / Fs

    phi = 2 * np.pi * (a * t**3 / 3 + b * t**2 / 2 + c * t)
    x = np.cos(phi)

    wavfile.write(filename, int(Fs), np.int16(0.9 * x * 32767))
    return t, x

if __name__ == "__main__":
    gwave(100, 200, 300, 3, 8000)