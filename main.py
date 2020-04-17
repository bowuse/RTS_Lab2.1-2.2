import matplotlib.pyplot as plt
import math
from random import *
import numpy as np
from multiprocessing import Pool
from time import time

N = 256
w = 900
n = 10



def generateFreq(w, n):
    freq = []
    step = w / n
    for i in range(n):	
        freq.append(w - step * i)
    return freq


def generateXt(N, n, A, freq, alpha):
    x = [0] * N
    for j in range(N):
        for i in range(n):
            x[j] += A[i] * math.sin(freq[i] * j + alpha[i])
    return x


def mathExpecation(x, N):
    mx = 0.0
    for i in range(N):
        mx += x[i]
    return mx / N


def dispersion(x, N, mx):
    dx = 0.0
    for i in range(N):
        dx += math.pow(x[i] - mx, 2)
    return dx / (N - 1)


def arrGenerator(n, min, max):
    arr = [0] * n
    for i in range(n):
        arr[i] = randint(min, max)
    return arr

def dpf(signal):
    n = len(signal)
    p = np.arange(n)
    k = p.reshape((n, 1))
    w = np.exp(-2j * np.pi * p * k / n)
    return np.dot(w, signal)

def fft_treads(signal):
    p = Pool(2)
    signal = np.asanyarray(signal, dtype=float)
    signals = (signal[::2], signal[1::2])
    N = len(signal)
    signal_even, signal_odd = p.map(fft, signals)
    terms = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([signal_even + terms[:N // 2] * signal_odd,
                           signal_even + terms[N // 2:] * signal_odd])

def fft(signal):
    signal = np.asanyarray(signal, dtype=float)
    N = len(signal)
    if N <= 2:
        return dpf(signal)
    else:
        signal_even = fft(signal[::2])
        signal_odd = fft(signal[1::2])
        terms = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([signal_even + terms[:N // 2] * signal_odd,
                               signal_even + terms[N // 2:] * signal_odd])


def draw_DPF():
    plt.title("DPF")
    plt.plot(t, x_dpf_real, 'b', t, x_dpf_img, 'r')
    plt.show()

def draw_FFT(t, x_fft_real, x_fft_img):
    plt.title("FFT")
    plt.plot(t, x_fft_real, 'b', t, x_fft_img, 'r')
    plt.show()


def time_measure(N, func):
    s = time()
    func(N)
    score = time() - s
    print("N={} {}s".format(N, score))

def main(N):
    A = arrGenerator(n, 0, 5)
    alpha = arrGenerator(n, 0, 5)
    freq = generateFreq(w, n)
    x = generateXt(N, n, A, freq, alpha)
    x_fft = fft_treads(x)

    t = np.linspace(0, 10, N)
    x_fft_real = x_fft.real
    x_fft_img = x_fft.imag

    draw_FFT(t, x_fft_real, x_fft_img)


if __name__ == '__main__':
    for i in (256, 1024, 65536):
        time_measure(i, main)