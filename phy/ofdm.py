import numpy as np

def ofdm_modulate(rg: np.ndarray,
         fft_size: int,
         N_cp: int) -> np.ndarray:

    signal = np.fft.ifft(rg, n=fft_size, axis=0)
    signal = np.concatenate([signal[-N_cp:, :], signal])

    return signal

def ofdm_demodulate(signal: np.ndarray,
                    fft_size: int,
                    N_cp: int) -> np.ndarray:
    signal = signal[N_cp:, :]
    symbol = np.fft.fft(signal, n=fft_size, axis=0)
    return symbol
