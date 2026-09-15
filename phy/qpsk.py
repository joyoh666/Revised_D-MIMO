import numpy as np

constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)

def qpsk_modulate(data: np.ndarray) -> np.ndarray:
    if len(data) % 2 != 0:
        raise ValueError("Input data length must be even")
    
    symbol = np.array([], dtype=complex)
    for i in range(0, len(data), 2):
        b0 = data[i]
        b1 = data[i+1]
        idx = 2 * b0 + b1
        symbol = np.append(symbol, constellation[idx])

    return symbol

def qpsk_demodulate(data: np.ndarray) -> np.ndarray:
    distances = np.abs(data[:, np.newaxis] - constellation[np.newaxis,:])
    indices = np.argmin(distances, axis=1)
    bits = np.zeros(len(indices) * 2, dtype=int)
    bits[0::2] = (indices >> 1) & 1
    bits[1::2] = indices & 1
    return bits