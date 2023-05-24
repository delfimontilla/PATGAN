import numpy as np

def addnoise(signal, snr, mode = None):
    if mode == None or mode == 'rms':
        signal_rms = np.sqrt(np.mean(signal**2))
    elif mode == 'peak':
        signal_rms = signal.max()        
    else:
        raise Exception(f'Unknown parameter {mode} for optional input mode')

    std_dev = signal_rms/(10**(snr/20))
    noise = np.random.normal(loc = 0, scale = std_dev, size = signal.shape)
    noise_rms = np.sqrt(np.mean(noise**2))
    snr = 20*np.log10(signal_rms/noise_rms)
    signal = signal + noise
    return signal, snr