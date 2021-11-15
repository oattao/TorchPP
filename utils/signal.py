import scipy.io as so
import numpy as np
import pywt
import cv2 as cv


def read_mat_file(mat_path):
    data = so.loadmat(mat_path)
    
    data_key = None
    for key in data.keys():
        if key.endswith('DE_time'):
            data_key = key
            break
    
    if data_key is None:
        raise ValueError("No DE_time data in matfile. Check file.")
    
    signal = data[data_key]
    signal = np.squeeze(signal, axis=-1)
    return signal

def add_noise(signal, noise_level):
    # compute signal power
    sig_avg = np.mean(signal)
    sig_avg_db = 10 * np.log10(sig_avg)
    
    # Calculate noise
    noise_avg_db = sig_avg_db - noise_level
    noise_avg = 10 ** (noise_avg_db / 10)
    
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg), len(signal))
    
    # Noise up the original signal
    noisy_signal = signal + noise
    return noisy_signal

def standardize_image(image):
    image = image / image.max() * 255
    # image = image.astype(np.uint8)
    return image

def signal_to_gray_image(signal):
    img_size = int(np.sqrt(len(signal)))
    assert img_size**2 == len(signal), "Length of signal should be a square number"
    
    image = np.reshape(signal, (img_size, img_size))
    image = standardize_image(image)
    return image

def signal_to_scalorgram_image(signal):
    dsize = 224
    widths = np.arange(1, 64)
    cwtmatr, freqs = pywt.cwt(signal, widths, 'morl')
    image = cv.resize(cwtmatr, (dsize, dsize))
    image = standardize_image(image)
    return image