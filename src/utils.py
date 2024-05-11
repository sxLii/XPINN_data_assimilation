import numpy as np
import matplotlib.pyplot as plt
#plt.rcParams['font.family'] = 'Times New Roman'
def add_noise(insignal):
    """
    add noise for smooth datas
    """

    target_snr_db = 25
    # Calculate signal power and convert to dB
    sig_avg = np.mean(insignal)
    sig_avg_db = 10 * np.log10(sig_avg)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_avg), len(insignal))
    # Noise up the original signal
    sig_noise = insignal + noise
    return sig_noise

def time_to_seconds(time_str):
    """
    convert scheme of time form 00:01 to 60s
    """
    hours, minutes = map(int, time_str.split(':'))
    return hours * 3600 + minutes * 60

def interp1(t,ha):
    """
    return interpolation of 'ha' by the time 't'
    ha[:,0] is origin time
    ha[:,1] is the origin data, return the same length as 't'
    """
    return np.interp(t, ha[:, 0], ha[:, 1])