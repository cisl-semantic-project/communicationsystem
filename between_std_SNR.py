import numpy as np

## standard deviation 기준으로 SNR 입력
def std_2_SNR(std):
    Eb_N0 = 1/(2*std**2)
    SNR = 10*np.log10(Eb_N0)
    return SNR

## SNR(dB scale)기준으로 standard deviation 입력
def SNR_2_std(SNR):
    Eb_N0 = 10**(SNR/10)
    std = 1/np.sqrt(2*Eb_N0)
    return std