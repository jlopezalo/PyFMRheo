import sys
sys.path.insert(0, '../../src')
from matplotlib.mlab import psd, csd, detrend_linear, window_hanning
import matplotlib.pyplot as plt
from scipy import signal, fft
import pandas as pd
import numpy as np
from analysis.models.rheology import ComputeComplexModulus

# Hard coded values
frequencies = [0.1000, 0.3500, 1.1500, 3.5500, 11.4500] # Hz
wc = 826.9249 # nm
freqosc = 0.05 # Hz
bcoef = 1.55 * 1e-6 # From BCN's code

# Results from BCN code
GRealBCN = [2.6215, 2.8147, 3.1955, 3.7234, 4.1157]
GImagBCN = [0.2325, 0.4586, 0.7504, 1.1198, 2.1644]    
gamma2BCN = [0.9847, 0.9992, 0.9997, 0.9999, 0.9998]
    
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return signal.lfilter(b, a, data)

def model_paraboloid(G, wc, freq):
    tip_radius = 5000 * 1e-9
    wc = wc * 1e-9
    poisson_ratio = 0.5
    div = 4 * np.sqrt(tip_radius * wc)
    coeff = (1.0 - poisson_ratio) / div
    G_corr = coeff * 2 * np.pi * bcoef * -1j * freq
    G_complex =  G * coeff - G_corr
    return G_complex.real * 1e-3, G_complex.imag * 1e-3

def readBCNData(file_path):
    header = pd.read_csv(file_path, sep=' =', names=["Param", "Value"], nrows=19, header=None, encoding='latin-1', engine="python")
    header = header.set_index('Param').T.to_dict('list')
    header["wc"] = wc
    k = float(header["Spring constant (nN/nm)"][0])


    data = pd.read_csv(file_path, sep='\t', names=["FN", "SG", "SUM", "FL"], skiprows=21, header=None, encoding='latin-1', engine="c")
    
    force = data["FN"]                  # nN
    deflection = data["FN"] / k         # nm
    zheight = data["SG"] * 1e3          # nm
    indentation = zheight - deflection  # nm

    return force, indentation, header

def computeWhole(force, indentation, params, freq):
    wc = params["wc"]
    fs = int(params["fs (Hz)"][0])
    nfft = int(params["Number of Points per cicle"][0])
    order = 4
    cutoff = freqosc / (fs/2)

    indentation = detrend_linear(indentation)
    force = detrend_linear(force)

    indentation = indentation - butter_lowpass_filter(indentation, cutoff, fs, order)
    force = force - butter_lowpass_filter(force, cutoff, fs, order)

    # x --> indentation
    # y --> force
    # Pxy, freqs_xy = csd(indentation, force, window=window_hanning, NFFT=nfft, Fs=fs)
    # Pxx, freqs_xx = psd(indentation, window=window_hanning, NFFT=nfft, Fs=fs)

    fxy, Pxyf = signal.csd(indentation, force, fs, nperseg=nfft, noverlap=0, nfft=nfft, return_onesided=True, detrend=False)
    fxx, Pxxf = signal.welch(indentation, fs, nperseg=nfft, noverlap=0, nfft=nfft, return_onesided=True, detrend=False)

    if fxy.all() != fxx.all():
        print("Frequencies computed by the csd and psd are different!")
        return

    H1 = Pxyf / Pxxf

    # H1_scipy = Pxy_scipy / Pxx_scipy

    if len(fxx) != len(H1):
        print("Frequency array and H1 do not have the same size!")
        return

    idx = (np.abs(fxx - freq)).argmin()

    # print(freqs_xx[idx])

    G = H1[idx]

    G_storage, G_loss = model_paraboloid(G, wc, freq)

    return G_storage, G_loss


def computeBlocs(force, indentation, params, freq):

    # print(params.keys())

    nfft = len(force)
    wc = params["wc"]
    fs = int(params["fs (Hz)"][0])

    indentation = detrend_linear(indentation)
    force = detrend_linear(force)

    # x --> indentation
    # y --> force
    Pxy, freqs_xy = csd(indentation, force, window=None, NFFT=nfft, Fs=fs)
    Pxx, freqs_xx = psd(indentation, window=None, NFFT=nfft, Fs=fs)

    # fxy, Pxyf = signal.csd(indentation, force, fs, nperseg=nfft, noverlap=0, nfft=nfft, return_onesided=True, detrend=False)
    # fxx, Pxxf = signal.welch(indentation, fs, nperseg=nfft, noverlap=0, nfft=nfft, return_onesided=True, detrend=False)

    if freqs_xx.all() != freqs_xy.all():
        print("Frequencies computed by the csd and psd are different!")
        return

    H1 = Pxy / Pxx
    # H2 = Pxyf / Pxxf

    if len(freqs_xx) != len(H1):
        print("Frequency array and H1 do not have the same size!")
        return

    idx = (np.abs(freqs_xx - freq)).argmin()

    # print(freqs_xx[idx])

    G = H1[idx]

    print(f"G Pxy/Pxx: {G}")

    G_storage, G_loss = model_paraboloid(G, wc, freq)

    return G_storage, G_loss

    

if __name__ == "__main__":
    file_path = "./BCNData/mf_5kpa_constsignal.001.txt"
    force, indentation, params = readBCNData(file_path)
    ppc = int(params["Number of Points per cicle"][0])
    ncycles = int(params["Ncycles"][0])
    fs = int(params["fs (Hz)"][0])
    order = 4
    cutoff = freqosc / (fs/2)
    r = 5000 * 1e-9
    force_1, indentation_1 = force[:ppc], indentation[:ppc]

    force_blocs = np.split(force.values, ncycles)
    indentation_blocs = np.split(indentation.values, ncycles)

    G_storage_means_txy = []
    G_loss_means_txy = []
    G_storage_means_fft = []
    G_loss_means_fft = []
    G_storage_whole_all = []
    G_loss_whole_all = []
    
    for i, freq in enumerate(frequencies):
        # print(force_1 == force_blocs[0])
        G_storage_all_txy = []
        G_loss_all_txy  = []
        G_storage_all_fft = []
        G_loss_all_fft = []
        G_storage_whole, G_loss_whole = computeWhole(force, indentation, params, freq)
        G_storage_whole_all.append(G_storage_whole)
        G_loss_whole_all.append(G_loss_whole)
        for j in range(ncycles):
            indentation_process = indentation_blocs[j] - butter_lowpass_filter(indentation_blocs[j], cutoff, fs, order)
            force_process = force_blocs[j] - butter_lowpass_filter(force_blocs[j], cutoff, fs, order)
            G_storage, G_loss = computeBlocs(force_process, indentation_process, params, freq)
            G_storage_fft, G_loss_fft, gamma2 = ComputeComplexModulus(force_process, indentation_process, fs, freq, "paraboloid", r, wc*1e-9, bcoef=bcoef)
            print(gamma2)
            G_storage_all_txy.append(G_storage)
            G_loss_all_txy.append(G_loss)
            G_storage_all_fft.append(G_storage_fft * 1e-3)
            G_loss_all_fft.append(G_loss_fft * 1e-3)

        G_storage_all = np.asarray(G_storage_all_txy)
        G_loss_all = np.asarray(G_loss_all_txy)

        G_storage_mean = G_storage_all.mean()
        G_loss_mean = G_loss_all.mean()

        G_storage_all_fft = np.asarray(G_storage_all_fft)
        G_loss_all_fft = np.asarray(G_loss_all_fft)

        G_storage_mean_fft = G_storage_all_fft.mean()
        G_loss_mean_fft = G_loss_all_fft.mean()

        G_storage_means_txy.append(G_storage_mean)
        G_loss_means_txy.append(G_loss_mean)
        G_storage_means_fft.append(G_storage_mean_fft)
        G_loss_means_fft.append(G_loss_mean_fft)

        print(f"Applied frequency: {freq} Hz")                          # Hz
        print(f"G Storage Fragmentwise: {G_storage_mean * 1e3} Pa")     # Pa
        print(f"G Storage Whole: {G_storage_whole * 1e3} Pa")           # Pa
        print(f"G Loss Fragmentwise: {G_loss_mean * 1e3} Pa")           # Pa
        print(f"G Loss Whole: {G_loss_whole * 1e3} Pa")                 # Pa
        print(f"G Storage BCN - G Storage Fragmentwise: {(GRealBCN[i] - G_storage_mean_fft)  * 1e3} Pa")   # Pa
        print(f"G Loss BCN - G Loss Fragmentwise: {(GImagBCN[i] - G_loss_mean_fft) * 1e3} Pa")             # Pa
        print(f"G Storage BCN - G Storage Whole: {(GRealBCN[i] - G_storage_whole)  * 1e3} Pa")         # Pa
        print(f"G Loss BCN - G Loss Whole: {(GImagBCN[i] - G_loss_whole) * 1e3} Pa")                   # Pa
        print("\n")

    plt.plot(frequencies, GRealBCN, label="Matlab Real")
    plt.plot(frequencies, G_storage_means_fft, label="Python Real")
    plt.plot(frequencies, GImagBCN, label="Matlab Imag")
    plt.plot(frequencies, G_loss_means_fft, label="Python Imag")
    plt.xlabel("Frequencies [Hz]")
    plt.ylabel("G', G'' [KPa]")
    plt.xscale("log")
    plt.legend()
    plt.show()