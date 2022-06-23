import numpy as np
import pandas as pd
from scipy.signal import detrend

def numdiff(y):
    diffy = np.zeros(len(y))
    idx = np.arange(2, len(y)-3)
    diffy[idx] = (-y[idx+2]+8*y[idx+1]-8*y[idx-1]+y[idx-2])/12
    diffy[:2]=diffy[2]
    diffy[len(diffy)-3:]=diffy[len(diffy)-4]
    return diffy

def smoothM(d, parS):
    y = d
    DL = len(d)-1
    for ij in range(len(d)-1):
        if np.isnan(y[ij]):
            k = 0
            while np.isnan(y[ij]) and ij+k < DL:
                k += 1
                y[ij] = y[ij+k]
    if parS > 1:
        y[1] = (d[1] + d[2] + d[3])/3
        y[-2] = (d[DL-2] + d[DL-1] + d[DL])/3
    if parS in [2, 3]: #for 2 and 3
        for ij in range(2, DL-2):
            y[ij] = (d[ij-1] + d[ij] + d[ij+1])/3
    if parS >= 4:  # :for 4 and 5 and any more
        for n in range(2, DL-2):
            y[n] = (d[n-2] + d[n-1] + d[n] + d[n+1] + d[n+2])/5
    return y

def detrend_rolling_average(
    seg_freq, seg_in_signal, seg_out_signal, seg_time, in_signal_label, out_signal_label, messages):
    """
    Detrend signal substracting rolling average.
    """
    # Compute the period in seconds from the frequency in Hz
    period = 1 / seg_freq

    # Get the number of points for each signal.
    # Both signals should have the same amount of points.
    n_in = len(seg_in_signal)
    n_out = len(seg_out_signal)

    if n_in == n_out:
        n = n_in
    
    else:
        messages.append(f"[!] {in_signal_label} and {out_signal_label} signals have different lenght!")
        n = np.min(n_in, n_out)

    seg_duration = np.amax(seg_time)
    sampling_rate = n / seg_duration
    points_per_period = int(np.round(period * sampling_rate))

    # Compute rolling forward averages
    rolling_average_in_forw = pd.Series(seg_in_signal).rolling(window=points_per_period, center=True).mean()
    rolling_average_out_forw = pd.Series(seg_out_signal).rolling(window=points_per_period, center=True).mean()

    # Substract rolling forwards average to in signal and out signal
    ntra_in = seg_in_signal - rolling_average_in_forw
    ntra_out = seg_out_signal - rolling_average_out_forw

    # Compute rolling backward averages
    rolling_average_in_bakw = pd.Series(ntra_in[::-1]).rolling(window=points_per_period, center=True).mean()
    rolling_average_out_bakw = pd.Series(ntra_out[::-1]).rolling(window=points_per_period, center=True).mean()

    # Substract rolling backward average to in signal and out signal
    ntra_in = ntra_in - rolling_average_in_bakw
    ntra_out = ntra_out - rolling_average_out_bakw

    ntra_time = seg_time[np.isfinite(ntra_in)]

    # Drop NaN values
    ntra_in = ntra_in.dropna().values
    ntra_out = ntra_out.dropna().values

    if not ntra_in.size or not ntra_out.size:
      messages.append(f"[!] Failed to detrend signal with frequency {seg_freq}")
      ntra_out = detrend(seg_out_signal) # Decide if we want to apply scipy detrend or do nothing!
      ntra_in = detrend(seg_in_signal)
      ntra_time = seg_time

    return ntra_in, ntra_out, ntra_time

def TransferFunction(input_signal, output_signal, fs, frequency=None, nfft=None, freq_tol=0.0001):
    # Define nfft
    if not nfft:
        nfft = len(output_signal)
    # Compute deltat from sampling frequency
    deltat = 1/fs
    # Compute frequency vector
    W = fftfreq(nfft, d=deltat)
    # Compute fft of both signals
    input_signal_hat = fft(input_signal, nfft)
    output_signal_hat = fft(output_signal, nfft)
    # Compute transfer function
    G = output_signal_hat / input_signal_hat
    # Compute coherence
    coherence_params = {"fs": fs, "nperseg":nfft, "noverlap":0, "nfft":nfft, "detrend":False}
    _, gamma2 = coherence(input_signal_hat, output_signal_hat, **coherence_params)
    if frequency:
        # Compute index where to find the frequency
        idx = frequency / (1 / (deltat * nfft))
        idx = int(np.round(idx))
        # Check if the idx is at the right frequency
        if not abs(frequency - W[idx]) <= freq_tol:
            print(f"The frequency found at index {W[idx]} does not match with the frequency applied {frequency}")
        
        return W[idx], G[idx], gamma2[idx], input_signal_hat[idx], output_signal_hat[idx]
    
    else:
        return W, G, gamma2, input_signal_hat, output_signal_hat