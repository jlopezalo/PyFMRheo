import numpy as np
import pandas as pd
from scipy.signal import detrend
from scipy.special import beta

def hyp2f1_apprx(a, b, c, x):
    # approximation to hypergeometric function 2F1([a, b], c, x)
    # for 0<a<c, x>=0 and b>=0
    # (Butler and Wood, 2000, Annals of Statistics)
    tau=x*(b-a)-c
    yhat=2*a/(np.sqrt(tau**2-4*a*x*(c-b))-tau)

    if b>=0 and (x>=0).any():
        r21=yhat**2/a+(1-yhat)**2/(c-a)-b*x**2./(1-x*yhat)**2*yhat**2/a*(1-yhat)**2/(c-a)
        return c ** (c - 1 / 2) * r21 ** (-1 / 2) * (yhat / a) ** a * ((1 - yhat) / (c - a)) ** (c - a) * (1 - x * yhat) ** (-b)

    else:
        # for b<0 and x<0
        j21=a*(1-yhat)**2+(c-a)*yhat**2-b*x**2.*yhat**2.*(1-yhat)**2./(1-x*yhat)**2
        return np.sqrt(2 * np.pi) * beta(a, c - a) ** (-1) * j21 ** (-1 / 2) * yhat**a * (1 - yhat) ** (c - a) * (1 - x * yhat) ** (-b)

def numdiff(y):
    diffy = np.zeros(len(y))
    idx = np.arange(2, len(y)-3)
    diffy[idx] = (-y[idx+2]+8*y[idx+1]-8*y[idx-1]+y[idx-2])/12
    diffy[:2]=diffy[2]
    diffy[len(diffy)-3:]=diffy[len(diffy)-4]
    return diffy

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

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