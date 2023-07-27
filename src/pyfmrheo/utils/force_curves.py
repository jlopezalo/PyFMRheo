# Module containing helper methods for AFM force curves
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

def checkIfIncreasing(data):
  i = len(data)
  if i < 2: 
    raise Exception('Need at least 2 points!')
  return data[i - 2] >= data[1] if i > 5 else data[i - 1] >= data[0]

def regulaFalsi(p1, p2, p3, p4):
  return (p1 * p4 - p2 * p3) / (p4 - p3)

def get_poc_regulaFalsi_method(app_height, app_deflection, sigma=0):
  # Reference: https://www.codesansar.com/numerical-methods/regula-falsi-or-false-position-method-algorithm.htm
  ydata_smoothed = gaussian_filter(app_deflection, sigma)
  i = len(app_deflection)
  if i < 2:
    raise Exception('Need at least 2 points to determine PoC!')
  if checkIfIncreasing(app_height): j = i - 1; b = -1
  else:  j = 0; b = 1
  if ydata_smoothed[j] < 0.0:
    raise Exception('First value of deflection is negative!')
  d1 = np.nan
  d2 = app_height[j]
  d3 = ydata_smoothed[j]
  k = j + b
  while 0 <= k < i:
    d4 = app_height[k]
    d5 = ydata_smoothed[k]
    if d5 <= 0.0:
      d1 = regulaFalsi(d4, d2, d5, d3)
      if np.isnan(d1) or not np.isfinite(d1):
        raise Exception('Could not compute PoC!')
      return np.array([d1, 0])
    d2 = d4
    d3 = d5
    k += b

def get_poc_RoV_method(app_height, app_deflection, windowforCP=350*1e-9):
  deltaz=np.abs(app_height.max()-app_height.min())
  zperpt=deltaz/len(app_height)
  win_size=int(windowforCP/2/zperpt)*2
  rov_dfl_1 = pd.Series(app_deflection[win_size+1:])
  rov_dfl_2 = pd.Series(app_deflection[:-win_size])
  rovi = rov_dfl_1.rolling(win_size, center=True, min_periods=1).var(ddof=0)/\
          rov_dfl_2.rolling(win_size, center=True, min_periods=1).var(ddof=0)
  rovi_idx = rovi.idxmax()
  rov_poc_x = app_height[rovi_idx]
  rov_poc_y = app_deflection[rovi_idx]
  return np.array([rov_poc_x, rov_poc_y])

def correct_offset(
  height, deflection, max_offset=1e-6, min_offset=10e-9):
  mask = (height>=min_offset) & (height<=max_offset)
  return deflection - np.nanmedian(deflection[mask])

def correct_tilt(
  height, deflection, max_offset=1e-6, min_offset=10e-9):
  mask = (height>=min_offset) & (height<=max_offset)
  z = np.poly1d(np.polyfit(height[mask], deflection[mask], 1))
  deflection = deflection-z(height)
  return deflection

def correct_viscous_drag(
  ind_approach, force_approach, ind_retract, force_retract, poly_order=2, speed=None):

  # Fit approach
  mask_app = ind_approach < 0
  approach = np.polyfit(
    ind_approach[mask_app], force_approach[mask_app], poly_order
  )
  approach_pol = np.poly1d(approach)

  # Fit retract
  mask_ret = ind_retract < 0
  retract = np.polyfit(
    ind_retract[mask_ret], force_retract[mask_ret], poly_order
  )
  retract_pol = np.poly1d(retract)

  if len(mask_app) <= len(mask_ret): mask = mask_app
  else: mask = mask_ret

  approach_vals = approach_pol(ind_approach[mask])
  retract_vals = retract_pol(ind_approach[mask])

  median_dif = np.median(approach_vals-retract_vals)

  if speed != 0: correction = (median_dif/2) / speed
  else: correction = median_dif/2

  corrected_app_force = force_approach - correction
  corrected_ret_force = force_retract + correction

  return corrected_app_force - corrected_app_force[0], corrected_ret_force - corrected_app_force[0]
