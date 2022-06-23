# Module containing helper methods for AFM force curves

import numpy as np
import matplotlib.pyplot as plt

def get_poc_RoV_method(piezo_height, deflection, win_size=20, plot_process=False, savepath="."):
    """
    Compute point of contact using the ratio of variances method.
    Reference: doi: 10.1038/srep21267

    Arguments:
    piezo_height -- z position of the piezo in m
    deflection -- deflection of the cantilever in m
    savepath -- path so save the plots showing the results
    win_size -- number of points to include in each window
    plot_plocess -- generate plots

    Returns:
    Array containing the values for the PoC [x, y] in m
    """

    # Compute indentation asuming d(0) = 0
    ind = piezo_height - deflection

    # Array for containing output values
    rov_arr = np.zeros(ind.shape)

    n = len(deflection)

    # Compute window from %

    win_size = n * (win_size/100)

    # Compute RoV for each point
    for i in range(n):
      
      # Compute variances
      var_seg1 = np.var(deflection[i + 1 : i + int(np.round(win_size / 2))])
      var_seg2 = np.var(deflection[i - int(np.round(win_size / 2)) : i - 1])
      
      # Compute RoV
      rov = var_seg1 / var_seg2

      # Check if the outcome of the RoV value is NaN or infinite.
      # This is cause by variances begin equal to 0.
      if np.isnan(rov) or np.isinf(rov):
        rov = 0

      # Assign value to the output array
      rov_arr[i] = rov
    
    # Get the index where the maximum value of RoV is located
    index = np.argmax(rov_arr)
    
    # Get the positions at the index
    pocx = ind[index]
    pocy = deflection[index]
    
    # Plot the results if requested
    if plot_process and savepath:
      fig = plt.figure(figsize=(10, 5))
      ax = plt.subplot(111)
      ax.plot(ind, rov_arr)
      ax.axvline(x=pocx, color='k', linestyle='--', label = 'PoC=%5.3f' % (pocx))
      ax.set_title("PoC determination by RoV")
      ax.set_xlabel("height - deflection")
      ax.set_ylabel("RoV")
      fig.savefig(f"{savepath}/poc_rov.png")

    return np.array([pocx, pocy])
  
def correct_viscous_drag(
  ind_approach, force_approach, ind_retract, force_retract, poly_order=2, speed=None):

  # Fit approach
  mask_app = ind_approach < 0
  approach = np.polyfit(ind_approach[mask_app], force_approach[mask_app], poly_order)
  approach_pol = np.poly1d(approach)

  # Fit retract
  mask_ret = ind_retract < 0
  retract = np.polyfit(ind_retract[mask_ret], force_retract[mask_ret], poly_order)
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
