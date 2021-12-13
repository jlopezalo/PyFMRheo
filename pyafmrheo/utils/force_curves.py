# Module containing helper methods for AFM force curves

import numpy as np
import matplotlib.pyplot as plt

def preprocess_segment(segment, height_channel_key, deflection_sens, y0=None):
    deflection_v = segment.segment_formated_data["vDeflection"]
    if segment.segment_metadata["baseline_measured"]:
        deflection_v = deflection_v - segment.segment_metadata["baseline"]
    elif y0 is not None:
        deflection_v = deflection_v - y0
    deflection = deflection_v * deflection_sens
    height = segment.segment_formated_data[height_channel_key]
    time = segment.segment_formated_data["time"]

    return deflection, height, time

def get_force_vs_indentation_curve(piezo_height, deflection, poc, spring_constant):
    """
    Compute force vs indentation curve from deflection and piezo_height.

    Indentation = piezo_height(m) − deflection(m) − (piezo_height(CP)(m) − deflection(CP)(m))
    Force = Kc(N/m) * deflection(m)

    Reference: doi: 10.1002/jemt.22776

    Arguments:
    piezo_height -- z position of the piezo in m
    deflection -- deflection of the cantilever in m
    poc -- point of contact in m
    spring_constant -- spring constant of the cantilever in N/m

    Returns:
    List containing 2 arrays, where the first array is the indentation in m
    and the second array is the force in N.
    """
    # Set the center position to 0, 0 and get a force curve
    center_force_x = poc[0] - poc[1]
    center_force_y = poc[1] * spring_constant

    # Indentation = piezo_height(m) − deflection(m) − (piezo_height(CP)(m) − deflection(CP)(m))
    # Force = Kc(N/m) * deflection(m)
    return [np.array(piezo_height - deflection - center_force_x),
            np.array(deflection * spring_constant - center_force_y)]


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
    for i in range(0, n):
      
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
