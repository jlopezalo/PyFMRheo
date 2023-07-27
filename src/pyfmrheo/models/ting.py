from json import detect_encoding
import numpy as np
from scipy.special import gamma
from lmfit import Model, Parameters
from .geom_coeffs import get_coeff
from ..utils.signal_processing import numdiff, smooth, hyp2f1_apprx

class TingModel:
    def __init__(self, ind_geom, tip_param, modelFt) -> None:
        # Tip geomtry params
        self.ind_geom = ind_geom         # No units
        self.tip_parameter = tip_param   # If radius units is meters, If half angle units is degrees
        self.modelFt = modelFt
        # Compiutation params
        self.fit_hline_flag = False
        self.apply_bec_flag = False
        self.bec_model = None
        # Model params #####################
        self.n_params = None
        # Scaling time
        self.t0 = 0
        # Apparent Young's Modulus
        self.E0 = 1000
        self.E0_init = 1000
        self.E0_min = 0
        self.E0_max = np.inf
        # Time of contact
        self.tc = 0
        self.tc_init = 0
        self.tc_max = 0
        self.tc_min = 0
        # Fluidity exponent
        self.betaE = 0.2
        self.betaE_init = 0.2
        self.betaE_min = 0.01
        self.betaE_max = 1
        # Contact force
        self.F0 = 0
        self.F0_init = 0
        self.F0_min = -np.inf
        self.F0_max = np.inf
        # Poisson ratio
        self.poisson_ratio = 0.5
        # Viscous drag factor
        self.vdrag = 0
        # v0t
        self.v0t = None
        # v0r
        self.v0r = None
        # Smooth window
        self.smooth_w = None
        # Moximum indentation time
        self.idx_tm = None
    
    def build_params(self):
        params = Parameters()
        params.add('E0', value=self.E0_init, min=self.E0_min, max=self.E0_max)
        params.add('tc', value=self.tc_init, min=self.tc_min, max=self.tc_max)
        params.add('betaE', value=self.betaE_init, min=self.betaE_min, max=self.betaE_max)
        params.add('F0', value=self.F0_init, min=self.F0_min, max=self.F0_max)
        return params

    def SolveAnalytical(self, ttc, trc, t1, model_probe, geom_coeff, v0t, v0r, v0, E0, betaE, t0, F0, vdrag):
        # TO DO: ADD REFERENCE!!!
        # Paraboloidal geometry
        if model_probe == 'paraboloid':
            Cp=1/geom_coeff
            Ftp=3/2*v0t**(3/2)*E0*t0**betaE*np.sqrt(np.pi)*np.array(gamma(1-betaE), dtype=float)/(Cp*2*np.array(gamma(5/2-betaE), dtype=float))*ttc**(3/2-betaE)
            if np.abs(v0r-v0t)/v0t<0.01:
                Frp=3/Cp*E0*v0**(3/2)*t0**betaE/(3+4*(betaE-2)*betaE)*t1**(-1/2)*(trc-t1)**(1-betaE)*\
                    (-trc+(2*betaE-1)*t1+trc*hyp2f1_apprx(1, 1/2-betaE, 1/2, t1/trc))
            else:
                Frp=3/Cp*E0*v0t**(3/2)*t0**betaE/(3+4*(betaE-2)*betaE)*t1**(-1/2)*(trc-t1)**(1-betaE)*\
                    (-trc+(2*betaE-1)*t1+trc*hyp2f1_apprx(1, 1/2-betaE, 1/2, t1/trc))
            return np.r_[Ftp, Frp]
        # Conical/Pyramidal geometry
        elif model_probe in ('cone', 'pyramid'):
            Cc=1/geom_coeff
            if np.abs(v0r-v0t)/v0t<0.01:
                Ftc=2*v0**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
                Frc=-2*v0**2.*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                    trc**(1-betaE)*(trc))
            else:
                Ftc=2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*ttc**(2-betaE)
                Frc=-2*v0t**2*E0*t0**betaE/Cc/(2-3*betaE+betaE**2)*((trc-t1)**(1-betaE)*(trc+(1-betaE)*t1)-\
                    trc**(1-betaE)*(trc))
            return np.r_[Ftc, Frc]
    
    def SolveNumerical(self, delta, time_, geom_coeff, geom_exp, v0t, v0r, E0, betaE, F0, vdrag, smooth_w, idx_tm, idxCt, idxCr):
        delta0 = delta - delta[idxCt[0]]
        delta_Uto_dot = np.zeros(len(delta0))
        A = smooth(np.r_[numdiff(delta0[idxCt]**geom_exp), numdiff(delta0[idxCr[0]:]**geom_exp)], smooth_w)
        if len(A) < len(delta_Uto_dot[idxCt[0]:]):
            A = np.append(A, A[-1])
        delta_Uto_dot[idxCt[0]:] = A
        delta_dot = np.zeros(len(delta0))
        B = smooth(np.r_[numdiff(delta0[idxCt]), numdiff(delta0[idxCr[0]:])], smooth_w)
        if len(B) < len(delta_Uto_dot[idxCt[0]:]):
            B = np.append(B, B[-1])
        delta_dot[idxCt[0]:] = B
        Ftc = np.zeros(len(idxCt))
        for i in range(len(idxCt)):
            idx = idxCt[0] + np.arange(1, i)
            Ftc[i] = geom_coeff * E0 * np.sum(delta_Uto_dot[idx]*np.flipud(time_[idx])**(-betaE))
        idx_min_phi0 = np.zeros(len(idxCt))
        Frc = np.zeros(len(idxCt))
        for j in range(idx_tm+1, idx_tm+len(idxCt)):
            phi0 = np.flipud(np.cumsum(np.flipud(time_[j-1:idxCt[1]-1:-1]**(-betaE)*delta_dot[idxCt[1]+1:j+1]), axis=0))
            phi0 = phi0[:len(idxCt)]
            idx_min_phi0 = np.argmin(np.abs(phi0))
            idxCr0 = np.arange(j+1, j-idx_min_phi0+1, -1)
            t10 = time_[idxCr0]
            idx = np.arange(idxCt[0]+1, idxCt[0]+idx_min_phi0+1)
            Frc[j-idx_tm-1] = geom_coeff * E0 * np.trapz(delta_Uto_dot[idx]*t10**(-betaE))
        return np.r_[Ftc, Frc]
    
    def model(
        self, time, E0, tc, betaE, F0, t0, F, delta, modelFt, vdrag,
        idx_tm=None, smooth_w=None, v0t=None, v0r=None
        ):
        # Get indenter shape coefficient and exponent
        geom_coeff, geom_exp = get_coeff(self.ind_geom, self.tip_parameter, self.poisson_ratio)
        # Shift time using t at contact.
        time=time-tc
        # print(f'time --> {time}')
        # print(f'tc --> {tc}')
        # Compute deltat.
        deltat=time[1]-time[0]
        # print(f'idx_tm --> {idx_tm}')
        # If no t max index is given search the index of F max.
        if idx_tm is None:
            idx_tm = np.argmax(F)
        # Get t max value.
        tm = time[idx_tm]
        # print(f'tm --> {tm}')
        # Determine non contact trace region.
        idxNCt=np.where(time<0)[0]
        # Determine contact trace region
        idxCt=np.where(time>=0)[0]
        # Get indices corresponding to contact trace region.
        # Including t max.
        idxCt = np.arange(idxCt[0], idx_tm + 1)
        # Determine contact time trace.
        ttc=time[idxCt]
        if v0t is None:
            # Define range to compute trace speed.
            # Including t max.
            range_v0t=np.arange((idx_tm-int(len(ttc)*3/4)), idx_tm)
            # Fit 1 degree polynomial (x0 + m) to trace and retrace for determining
            # the corresponding speeds (x0)
            v0t = np.polyfit(time[range_v0t], delta[range_v0t], 1)[0]
            self.v0t = v0t
        if v0r is None:
            # Define range to compute retrace speed.
            # Excluding t max.
            range_v0r=np.arange(idx_tm+2, (idx_tm+1+int(len(ttc)*3/4)))
            # Fit 1 degree polynomial (x0 + m) to trace and retrace for determining
            # the corresponding speeds (x0) 
            v0r = -1 * np.polyfit(time[range_v0r], delta[range_v0r], 1)[0]
            self.v0t = v0r
        # Compute mean speed.
        v0=(v0r+v0t)/2
        # Compute retrace contact time.
        # TO DO: ADD REFERENCE TO ARTICLE!!!!
        tcr=(1+v0r/v0t)**(1/(1-betaE))/((1+v0r/v0t)**(1/(1-betaE))-1)*tm
        # If the retrace contact time is smaller than t max,
        # define the end of the contact retrace region as 3 times t max.
        if not tcr<tm:
            idxCr=np.where((time>tm) & (time<=tcr))[0]
        else:
            idxCr=np.where((time>tm) & (time<=3*tm))[0]
        # print(idxCr)
        # Define in contact retrace region.
        trc=time[idxCr]
        # Compute t1
        # TO DO: ADD REFERENCE TO ARTICLE!!!!
        t1=trc-(1+v0r/v0t)**(1/(1-betaE))*(trc-tm)
        # Select only the values larger than 0 of t1.
        t1=t1[t1>0]
        # Select the region of retrace time where t1 is larger than 0.
        trc=trc[t1>0]
        # Select the retrace contact indices corresponding to the retrace
        # time region where t1 is larger than 0. 
        idxCr=idxCr[:len(trc)]
        # print(idxCr)
        # Assign the value of F0 to the non contact region.
        FtNC=F0*np.ones(idxNCt.size)
        # Compute Force according to the selected mode:
        if modelFt == 'analytical':
            FJ = self.SolveAnalytical(
                ttc, trc, t1, self.ind_geom, geom_coeff, v0t, v0r, v0, E0, betaE, t0, F0, vdrag
            )
        elif modelFt == 'numerical':
            FJ = self.SolveNumerical(
                delta, time, geom_coeff, geom_exp, v0t, v0r, E0, betaE, F0, vdrag, smooth_w, idx_tm, idxCt, idxCr
            )
        else:
            print(f'The modelFt {modelFt} is not supported. Current valid modelFt: analytical, numerical.')
        # Determine non contact retrace region.
        idxNCr=np.arange((len(FJ)+len(FtNC)+1),len(delta)+1)
        # Assign the value of F0 to the non contact region.
        FrNC=F0*np.ones(idxNCr.size)
        # Concatenate non contact regions to the contact region. And return.
        return np.r_[FtNC, FJ+F0, FrNC]+smooth(numdiff(delta)*vdrag/numdiff(time), 21)
    
    def fit(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        # Define fixed params
        self.t0 = t0
        self.idx_tm = idx_tm
        self.smooth_w = smooth_w
        self.v0t = v0t
        self.v0r = v0r
        
        # Define fixed params
        fixed_params = {
            't0': self.t0, 'F': F, 'delta': delta,
            'modelFt': self.modelFt, 'vdrag': self.vdrag, 'smooth_w': self.smooth_w,
            'idx_tm': self.idx_tm, 'v0t': self.v0t, 'v0r': self.v0r
        }
        
        # Prepare model for fit using fixed params
        tingmodel =\
            lambda time, E0, tc, betaE, F0: self.model(time, E0, tc, betaE, F0, **fixed_params)
        
        tingmodelfit = Model(tingmodel)
        
        # Define free params
        params = self.build_params()
        
        # Do fit
        self.n_params = len(tingmodelfit.param_names)

        result_ting = tingmodelfit.fit(F, params, time=time)
        
        # Assign fit results to model params
        self.E0 = result_ting.best_values['E0']
        self.tc = result_ting.best_values['tc']
        self.betaE = result_ting.best_values['betaE']
        self.F0 = result_ting.best_values['F0']

        # Compute metrics
        modelPredictions = self.eval(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)

        absError = modelPredictions - F

        self.MAE = np.mean(absError) # mean absolute error
        self.SE = np.square(absError) # squared errors
        self.MSE = np.mean(self.SE) # mean squared errors
        self.RMSE = np.sqrt(self.MSE) # Root Mean Squared Error, RMSE
        self.Rsquared = 1.0 - (np.var(absError) / np.var(F))

        # Get goodness of fit params
        self.chisq = self.get_chisq(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)
        self.redchi = self.get_red_chisq(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)

    def eval(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        return self.model(
            time, self.E0, self.tc, self.betaE, self.F0, t0, F, delta,
            self.modelFt, self.vdrag, idx_tm, smooth_w, v0t, v0r)

    def get_residuals(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        return F - self.eval(time, F, delta, t0,idx_tm, smooth_w, v0t, v0r)

    def get_chisq(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        a = (self.get_residuals(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r)**2/F)
        return np.sum(a[np.isfinite(a)])
    
    def get_red_chisq(self, time, F, delta, t0, idx_tm=None, smooth_w=None, v0t=None, v0r=None):
        return self.get_chisq(time, F, delta, t0, idx_tm, smooth_w, v0t, v0r) / self.n_params
    
    def fit_report(self):
        print(f"""
        # Fit parameters
        Indenter shape: {self.ind_geom}\n
        Tip paraneter: {self.tip_parameter}\n
        Model Format: {self.modelFt}\n
        Viscous Drag: {self.vdrag}\n
        Smooth Window: {self.smooth_w}\n
        t0: {self.t0}\n
        Maximum Indentation Time: {self.idx_tm}\n
        Number of free parameters: {self.n_params}\n
        E0: {self.E0}\n
        tc: {self.tc}\n
        betaE: {self.betaE}\n
        F0: {self.F0}\n
        # Fit metrics
        MAE: {self.MAE}\n
        MSE: {self.MSE}\n
        RMSE: {self.RMSE}\n
        Rsq: {self.Rsquared}\n
        Chisq: {self.chisq}\n
        RedChisq: {self.redchi}\n
        """
        )
