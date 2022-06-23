import numpy as np
import matplotlib as plt

poisson_ratio = 0.5
tip_angle = 35.0
v0t = 1e-06
v0r = 1e-06
t0 = 1
E0 = 2500
betaE = 0.19
alpha = 0.19
f0 = 0
slope = 0

tm = 1
tf = 2
ttc = np.linspace(0, tm, 100)
trc = np.linspace(tm, tf, 100)

time = np.r_[ttc, trc]

d0 = 0
dTp = 1
dT = time[1] - time[0]
indentation = np.piecewise(time, [time <= tm, time >= tm], [lambda t: t * v0t, lambda t: (tf-t) * v0r])

idxCt = indentation.argmax()

print(idxCt)


plt.plot(time, indentation)
plt.show()

coeff = 1/np.sqrt(2) * np.tan(np.radians(tip_angle)) * 1/(1-poisson_ratio**2)
Upto=2
delta0 = indentation
delta_Upto_dot=np.diff(delta0**Upto)
delta_Upto_dot = np.append(delta_Upto_dot, delta_Upto_dot[-1])
delta_dot=np.diff(delta0)
delta_dot = np.append(delta_dot, delta_dot[-1])
for i in range(1, idxCt+1)
    Ftc[i: 1]=coeff*E0*sum(delta_Upto_dot(idxCt(1)+(1:jotaint-1)).*flipud(time(idxCt(1)+(1:jotaint-1))).^(-betaE));
end
% toc

% tic
% find t1(t) from int(E(t)delta_dot, t1, t)=0
idx_min_phi0=zeros(1, length(idxCt));
for jotaint=idx_tm+1:idx_tm+length(idxCt)-1
    phi0=flipud(cumsum(flipud(time(jotaint-1:-1:idxCt(2)).^(-betaE).*delta_dot(idxCt(2)+1:jotaint))));
    phi0=phi0(1:length(idxCt)-1);
    [min_phi0 idx_min_phi0(jotaint)]=min(abs(phi0));
    idx_min_phi0(jotaint)=idx_min_phi0(jotaint)-1;
    idxCr0=(jotaint-1):-1:(jotaint-idx_min_phi0(jotaint)+1);
    t10=time(idxCr0);

    Frc(jotaint-idx_tm, 1)=coeff*E0*(trapz((delta_Upto_dot((idxCt(1)+1):idxCt(1)+idx_min_phi0(jotaint)-1).*(t10).^(-betaE))))';
end
FJ=[Ftc+v0t*vdrag; Frc-v0r*vdrag]+F0;
% toc