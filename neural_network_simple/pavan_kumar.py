
import numpy as np
Por = np.array([2,5,3])
Wts = np.array([50,50,50])
Opt = (Por*Wts).sum()
print(Opt)
Act = 850
Error = Act - Opt
print(Error)
corr = Por*(1/35)*(Error)
up_wts = Wts+corr
Opt1 = (Por*up_wts).sum()
while(abs(Act-Opt)>0):
    corr = Por*(1/35)*abs(Act-Opt)
    up_wts = Wts+corr
    Opt = (Por*up_wts).sum()
    if(np.round(Opt-Act) == 0 ):
        print(Opt,up_wts)
