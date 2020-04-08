import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N       = 5000000    # Population size
c       = 0.1        # testing rate for symptomatic cases, per diem
alpha   = 0.25       # E to P transition rate, per diem
beta    = 0.2463     # transmission coefficient, per diem  
gamma   = 0.1        # I to R tranition, per diem
delta   = 1          # P to I, per diem
epsilon = 0.15       # relative infectiousness
CFR1    = 2.0/100    # case fatality rate with cases under ICU max
CFR0    = 1.0/100    # case fatality rate for cases exceeding ICU max
nICU    = 300        # number of ICU beds
pICU    = 1.25/100   # proportion of cases requiring ICU

# S    Susceptible
# E    Exposed
# P    pre-symptomatic
# I0   Infectious untested
# I1   Infectious tested
# R0   Recovered untested
# R1   Recovered tested
def model(t,y,
          N       = 5000000,    # Population size
          c       = 0.1,        # testing rate for symptomatic cases, per diem
          alpha   = 0.25,       # E to P transition rate, per diem
          beta    = 0.2463,     # transmission coefficient, per diem  
          gamma   = 0.1,        # I to R tranition, per diem
          delta   = 1,          # P to I, per diem
          epsilon = 0.15,       # relative infectiousness
          CFR1    = 2.0/100,       # case fatality rate with cases under ICU max
          CFR0    = 1.0/100,       # case fatality rate for cases exceeding ICU max
          nICU    = 300,        # number of ICU beds
          pICU    = 1.25/100):    # proportion of cases requiring ICU
    
    S,E,P,I0,I1,R0,R1 = y
    I = I0+I1
    CFR = CFR1 - (nICU/(N*I* pICU)) * (CFR1 - CFR0) if I>0 else I1
    dy = [
        -beta*S*(epsilon*P+I0+I1),
        +beta*S*(epsilon*P+I0+I1) - alpha*E,
        alpha*E - delta * P,
        delta*P -(gamma + c) *I0,
        c*I0 - gamma*I1,
        gamma*(1-CFR)*I0,
        gamma*(1-CFR)*I1
    ]

    #D = 1 - S - E -P -I0 -I1 -R0 -R1,
    
    return dy
    

def get_death(j,y):
    return 1-sum([y[i][j] for i in range(0,7)])


seed   = 20
y0     = [1-seed/N,seed/N,0,0,0,0,0]
sol    = solve_ivp(model, (0,720), y0, args=(N, c, alpha, beta, gamma, delta, epsilon, CFR1, CFR0, nICU, pICU))
deaths = [get_death(j,sol.y) for j in range(len(sol.y[1]))]
plt.figure(figsize=(20,6))
#plt.plot(sol.t, sol.y[0], color='b',                 label='S (Susceptible)')
plt.plot(sol.t, sol.y[1], color='g',                 label='E (Exposed)')
plt.plot(sol.t, sol.y[2], color='r',                 label='P (Presymptomatic)')
plt.plot(sol.t, sol.y[ 3], color='c', linestyle=':',  label='I0 (Infectious, untested')
plt.plot(sol.t, sol.y[ 4], color='c', linestyle='--', label='I1 (Infectious, untested)')
plt.plot(sol.t, sol.y[ 5], color='m', linestyle=':',  label='R0 (Recovered, untested)')
plt.plot(sol.t, sol.y[ 6], color='m', linestyle='--', label='R1 (Recovered, tested)')
plt.plot(sol.t, deaths, color='k', label='Deaths')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()