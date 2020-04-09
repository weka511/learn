import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# S    Susceptible
# E    Exposed
# P    pre-symptomatic
# I0   Infectious untested
# I1   Infectious tested
# R0   Recovered untested
# R1   Recovered tested
def model(t,y,
          N       = 5000000,         # Population size
          c       = 0.1,             # testing rate for symptomatic cases, per diem
          alpha   = 0.25,            # E to P transition rate, per diem
          beta    = 0.2463,          # transmission coefficient, per diem  
          gamma   = 0.1,             # I to R tranition, per diem
          delta   = 1,               # P to I, per diem
          epsilon = 0.15,            # relative infectiousness
          CFR1    = 2.0/100,         # case fatality rate with cases exceedinging ICU max
          CFR0    = 1.0/100,         # case fatality rate for cases under ICU max
          nICU    = 300,             # number of ICU beds
          pICU    = 1.25/100      # proportion of cases requiring ICU
          ):
    
    S,E,P,I0,I1,R0,R1 = y
    I = I0+I1
    CFR = CFR1 - (nICU/(N*I* pICU)) * (CFR1 - CFR0) if I>0 else I1
    return [
        -beta*S*(epsilon*P+I0+I1),
        +beta*S*(epsilon*P+I0+I1) - alpha*E,
        alpha*E - delta * P,
        delta*P -(gamma + c) *I0,
        c*I0 - gamma*I1,
        gamma*(1-CFR)*I0,
        gamma*(1-CFR)*I1
    ]
    
    

def get_death(j,y):
    return 1-sum([y[i][j] for i in range(0,7)])

def scale(ys,N=1):
    return [N*y for y in ys]

def aggregate(ys):
    y0s,y1s = ys
    return [ y0s[i] + y1s[i] for i in range(len(y0s)) ]


if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Model COVID19 evolution')
    parser.add_argument('--R0', nargs='+',type=float,default=2.5)
    parser.add_argument('--seed',         type=int,default=20,help='Number of exposed people whom we inject into population')
    parser.add_argument('--N',            type=int,default=5000000,help='Population size')
    parser.add_argument('--nICU',         type=int,default=300,help='Number of ICU beds')
    parser.add_argument('--end',          type=int,default=365,help='Number of days to be simulated')
    parser.add_argument('--c',            type=float,default=0.1,help='testing rate for symptomatic cases, per diem')
    parser.add_argument('--alpha',        type=float,default=0.25,help='E to P transition rate, per diem')
    parser.add_argument('--gamma',        type=float,default=0.1,help='I to R tranition, per diem')
    parser.add_argument('--delta',        type=float,default=1.0,help='P to I, per diem')
    parser.add_argument('--epsilon',      type=float,default=0.15,help='relative infectiousness')
    parser.add_argument('--CFR1',         type=float,default=2.0/100,help='case fatality rate with cases exceeding ICU max')
    parser.add_argument('--CFR0',         type=float,default=1.0/100,help='case fatality rate for cases under ICU max')
    parser.add_argument('--pICU',         type=float,default=1.25/100,help='proportion of cases requiring ICU')
    args = parser.parse_args()
    
    for R0 in args.R0 if isinstance(args.R0, list) else [args.R0]:
        beta    = R0/(args.epsilon/args.delta + 1/args.gamma)     # transmission coefficient, per diem   
        
        sol    = solve_ivp(model, 
                           (0,args.end),
                           [1-args.seed/args.N,args.seed/args.N,0,0,0,0,0],
                           args=(args.N, args.c, args.alpha, beta, args.gamma, args.delta, 
                                 args.epsilon, args.CFR1, args.CFR0, args.nICU, args.pICU))
        
        deaths         = scale([get_death(j,sol.y) for j in range(len(sol.y[1]))],N=args.N)
        total_infected = scale(aggregate((sol.y[ 3],sol.y[4])),N=args.N)
        
        plt.figure(figsize=(20,6))
        plt.plot(sol.t, scale(sol.y[1]),  color='g',                 label='E (Exposed)')
        plt.plot(sol.t, scale(sol.y[2]),  color='r',                 label='P (Presymptomatic)')
        plt.plot(sol.t, scale(sol.y[ 3]), color='c', linestyle=':',  label='I0 (Infectious, untested')
        plt.plot(sol.t, scale(sol.y[ 4]), color='c', linestyle='--', label='I1 (Infectious, untested)')
        plt.plot(sol.t, total_infected,   color='c',                 label='I (Infectious)')
        plt.plot(sol.t, scale(sol.y[ 5]), color='m', linestyle=':',  label='R0 (Recovered, untested)')
        plt.plot(sol.t, scale(sol.y[ 6]), color='m', linestyle='--', label='R1 (Recovered, tested)')
        plt.plot(sol.t, deaths,           color='k',                 label='Deaths')
        plt.legend(loc='best')
        plt.xlabel('t')
        plt.title('Progression of COVID-19: R0 = {}'.format(R0))
        plt.grid()
    
    plt.show()