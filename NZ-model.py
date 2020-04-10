# NZ-model.py

# Copyright (C) 2020 Greenweaves Software Limited

# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.GA

# You should have received a copy of the GNU General Public License
# along with GNU Emacs.  If not, see <http://www.gnu.org/licenses/>.

# This model is based on Suppression and Mitigation Strategies for Control of COVID-19 in New Zealand
# 25 March 2020, Alex James, Shaun C. Hendy, Michael J. Plank, Nicholas Steyn

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# dy
#
# Computer derivative of state vector
#
# Parameters:
#    t
#    y       State vector
#       S      0: Susceptible
#       E      1: Exposed
#       P      2: pre-symptomatic
#       I0     3: Infectious untested
#       I1     4: Infectious tested
#       R0     5: Recovered untested
#       R1     6: Recovered tested
#    N       Population size
#    c       Testing rate for symptomatic cases, per diem
#    alpha   E to P transition rate, per diem
#    beta    transmission coefficient, per diem  
#    gamma   I to R tranition, per diem
#    delta   P to I, per diem
#    epsilon relative infectiousness
#    CFR1    case fatality rate with cases exceedinging ICU max
#    CFR0    case fatality rate for cases under ICU max
#    nICU    number of ICU beds
#    pICU    proportion of cases requiring ICU

def dy(t,y,
       N       = 5000000,  
       c       = 0.1, 
       alpha   = 0.25,   
       beta    = 0.2463,   
       gamma   = 0.1,  
       delta   = 1,   
       epsilon = 0.15, 
       CFR1    = 2.0/100,  
       CFR0    = 1.0/100,  
       nICU    = 300,   
       pICU    = 1.25/100 ):
    
    S,E,P,I0,I1,R0,R1 = y
    I                 = I0 + I1
    CFR               = CFR1 - (nICU/(N*I* pICU)) * (CFR1 - CFR0) if I>0 else 0
    new_exposed       = beta * S * (epsilon*P + I)
    return [
        -new_exposed,
        +new_exposed - alpha*E,
        alpha*E - delta * P,
        delta*P -(gamma + c) *I0,
        c*I0 - gamma*I1,
        gamma*(1-CFR)*I0,
        gamma*(1-CFR)*I1
    ]

def scale(ys,N=1):
    return [N*y for y in ys]

def aggregate(ys):
    y0s,y1s = ys
    return [ y0s[i] + y1s[i] for i in range(len(y0s)) ]

def plot_detail(t=[],y=[],N=1,Rc=1):  
    plt.plot(t, scale(y[1],N=N),                   color='g',                 label='E (Exposed)')
    plt.plot(t, scale(y[2],N=N),                   color='r',                 label='P (Presymptomatic)')
    plt.plot(t, scale(y[3],N=N),                   color='c', linestyle=':',  label='I0 (Infectious, untested')
    plt.plot(t, scale(y[4],N=N),                   color='c', linestyle='--', label='I1 (Infectious, untested)')
    plt.plot(t, scale(aggregate((y[3],y[4])),N=N), color='c',                 label='I (Total Infectious)')
    plt.plot(t, scale(y[5],N=N),                   color='m', linestyle=':',  label='R0 (Recovered, untested)')
    plt.plot(t, scale(y[6],N=N),                   color='m', linestyle='--', label='R1 (Recovered, tested)')
    plt.plot(t, scale(aggregate((y[5],y[6])),N=N), color='m',                 label='R (Total Recovered)')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.title('Progression of COVID-19: Rc = {0:.2f}'.format(Rc))
    plt.grid() 

def plot_infections(infections,control=5):
    for (Rc,t,y) in infections:
        plt.plot (t,y,label='{0:.2f}'.format(Rc))
    plt.title("Total Infections")
    plt.legend(loc='best',title='Basic Reproduction number (Rc)')
    plt.xlabel('Days')
    plt.grid() 
    plt.axvspan(0, control, facecolor='b', alpha=0.125)
    plt.savefig('totals')
    
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Model COVID19 evolution')
    parser.add_argument('--Rc', nargs='+', type=float, default=2.5,      help='Basic Reproduction number')
    parser.add_argument('--seed',          type=int,   default=20,       help='Number of exposed people at start')
    parser.add_argument('--N',             type=int,   default=5000000,  help='Population size')
    parser.add_argument('--nICU',          type=int,   default=300,      help='Number of ICU beds')
    parser.add_argument('--control',       type=int,   default=400,      help='Number of controlled days to be simulated')
    parser.add_argument('--end',           type=int,   default=800,      help='Number of days to be simulated')
    parser.add_argument('--c',             type=float, default=0.1,      help='testing rate for symptomatic cases, per diem')
    parser.add_argument('--alpha',         type=float, default=0.25,     help='E to P transition rate, per diem')
    parser.add_argument('--gamma',         type=float, default=0.1,      help='I to R tranition, per diem')
    parser.add_argument('--delta',         type=float, default=1.0,      help='P to I, per diem')
    parser.add_argument('--epsilon',       type=float, default=0.15,     help='relative infectiousness')
    parser.add_argument('--CFR1',          type=float, default=2.0/100,  help='case fatality rate with cases exceeding ICU max')
    parser.add_argument('--CFR0',          type=float, default=1.0/100,  help='case fatality rate for cases under ICU max')
    parser.add_argument('--pICU',          type=float, default=1.25/100, help='proportion of cases requiring ICU')
    args = parser.parse_args()
    
    beta_divisor = args.epsilon/args.delta + 1/args.gamma
    infections = []
    
    for Rc in args.Rc if isinstance(args.Rc, list) else [args.Rc]:
        sol    = solve_ivp(dy, 
                           (0,args.control),
                           [1-args.seed/args.N, args.seed/args.N, 0, 0, 0, 0, 0],
                           args=(args.N, args.c, args.alpha, Rc/beta_divisor,
                                 args.gamma, args.delta, args.epsilon, args.CFR1, args.CFR0, args.nICU, args.pICU))
        Rbasic = max(args.Rc)
        plt.figure(figsize=(20,6))
        plot_detail(t=sol.t,y=sol.y,N=args.N,Rc=Rc)
        sol_extended  = solve_ivp(dy, 
                           (sol.t[-1],args.end),
                           [y[-1] for y in sol.y],
                           args=(args.N, args.c, args.alpha, Rbasic/beta_divisor,
                                 args.gamma, args.delta, args.epsilon, args.CFR1, args.CFR0, args.nICU, args.pICU))
        infections.append((Rc,
                           np.concatenate((sol.t,sol_extended.t[1:])),
                           aggregate((np.concatenate((sol.y[3],sol_extended.y[3][1:])),
                                      np.concatenate((sol.y[4],sol_extended.y[4][1:]))))))
    
    plt.figure(figsize=(20,6))
    plot_infections(infections,control=args.control)
    
    plt.show()