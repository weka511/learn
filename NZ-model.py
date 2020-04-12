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

# This program is based on my reading of the model used for 
# Suppression and Mitigation Strategies for Control of COVID-19 in New Zealand 
# Alex James, Shaun C. Hendy, Michael J. Plank, and Nicholas Steyn (25 March 2020).
# Disclaimer: the program has not been discussed with these authors; it is
# unreviewed, and in no way endorsed by them.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# dy
#
# Compute derivative of state vector
#
# Parameters:
#    t
#    y       State vector. All components are fraction of the population
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
    return [
        -beta * S * (epsilon*P + I),
        +beta * S * (epsilon*P + I) - alpha*E,
        alpha*E - delta * P,
        delta*P -(gamma + c) *I0,
        c*I0 - gamma*I1,
        gamma*(1-CFR)*I0,
        gamma*(1-CFR)*I1
    ]

# scale
#
# Scale values for display
def scale(ys,N=1):
    return [N*y for y in ys]

# aggregate
#
# Used to add related components together for display, e.g. tested and untested.
#
# Parameters:
#     y
#     selector

def aggregate(y,selector=range(7)):
    return [sum([y[i][j] for i in selector]) for j in range(len(y[0]))]

# plot_detail
#
# Plot state vector for one value of Basic Reproduction number
#
# Parameters:
#     t     time
#     y     state vector
#     N     Population size (for scaling)
#     Rc    Basic Reproduction number (for use in title)

def plot_detail(t=[],y=[],N=1,Rc=1,control_days=400,R_uncontrolled=2.5): 
    plt.plot(t, scale(y[0],N=N),                             color='b',                 label='S (Susceptible)')
    plt.plot(t, scale(y[1],N=N),                             color='g',                 label='E (Exposed)')
    plt.plot(t, scale(y[2],N=N),                             color='r',                 label='P (Presymptomatic)')
    plt.plot(t, scale(y[3],N=N),                             color='c', linestyle=':',  label='I0 (Infectious, untested')
    plt.plot(t, scale(y[4],N=N),                             color='c', linestyle='--', label='I1 (Infectious, untested)')
    plt.plot(t, scale(aggregate(y,selector=range(3,5)),N=N), color='c',                 label='I (Total Infectious)')
    plt.plot(t, scale(y[5],N=N),                             color='m', linestyle=':',  label='R0 (Recovered, untested)')
    plt.plot(t, scale(y[6],N=N),                             color='m', linestyle='--', label='R1 (Recovered, tested)')
    plt.plot(t, scale(aggregate(y,selector=range(5,6)),N=N), color='m',                 label='R (Total Recovered)')
    plt.plot(t, scale([1-d for d in aggregate(y)],N=N),      color='k',                 label='Deaths')
    plt.legend(loc='best')
    plt.xlabel('Days')
    plt.title('Progression of COVID-19: Rc = {0:.2f} for {1} days, then {2:.2f}'.format(Rc,control_days,R_uncontrolled))
    plt.grid()
    plt.axvspan(0, control_days, facecolor='b', alpha=0.125)
    plt.savefig('{0}.png'.format(Rc))
    
# plot_infections
#
# Compare infection rates for a range of Rc
#
# Parameters:
#       infections
#       control_days    Number of days controled (for display only)

def plot_infections(infections,control_days=400):
    for (Rc,t,y) in infections:
        plt.plot (t,y,label='{0:.2f}'.format(Rc))
    plt.title('Total Infections: {0} days before control lifted.'.format(control_days))
    plt.legend(loc='best',title='Rc during control')
    plt.xlabel('Days')
    plt.grid() 
    plt.axvspan(0, control_days, facecolor='b', alpha=0.125)
    plt.savefig('totals')

# get_beta
#
# Compute transmission coefficient

def get_beta(R0=2.5,gamma=0.1,delta=1.0,epsilon=0.15):
    return R0/(epsilon/delta + 1/gamma)

if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Model COVID19 evolution')
    parser.add_argument('--Rc',      type=float, default=2.5,      help='Basic Reproduction number', nargs='+')
    parser.add_argument('--seed',    type=int,   default=20,       help='Number of exposed people at start')
    parser.add_argument('--N',       type=int,   default=5000000,  help='Population size')
    parser.add_argument('--nICU',    type=int,   default=300,      help='Number of ICU beds')
    parser.add_argument('--control', type=int,   default=400,      help='Number of controlled days to be simulated')
    parser.add_argument('--end',     type=int,   default=800,      help='Number of days to be simulated')
    parser.add_argument('--c',       type=float, default=0.1,      help='testing rate for symptomatic cases, per diem')
    parser.add_argument('--alpha',   type=float, default=0.25,     help='E to P transition rate, per diem')
    parser.add_argument('--gamma',   type=float, default=0.1,      help='I to R tranition, per diem')
    parser.add_argument('--delta',   type=float, default=1.0,      help='P to I, per diem')
    parser.add_argument('--epsilon', type=float, default=0.15,     help='relative infectiousness')
    parser.add_argument('--CFR1',    type=float, default=2.0/100,  help='case fatality rate with cases exceeding ICU max')
    parser.add_argument('--CFR0',    type=float, default=1.0/100,  help='case fatality rate for cases under ICU max')
    parser.add_argument('--pICU',    type=float, default=1.25/100, help='proportion of cases requiring ICU')
    parser.add_argument('--show',                default=False,    help='Show plots at end of run', action='store_true')
    args = parser.parse_args()
    
    infections = []
     
    for Rc in args.Rc if isinstance(args.Rc, list) else [args.Rc]:
        sol    = solve_ivp(dy, 
                           (0,args.control),
                           [1-args.seed/args.N, args.seed/args.N, 0, 0, 0, 0, 0],
                           args=(args.N, args.c, args.alpha,
                                 get_beta(R0=Rc,gamma=args.gamma,delta=args.delta,epsilon=args.epsilon),
                                 args.gamma, args.delta, args.epsilon, args.CFR1, args.CFR0, args.nICU, args.pICU))
        R_uncontrolled = max(args.Rc) if isinstance(args.Rc, list) else args.Rc
        sol_extended  = solve_ivp(dy, 
                           (sol.t[-1],args.end),
                           [y[-1] for y in sol.y],
                           args=(args.N, args.c, args.alpha,
                                 get_beta(R0=R_uncontrolled,gamma=args.gamma,delta=args.delta,epsilon=args.epsilon),
                                 args.gamma, args.delta, args.epsilon, args.CFR1, args.CFR0, args.nICU, args.pICU))
        infections.append((Rc,
                           np.concatenate((sol.t,sol_extended.t[1:])),
                           np.concatenate((aggregate(sol.y,selector=range(3,5)),
                                           aggregate(sol_extended.y,selector=range(3,5))[1:]))
                           ))
 
        plt.figure(figsize=(20,6))
        plot_detail(t=np.concatenate((sol.t,sol_extended.t[1:])),
                    y=[np.concatenate((sol.y[i],sol_extended.y[i][1:])) for i in range(len(sol.y))],
                    N=args.N,
                    Rc=Rc,
                    control_days=args.control,
                    R_uncontrolled=R_uncontrolled)        
    
    if isinstance(args.Rc, list):
        plt.figure(figsize=(20,6))
        plot_infections(infections,control_days=args.control)
    
    if args.show:
        plt.show()