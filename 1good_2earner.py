
import numpy as np
from scipy.optimize import minimize
import sys
import matplotlib.pyplot as plt

# Define household class
class household:
    
    # Initialize a household object.
    # Provide predetermined parameter values
    def __init__(self, σ=2, 𝛽=0.995, Na=50, T=2):

        # Store the characteristics of household in the class object
        self.σ          = σ                   # Coefficient of RRA
        self.𝛽          = 𝛽                   # Discount factor
        self.r          = 0.02
        self.w1         = 1
        self.w2         = 0.491
        self.Na         = Na                  # Number of grid points for a0 state
        self.T          = T                  # 
        self.a_min      = 0
        self.a_max      = 4
        self.a0_state   = np.linspace(self.a_min,self.a_max,self.Na) # Discretize a0 state
        self.e0_state   = np.asarray([0, 1]) # Construct e0 state as ndarray type instead of list type
        self.Ne         = len(self.e0_state)
        self.Π          = np.asarray([[0.94,0.06],[0.1,0.9]])   # Stochastic matrix, Prob(e1|e0), as ndarray type
        self.Vf         = np.zeros((self.Na,self.Ne,T+1))   # Value function, a1
        self.ap         = np.zeros((self.Na,self.Ne,T))   # Policy function, a1
        self.hp1         = np.zeros((self.Na,self.Ne,T))  # Policy function, current hours for earner 1
        self.hp2         = np.zeros((self.Na,self.Ne,T))  # Policy function, current hours for earner 2
        self.b          = 0.3
        self.ψ1         = 2.538
        self.ψ2         = 1.953
        self.η1         = 0.528
        self.η2         = 0.850

    def util(self,cons,h1,h2):
        '''
        This function returns the value of the additively separable utility function
        '''
        if self.σ != 1:
            uu = cons**(1-self.σ)/(1-self.σ) - self.ψ1*h1**(1+1/self.η1)/(1+1/self.η1) - self.ψ2*h2**(1+1/self.η2)/(1+1/self.η2)
        else:
            uu = np.log(cons)
        return uu

    def get_Vf(self,age):
        '''
        This function updates the value function
        '''
        return self.Vf[:,:,age]
    
    # Update the policy function, a1
    def set_Vf(self, V0, age):
        self.Vf[:,:,age] = V0
    
    def get_ap(self,age):
        '''
        This function updates the value function
        '''
        return self.ap[:,:,age]

    # Update the policy function, a1
    def set_ap(self, a1, age):
        self.ap[:,:,age] = a1
        
    # Update the policy function, h1
    def set_hp1(self, h1, age):
        self.hp1[:,:,age] = h1
        
    # Update the policy function, h2
    def set_hp2(self, h2, age):
        self.hp2[:,:,age] = h2
        
def bellman(pol,hh,a0,e0,VV,P):
    '''
    This function computes bellman equation for a given state (a0,e0).
    Input:
        ap: evaluating point
        hp1: evaluating point
        hp2: evaluating point
        hh: household object
        (a0,e0) state
        V1: value function at age t+1 evaluated at (a',e')
        P1: probability distribution of e' conditional on the current e0
    Output:
        -vv: bellman equation
    ''' 
    ap = pol[0]
    hp1 = pol[1]
    hp2 = pol[2]
    a0_state   = hh.a0_state
    w1         = hh.w1
    w2         = hh.w2
    r          = hh.r
    β          = hh.β
    b          = hh.b
    
    # Interpolate next period's value function evaluated at (a',e')
    # using 1-dimensional interpolation function in numpy
    V0      = np.interp(ap,a0_state,VV[:,0])
    V1      = np.interp(ap,a0_state,VV[:,1])
    EV      = P[0]*V0 + P[1]*V1
        
    # Interpolated value cannot be NaN or Inf
    if np.isnan(V0) or np.isinf(V0): print("bellman: V0 is NaN.")
    if np.isnan(V1) or np.isinf(V1): print("bellman: V1 is NaN.")

    # Compute consumption at a given (a0,e0) and a'       
    cons = w1 * hp1 * e0 + w2 * hp2 + b*(1 - e0) + (1 + r) * a0 - ap
    
    # Consumption must be non-negative
    if cons<=0:
        vv = (cons-10)*1000
    else:
        # Compute value function
        vv  = hh.util(cons,hp1,hp2) + β*EV
    print(a0,e0,ap,hp1,hp2,cons,vv)
    
    return -vv



if __name__ == "__main__":
    
    # Model parameters 

    # Create a household instance
    hh     = household()

    Na        = hh.Na
    Ne        = hh.Ne
    a0_state  = hh.a0_state
    e0_state  = hh.e0_state

    # Backward induction: solve an individual's problem from the last period
    for age in reversed(range(hh.T)):
        sys.stdout.write('age = %d.\n' % age)
    
        # Value function at age t+1
        V1    = hh.get_Vf(age+1)
        V0    = np.zeros((Na,Ne))
        a1    = np.zeros((Na,Ne))
        h1    = np.zeros((Na,Ne))
        h2    = np.zeros((Na,Ne))
    
        # Iterate for each state (a0,e0)
        # enumerate() gives an index and value of the list
        for ind in range(Na*Ne):
        
            ia  = ind // Ne
            ie  = ind % Ne
        
            a0 = a0_state[ia]
            e0 = e0_state[ie]
            P  = hh.Π[ie]
            
            # Find an optimal saving by using Nelder-Mead method
            amin = a0_state[0]
            amax = a0_state[Na-1]
            hmin = 0
            hmax = 1
            bnds = ((amin,amax),(hmin,hmax),(hmin,hmax))
            # print(bnds)
            init   = (0.1,0.1,0.1)
            result = minimize(bellman,init,args=(hh,a0,e0,V1,P),bounds=bnds,method='Nelder-Mead')
            V0[ia,ie] = -result.fun
            a1[ia,ie]= result.x[0]
            h1[ia,ie]= result.x[1]
            h2[ia,ie]= result.x[2]
#            print(result.x)
#            print(result.fun)

        # Store the policy function in the class object
        hh.set_Vf(V0,age)
        hh.set_ap(a1,age)
        hh.set_hp1(h1,age)
        hh.set_hp2(h2,age)

        # Plot Policy Function
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.plot(a0_state, a1[:,0],label="bad shock")
        plt.plot(a0_state, a1[:,1],label="good shock")
        plt.plot(a0_state, a0_state,'k--',label="$45^{\circ}$ line")
        plt.xlabel("State space, a")
        plt.ylabel("Policy function a' ")
        plt.legend(loc='upper left', fontsize = 14)
        plt.show()    
        
        
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.plot(a0_state, h1[:,0],label="bad shock")
        plt.plot(a0_state, h1[:,1],label="good shock")
        plt.xlabel("State space, a")
        plt.ylabel("Policy function h1' ")
        plt.legend(loc='upper right', fontsize = 14)
        plt.show()   
        
        
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.plot(a0_state, h2[:,0],label="bad shock")
        plt.plot(a0_state, h2[:,1],label="good shock")
        plt.xlabel("State space, a")
        plt.ylabel("Policy function h2' ")
        plt.legend(loc='upper right', fontsize = 14)
        plt.show()   