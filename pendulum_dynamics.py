from math import sin, cos
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Simulation:
    """
    This class encapsulates a dynamical simulation in a standard form
    It defines the numerical method and time stepping procedures and 
        is designed to make animations more convenient to implement
    """
    
    def __init__(self, state, acceleration, method, kwargs):
        """
        Initializes an instance of the class
        
        Inputs:
            state: The initial state (gets updated after each step)
            acceleration: The function used to update the state
            method: The numerical method used for time stepping
            kwargs: A dictionary of arguments for the acceleration function
        
        Returns
            None
        """
        self.t = 0
        self.state = state
        self.acceleration = acceleration
        self.method = method
        self.kwargs = kwargs
        self.plots = None
        return
    
    def step(self, dt):
        """
        Initializes an instance of the class
        
        Inputs:
            dt: The timestep used to update the current state
        
        Returns
            None
        """
        self.state = self.method(self.t, dt, self.state, self.acceleration, self.kwargs)
        self.t += dt
        return



def RK4(t, h, state, f, kwargs):
    """
    Implements the standard 4th order Runge-Kutta method 
    
    Inputs:
        t: The current time
        h: The temporal step size
        state: The current state (in a form compatible with f)
        f: The function to update the current state (e.g. acceleration)
        kwargs: A dictionary of keyword arguments (passed to f)
        
    Returns:
        The updated state after a single time step (as a numpy array)
    """
    dim = len(state)//2
    K = [ np.zeros_like(state) for i in range(4) ]
    C = [ 0, h/2., h/2., h, h]
    
    for i in range(4): 
        A = f(t + C[i], state + C[i]*K[i-1], kwargs)
        K[i] = np.asarray(A)
    
    return state + ( K[0] + 2*K[1] + 2*K[2] + K[3] ) * h / 6.0


def SSPRK3(t, h, state, f, kwargs):
    """
    Implements a four-stage Strong Stability Preserving Runge-Kutta 3 method
    It has a large region of stability and allows for larger step sizes (CFL < 2) 
    
    Inputs:
        t: The current time
        h: The temporal step size
        state: The current state (in a form compatible with f)
        f: The function to update the current state (e.g. acceleration)
        kwargs: A dictionary of keyword arguments (passed to f)
        
    Returns:
        The updated state after a single time step (as a numpy array)
    """
    dim = len(state)//2
    S = [ np.asarray(state) ] + [ np.zeros_like(state) for i in range(4) ]
    C = [ [ (0,0.5) ], [ (1, 0.5) ], [ (0,2./3.), (2,1./6.) ], [ (3,0.5) ] ]
    B = [ (0.5, 0, 0), (0.5, 1, h/2.), (1./6., 2, h), (0.5, 3, h/2.) ]
    
    for i in range(4):
        b, k, dt = B[i]
        A = f(t + dt, S[k], kwargs)
        S[i+1] = b * (S[k] + h * A ) 
        for j, c in C[i]:
            S[i+1] += c * S[j]
    return S[-1]

def crank_nicholson(t, h, state, f, kwargs):
    """
    Implements the unconditionally stable 2nd order Crank-Nicholson method
    
    Inputs:
        t: The current time
        h: The temporal step size
        state: The current state (in a form compatible with f)
        f: The function to update the current state (e.g. acceleration)
        kwargs: A dictionary of keyword arguments (passed to f)
        
    Returns:
        The updated state after a single time step (as a numpy array)
    """
    dim = len(state)//2
    K = [ np.zeros_like(state) for i in range(2) ]
    C = [ 0, h, h ]
    
    for i in range(2): 
        A = f(t + C[i], state + C[i]*K[i-1], kwargs)
        K[i][dim:] = np.asarray(A)
        K[i][:dim] = state[dim:] + C[i+1] * K[i][dim:]
    
    return state + ( K[0] + K[1] ) * h / 2.0



def acceleration_double_pendulum(t, state, kwargs, g = 9.81):
    """
    Calulates the acceleration of a double pendulum in a given state
    
    Inputs:
        t: The current time
        state: The current state of the pendulum in the form: 
                    np.array( [ X1,..., Xn, V1,...,Vn ] )
        kwargs: A dict defining the masses and lengths in the form:
                    {'M': [M1, M1] , 'L': [L1,L2] } 
        g: (Optional) the acceleration due to gravity
        
    Returns:
        The updated state after a single time step (as a numpy array)
    """
    theta = [ state[0], state[1] ]
    omega = [ state[2], state[3] ]
    M, L = kwargs['M'], kwargs['L']
    Mt = M[0] + M[1]
    
    denominator = 2 * M[0]  + M[1] - M[1] * cos( 2 * theta[0] - 2 * theta[1])
    sin_delta = sin( theta[0] - theta[1] )
    cos_delta = cos( theta[0] - theta[1] )
    d_state = [ 0, 0, 0, 0 ]
    
    d_state[2] = - g * (Mt + M[0]) * sin(theta[0]) + M[1] * g * sin(theta[0] - 2*theta[1])
    d_state[2] -= 2 * sin_delta * M[1] * (L[1]*omega[1]**2 + L[0] * cos_delta * omega[0]**2)
    d_state[2] /= L[0] * denominator 
    
    d_state[3] = L[0] * Mt * omega[0]**2 + g * Mt * cos(theta[0])
    d_state[3] += L[1] * M[1] * cos_delta * omega[1]**2
    d_state[3] *= 2 * sin_delta / ( L[1] * denominator )
    
    d_state[0] = omega[0] 
    d_state[1] = omega[1] 
    
    return np.asarray(d_state)



def acceleration_n_pendulum(t, state, kwargs, g = -9.81):
    """
    Calulates the acceleration of an arbitrary n-pendulum in a given state
    Implements the method used by Kloppel at al. in the following paper
        "Parallel simulation of large scale multibody systems" 
        https://onlinelibrary.wiley.com/doi/pdf/10.1002/pamm.201510327
    
    Inputs:
        t: The current time
        state: The current state of the pendulum in the form: 
                    np.array( [ X1,..., Xn, V1,...,Vn ] )
        kwargs: A dict defining the masses and lengths in the form:
                    {'M': [M1, M1] , 'L': [L1,L2] } 
        g: (Optional) the acceleration due to gravity
        
    Returns:
        The updated state after a single time step (as a numpy array)
    """
    L = kwargs['L']
    N = len(state)//2
    last = N-1
    
    A = [ sin(state[i]) * (N-i) * g / L[i] for i in range(N) ]
    for i in range(N):
        total = 0
        for j in range(i):
            total += sin( state[j] - state[i] ) * state[N+j] ** 2
        A[i] += (N-i) * total
        for j in range(i+1,N):
            A[i] += (N-j) * sin( state[j] - state[i] ) * state[N+j] ** 2
    
    M = [ [ 0 for j in range(N) ] for i in range(N) ]
    for i in range(N):
        for j in range(N):
            M[i][j] = (N - max(i,j)) * cos(state[j]-state[i])
    
    M = np.asarray(M)
    M = np.linalg.inv(M)
    d_state = state.copy()
    d_state[:N] = state[N:]
    d_state[N:] = np.dot(M,A)
    return d_state



def init(sim, figsize=(10,10)):
    """
    Initializes the plots for the animation function
    
    Inputs:
        sim: An initialized instance of the Simulation class
    
    Returns:
        fig: the pyplot figure used for animation
    """
    
    L = sim.kwargs['L']
    
    if 'steps_per_frame' in sim.kwargs:
        sim.kwargs['framesteps'] = range(sim.kwargs['steps_per_frame'])
    else:
        sim.kwargs["framesteps"] = [0]
    
    fig, ax = plt.subplots( figsize=figsize )
    ax.set_xlim( -sum(L), sum(L) )
    ax.set_ylim( -sum(L), sum(L) )

    P1, P2 = angles_to_cartesian(sim.state[:2], L)
    nframes = sim.kwargs['trail_frames']
    
    rods, = ax.plot([], [], c='y', lw=5)
    
    cmap1 = plt.get_cmap('Blues')(np.arange(0,1,1./nframes))
    cmap2 = plt.get_cmap('Greens')(np.arange(0,1,1./nframes))

    lc1 = matplotlib.collections.LineCollection([])
    lc2 = matplotlib.collections.LineCollection([])
    
    ax.add_collection(lc1)
    ax.add_collection(lc2)

    sim.plots = {"lc1": lc1, "lc2": lc2, 'cmap1': cmap1, 'cmap2': cmap2, 
                 "rods": rods, "trail1": [ [P1,P1] ], "trail2": [ [P2,P2] ] } 
    
    return fig

def angles_to_cartesian(angles, L):
    """
    Calculates the locations of all pendulum masses
    
    Inputs:
        angles: The list of rod angles
        L: The list of rod lengths
    
    Returns:
        P: The list of positions for all pendulum masses
    """
    N = len(angles)
    P = [ 0 for n in range(N) ]
    
    x = L[0]*np.sin(angles[0])
    y = -L[0]*np.cos(angles[0])
    P[0] = ( x, y )
    for i in range(1,N):
        x2 = x + L[i]*np.sin(angles[i])
        y2 = y - L[i]*np.cos(angles[i])
        P[i] = (x2,y2)
        x, y = x2, y2
    return P


def animate(i, sim):
    """
    Updates the frames of the animation
    
    Inputs:
        i: The frame number (not used, but required by the library)
        sim: An instance of the Simulation class
    
    Returns:
        (ax1,...axN): A tuple of all plot axes
    """
    for step in sim.kwargs['framesteps']:
        sim.step(sim.kwargs['dt'])
        
    dim = len(sim.state) // 2
    P = angles_to_cartesian(sim.state[:dim], sim.kwargs['L'])
    P1, P2 = P[-2:]
    
    last1, last2 = sim.plots['trail1'][-1], sim.plots['trail2'][-1]
    sim.plots['trail1'].append( [last1[-1], P1] )
    sim.plots['trail2'].append( [last2[-1], P2] )
    
    sim.plots['rods'].set_xdata( [ 0 ] + [ p[0] for p in P ] )
    sim.plots['rods'].set_ydata( [ 0 ] + [ p[1] for p in P ])
    
    nframes = sim.kwargs['trail_frames']
    sim.plots['lc1'].set( segments = sim.plots['trail1'][-nframes:], 
                          color = sim.plots['cmap1'][-nframes:] )
    sim.plots['lc2'].set( segments = sim.plots['trail2'][-nframes:], 
                          color = sim.plots['cmap2'][-nframes:] )
    
    return (sim.plots[key] for key in ['lc1', 'lc2', 'rods'])

