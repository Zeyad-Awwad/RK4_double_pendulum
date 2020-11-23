from math import sin, cos
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class Simulation:
    
    def __init__(self, state, acceleration, method, kwargs):
        self.t = 0
        self.state = state
        self.acceleration = acceleration
        self.method = method
        self.kwargs = kwargs
        self.plots = None
    
    def step(self, dt):
        self.state = self.method(self.t, dt, self.state, self.acceleration, self.kwargs)
        self.t += dt
    
    

def RK4(t, h, state, f, kwargs):
    dim = len(state)//2
    K = [ np.zeros_like(state) for i in range(4) ]
    C = [ 0, h/2., h/2., h]
    
    for i in range(4): 
        A = f(t + C[i], state + C[i]*K[i-1], kwargs)
        K[i][dim:] = np.asarray(A)
        K[i][:dim] = state[dim:] + C[i] * K[i][dim:]
    
    return state + ( K[0] + 2*K[1] + 2*K[2] + K[3] ) * h / 6.0


def acceleration_double_pendulum(t, state, kwargs, g = 9.81):
    theta = [ state[0], state[1] ]
    omega = [ state[2], state[3] ]
    M, L = kwargs['M'], kwargs['L']
    Mt = M[0] + M[1]
    
    denominator = 2 * M[0]  + M[1] - M[1] * cos( 2 * theta[0] - 2 * theta[1])
    sin_delta = sin( theta[0] - theta[1] )
    cos_delta = cos( theta[0] - theta[1] )
    d_omega = [ 0, 0 ]
    
    d_omega[0] = - g * (Mt + M[0]) * sin(theta[0]) + M[1] * g * sin(theta[0] - 2*theta[1])
    d_omega[0] -= 2 * sin_delta * M[1] * (L[1]*omega[1]**2 + L[0] * cos_delta * omega[0]**2)
    d_omega[0] /= L[0] * denominator 
    
    d_omega[1] = L[0] * Mt * omega[0]**2 + g * Mt * cos(theta[0])
    d_omega[1] += L[1] * M[1] * cos_delta * omega[1]**2
    d_omega[1] *= 2 * sin_delta / ( L[1] * denominator )
    
    return d_omega




def init(ax, sim): #state, nframes):
    P1, P2 = angles_to_cartesian(sim.state[:2], sim.kwargs['L'])
    nframes = sim.kwargs['trail_frames']
    
    rods, = ax.plot([], [], c='y', lw=5)
    #line1, = ax.plot([], [], c='b')
    #line2, = ax.plot([], [], c='g')
    
    
    cmap1 = plt.get_cmap('Blues')(np.arange(0,1,1./nframes))
    cmap2 = plt.get_cmap('Greens')(np.arange(0,1,1./nframes))

    lc1 = matplotlib.collections.LineCollection([])
    lc2 = matplotlib.collections.LineCollection([])
    
    ax.add_collection(lc1)
    ax.add_collection(lc2)

    sim.plots = {"lc1": lc1, "lc2": lc2, 'cmap1': cmap1, 'cmap2': cmap2, 
                 "rods": rods, "trail1": [ [P1,P1] ], "trail2": [ [P2,P2] ] } 
    
    return

def angles_to_cartesian(angles, L):
    a1, a2 = angles
    x, y = L[0]*np.sin(a1), -L[0]*np.cos(a1)
    x2 = x + L[1]*np.sin(a2)
    y2 = y - L[1]*np.cos(a2)
    P1 = (x, y) 
    P2 = (x2,y2) 
    return P1, P2

def animate(i, dt, sim):
    sim.step(dt)
    P1, P2 = angles_to_cartesian(sim.state[:2], sim.kwargs['L'])
    last1, last2 = sim.plots['trail1'][-1], sim.plots['trail2'][-1]
    sim.plots['trail1'].append( [last1[-1], P1] )
    sim.plots['trail2'].append( [last2[-1], P2] )
    
    nframes = sim.kwargs['trail_frames']
    #for key in ['trail1','trail2']: sim.plots[key] = sim.plots[key][-nframes:]
    
    sim.plots['rods'].set_xdata( [ 0, P1[0], P2[0] ])
    sim.plots['rods'].set_ydata( [ 0, P1[1], P2[1] ])
    
    
    sim.plots['lc1'].set( segments = sim.plots['trail1'][-nframes:], 
                          color = sim.plots['cmap1'][-nframes:] )
    sim.plots['lc2'].set( segments = sim.plots['trail2'][-nframes:], 
                          color = sim.plots['cmap2'][-nframes:] )

    
    return (sim.plots[key] for key in ['lc1', 'lc2', 'rods'])

