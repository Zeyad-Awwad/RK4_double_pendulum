from math import sin, cos
import numpy as np


class Simulation:
    
    def __init__(self, state, acceleration, method, plots, kwargs):
        self.t = 0
        self.state = state
        self.acceleration = acceleration
        self.method = method
        self.plots = plots
        self.kwargs = kwargs
    
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




def init(ax):
    rods, = ax.plot([], [], c='y', lw=3)
    line1, = ax.plot([], [], c='b')
    line2, = ax.plot([], [], c='g')
    return {"line1": line1, "line2": line2, "rods": rods, "trail1": [[],[]], "trail2": [[],[]] } 

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
    sim.plots['trail1'][0].append(P1[0])
    sim.plots['trail1'][1].append(P1[1])
    sim.plots['trail2'][0].append(P2[0])
    sim.plots['trail2'][1].append(P2[1])
    
    frames = sim.kwargs['trail_frames']
    for key in ['trail1','trail2']:
        sim.plots[key][0] = sim.plots[key][0][-frames:]
        sim.plots[key][1] = sim.plots[key][1][-frames:]
    
    sim.plots['rods'].set_xdata( [ 0, P1[0], P2[0] ])
    sim.plots['rods'].set_ydata( [ 0, P1[1], P2[1] ])
    sim.plots['line1'].set_xdata( sim.plots['trail1'][0] )
    sim.plots['line1'].set_ydata( sim.plots['trail1'][1] )
    sim.plots['line2'].set_xdata( sim.plots['trail2'][0] )
    sim.plots['line2'].set_ydata( sim.plots['trail2'][1] )
    
    return (sim.plots[key] for key in ['line1', 'line2', 'rods'])




