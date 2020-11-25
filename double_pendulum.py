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
        return
    
    def step(self, dt):
        self.state = self.method(self.t, dt, self.state, self.acceleration, self.kwargs)
        self.t += dt
        return



def RK4(t, h, state, f, kwargs):
    dim = len(state)//2
    K = [ np.zeros_like(state) for i in range(4) ]
    C = [ 0, h/2., h/2., h, h]
    
    for i in range(4): 
        A = f(t + C[i], state + C[i]*K[i-1], kwargs)
        K[i] = np.asarray(A)
    
    return state + ( K[0] + 2*K[1] + 2*K[2] + K[3] ) * h / 6.0


def crank_nicholson(t, h, state, f, kwargs):
    dim = len(state)//2
    K = [ np.zeros_like(state) for i in range(2) ]
    C = [ 0, h, h ]
    
    for i in range(2): 
        A = f(t + C[i], state + C[i]*K[i-1], kwargs)
        K[i][dim:] = np.asarray(A)
        K[i][:dim] = state[dim:] + C[i+1] * K[i][dim:]
    
    return state + ( K[0] + K[1] ) * h / 2.0



def acceleration_double_pendulum(t, state, kwargs, g = 9.81):
    theta = [ state[0], state[1] ]
    omega = [ state[2], state[3] ]
    M, L = kwargs['M'], kwargs['L']
    Mt = M[0] + M[1]
    
    denominator = 2 * M[0]  + M[1] - M[1] * cos( 2 * theta[0] - 2 * theta[1])
    sin_delta = sin( theta[0] - theta[1] )
    cos_delta = cos( theta[0] - theta[1] )
    d_omega = [ 0, 0, 0, 0 ]
    
    d_omega[2] = - g * (Mt + M[0]) * sin(theta[0]) + M[1] * g * sin(theta[0] - 2*theta[1])
    d_omega[2] -= 2 * sin_delta * M[1] * (L[1]*omega[1]**2 + L[0] * cos_delta * omega[0]**2)
    d_omega[2] /= L[0] * denominator 
    
    d_omega[3] = L[0] * Mt * omega[0]**2 + g * Mt * cos(theta[0])
    d_omega[3] += L[1] * M[1] * cos_delta * omega[1]**2
    d_omega[3] *= 2 * sin_delta / ( L[1] * denominator )
    
    d_omega[0] = omega[0] 
    d_omega[1] = omega[1] 
    
    return d_omega



def acceleration_n_pendulum(t, state, kwargs, g = -9.81):
    # "Parallel simulation of large scale multibody systems" by Kloppel et al.
    #Source: https://onlinelibrary.wiley.com/doi/pdf/10.1002/pamm.201510327
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
    d_state = np.asarray(state)
    d_state[:N] = state[N:]
    d_state[N:] = np.dot(M,A)
    return d_state



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


def animate(i, dt, sim):
    sim.step(dt)
    dim = len(sim.state) // 2
    P = angles_to_cartesian(sim.state[:dim], sim.kwargs['L'])
    P1, P2 = P[:2]
    
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

