"""
Método de Visscher para resolver a GPE para BEC's desta vez com vortex
Simão Sá
201205751
Projecto Final MCE
"""
from math import factorial
from numpy import pi, sqrt, exp, linspace, meshgrid, zeros, empty, random, cosh, arctan2
import numpy

from scipy.ndimage.filters import laplace
from scipy.special import hermite

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np

matplotlib.rcParams['figure.dpi'] = 100

class Vortex:
    """Load a wavefunction from file and add vortex"""
    def __init__(self, filename):
        self.filename = filename
        self.wf = numpy.load(filename)
        
    def __call__(self, x, y, t):
        a = 20
        shift = 0.3
        xs = x - shift
        rho = sqrt(xs*xs + y*y)
        f = (rho / a) / sqrt(1 + (rho/a)**2)
        theta = arctan2(y, xs)

        return self.wf * f * exp(1j*theta)
        
        
class Simulation:
    """
xlimit : maximum extent of boundary
N    : number of spatial points
init : initial conditions
potential    : potential : harmonic, softsquare
beta : beta in equations
"""
    def __init__(self, parameters):
        self.parameters = parameters
        # spatial 
        xlimit = parameters['xlimit']
        self.xlimit = xlimit
        N = parameters['N']
        self.N=N
        self.v = linspace(-xlimit, xlimit, N)
        self.dx = self.v[1] - self.v[0]
        
        self.x, self.y = meshgrid(self.v, -self.v)

        # time
        self.steps = 0
        self.time = 0
        self.dt = self.dx**2 / 4 #critério de visscher
       
        # wavefunction
        init_func = parameters['initial']
        self.wf = Wavefunction(init_func, self.x, self.y, self.dt)
        
        # Hamiltonian 

        self.T = Kinetic(self.dx)
        self.V = Potential(parameters['potential'])

        # beta
        self.betaa = parameters['beta']

    
    def timemachine(self, time): #evolui a função
        steps = int(time / self.dt)
        if steps == 0: 
            steps = 1 #  1 step no minimo
        for _ in range(steps):
            self.nonlinear_timejump() #faz o step nao linear que envolve o linear já (mais abaixo)
            
        self.update_time(steps)


    def linear_timejump(self):
        # short names
        x = self.x
        y = self.y
        t = self.time
        dt = self.dt
        real = self.wf.real
        imag = self.wf.imag
        prev = self.wf.prev
        T = self.T
        V = self.V

        # update a função de onda
        real += dt * (T.fastlap(imag) + V(x, y, t) * imag)
        prev[:] = imag[:] #assing nao funcionou
        
        imag -= dt * (T.fastlap(real) + V(x, y, t+dt/2) * real)


    def nonlinear_timejump(self):
        x = self.x
        y = self.y
        t = self.time
        dt = self.dt
        betaa = self.betaa
        real = self.wf.real
        imag = self.wf.imag
        prev = self.wf.prev
        T = self.T
        V = self.V

        # update
        real[:] = (real + dt*(T.fastlap(imag) + V(x, y, t)*imag + betaa*imag*imag*imag)) / (1 - dt*betaa*real*imag)
        prev[:] = imag 
        
        imag[:] = (imag - dt*(T.fastlap(real) + V(x, y, t+dt/2)*real + betaa*real*real*real)) / (1 + dt*betaa*real*imag)


    def update_time(self, steps):
        self.steps += steps
        self.time = self.steps * self.dt



class Kinetic: # termo cinético do hamiltoniano
    def __init__(self, dx):
        self.coef = -0.5 / dx**2

    def fastlap(self, wf):
        new = empty(wf.shape)
        new[1:-1, 1:-1] = self.coef * (wf[:-2, 1:-1] + wf[2:, 1:-1]  # good old dif finitas
                                       + wf[1:-1, :-2] + wf[1:-1, 2:]
                                       - 4*wf[1:-1, 1:-1])
        new[:,0] = new[:,-1] = new[0,:] = new[-1,:] = 0.0
        return new


class Potential(object):#termo do potencial, a classe e callable
    def __init__(self, potential_name):
        self.func = self.__getattribute__(potential_name)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

##formas dos potenciais
    def harmonic(self, x, y, t):
        return 0.5*(x*x + y*y)
        
    def softsquare(self,x,y,t):
        return 0.5*(x*x*x*x+y*y*y*y)
        
        
class Wavefunction: #funcao de onda complexa em que os a parte real e imaginaria estao em tempos diferentes (visscher)
    def __init__(self, init_func, x, y, dt):
        # parte real e imaginaria da funcao de onda atençao aos dt's
        self.real = init_func(x, y, 0).real
        self.imag = init_func(x, y, dt/2).imag

        # guardar parte imaginaria anterior para poder obter a norma (relatorio o porque)
        self.prev = init_func(x, y, -dt/2).imag

        # normalizar
        dx = x[0,1] - x[0,0]
        N = sqrt(abs(self.norm()).sum() * dx**2)
        self.real /= N
        self.imag /= N
        self.prev /= N

    def norm(self):
        return self.real**2 + self.imag*self.prev


##secção de graficos

def awesomeanimation(simulation, interval=0.5):
    L=simulation.xlimit
    x = np.linspace(-L, L, simulation.N)
    y = np.linspace(-L, L, simulation.N)
    wf=simulation.wf
    z=wf.norm()
    global p4, w
    
    # Create a GL View widget to display data
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('Bose-Einstein Condensate')
    w.setCameraPosition(distance=20)

    # Add a grid to the view
    g = gl.GLGridItem()
    g.scale(1,1,1)
    g.setDepthValue(10)  # draw grid after surfaces since they may be translucent
    w.addItem(g)

    
    p4 = gl.GLSurfacePlotItem(x=x,y=y,z=z, shader='viewNormalColor',smooth=True)
    p4.scale(1,1,10) #faco uma escala para se ver melhor
    p4.translate(0,0,-0.001) #faco uma tranlação minima do grafico para 0.001 para ser melhor na grelha o que esta a acontecer ao nivel z=0
    

    fig = plt.figure()
    w.addItem(p4)
    

    def update2():
        simulation.timemachine( interval/2)
        wf=simulation.wf
        N=wf.norm()
        p4.setData(z=N)


    timer = QtCore.QTimer()
    timer.timeout.connect(update2) #funcao que chama o update, tem timer
    timer.start(10)

    plt.show() #por alguma razao preciso de abrir uma janela normal para o python animar o opengl, caso contrario fica parado
    plt.close() #mas fecho logo 




## PARA BRINCAR COM ISTO MEXER DAQUI PARA BAIXO 
##ATENÇÃO AS VARIAVEIS, TEM DE SER IGUAIS AS DO PROGRAMA CORRIDO PARA FAZER O FICHEIRO WF.DAT

if __name__ == '__main__':
    params = {'N': 64, #potencia de 2
              'xlimit': 7,
              'beta': 4,
              'initial': Vortex('wf.dat'),
              'potential': 'harmonic', #harmonic ou softsquare
              }

    sim = Simulation(params)
    awesomeanimation(sim, 0.1)
    