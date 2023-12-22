"""
Método de Visscher para resolver a GPE para BEC's
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

class QHO:
    def __init__(self, n, xshift=0, yshift=0):
        self.n = n
        self.xshift = xshift
        self.yshift = yshift
        self.E = n + 0.5
        self.coef = 1 / sqrt(2**n * factorial(n)) * (1 / pi)**(1/4)
        self.hermite = hermite(n)

    def __call__(self, x, y, t):
        xs = x - self.xshift
        ys = y - self.yshift
        return self.coef * exp(-(xs**2 + ys**2) / 2 - 1j*self.E*t) * self.hermite(x) * self.hermite(y) #apply formula
        
        
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
def normalanimation(simulation,interval=0.2): 
    L = simulation.xlimit
    wf=simulation.wf
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    norm = ax.imshow(wf.norm() , extent=(-L, L, -L, L), cmap=plt.cm.jet_r)
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    norm1 = ax1.plot_surface(simulation.x, simulation.y, wf.norm() , rstride=1, cstride=1, cmap=plt.cm.jet_r , linewidth=0, antialiased=False)
    ax1.set_zlim3d(0, 0.3)

    
    def update1(i): #update da animação
        simulation.timemachine(interval/2)

        norm.set_data(wf.norm())
        ax1.clear()
        ax1.plot_surface(simulation.x, simulation.y, wf.norm() , rstride=1, cstride=1, cmap=plt.cm.jet_r , linewidth=0, antialiased=False)
        ax.set_title('T = {:3.2f}'.format(simulation.time)+r' $ \beta $ ='+str(simulation.betaa))
        ax1.set_zlim3d(0, 0.3)

    anim = animation.FuncAnimation(fig, update1, interval=1) #função que chama a funcao de update

    plt.show()

def awesomeanimation(simulation, interval=0.5):
    L=simulation.xlimit
    x = np.linspace(-L, L, simulation.N)
    y = np.linspace(-L, L, simulation.N)
    wf=simulation.wf
    z=wf.norm()
    global p4, w, aa
    # aa=0 to print
    
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

    # Assign color based on height
    cm = pg.ColorMap([0, 0.4, 1], 
                    [(0., 0., 1., 0.), 
                    (0., 1., 1., 0.), 
                    (0., 1., 0., 1.)])
    colors = cm.map((z+z.min()) / (z.max()-z.min()), mode='float')
    
    
    
    p4 = gl.GLSurfacePlotItem(x=x,y=y,z=z, shader='shaded',smooth=True, colors=colors)
    p4.scale(1,1,10) #faco uma escala para se ver melhor
    p4.translate(0,0,-0.001) #faco uma tranlação minima do grafico para 0.001 para ser melhor na grelha o que esta a acontecer ao nivel z=0
    

    fig = plt.figure()
    w.addItem(p4)
    

    def update2():
        #uncomment to print images, program will be slower, images are not that good
        # global aa
        # aa=aa+1
        # if aa%30== 0:
        #     filename = 'Bose'+str(aa)+'.png'
        #     d = w.renderToArray((10000, 10000))
        #     pg.makeQImage(d).save(filename)
        simulation.timemachine( interval/2)
        wf=simulation.wf
        N=wf.norm()
        p4.setData(z=N)


    timer = QtCore.QTimer()
    timer.timeout.connect(update2) #funcao que chama o update, tem timer
    timer.start(30)

    plt.show() #por alguma razao preciso de abrir uma janela normal para o python animar o opengl, caso contrario fica parado
    plt.close() #mas fecho logo 




## PARA BRINCAR COM ISTO MEXER DAQUI PARA BAIXO 

if __name__ == '__main__':
    params = {'N': 64, #potencia de 2
              'xlimit': 3,
              'beta': 6,
              'initial': QHO(n=0, xshift=0) ,
              'potential': 'harmonic', #harmonic ou softsquare
              }

    sim = Simulation(params)
    #normalanimation(sim, 0.3)
    awesomeanimation(sim, 0.3)
    

