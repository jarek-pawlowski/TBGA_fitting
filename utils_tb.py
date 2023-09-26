import os
import numpy as np
from scipy.linalg import eigh
import datetime
import matplotlib.pyplot as plt


class AtomicUnits:
    """Class storing atomic units.

    All variables, arrays in simulations are in atomic units.

    Attributes
    ----------
    Eh : float
        Hartree energy (in meV)
    Ah : float
        Bohr radius (in nanometers)
    Th : float
        time (in picoseconds)
    Bh : float
        magnetic induction (in Teslas)
    """
    # atomic units
    Eh=27211.4 # meV
    Ah=0.05292 # nm
    Th=2.41888e-5 # ps
    Bh=235051.76 # Teslas

au = AtomicUnits()


class TMDCmaterial:
    """ Class containing lattice model parameters.

    """
    def __init__(self, a0, dp, Vdps, Vdpp, Vd2s, Vd2p, Vd2d, Vp2s, Vp2p, Ed, Ep1, Ep0, lm, lx2):
        self.a0 = a0/au.Ah
        self.dr = self.a0/np.sqrt(3.)
        self.dp = dp/au.Ah
        self.dd = np.sqrt(self.dr**2+self.dp**2)
        self.dim = 12
        self.dim2 = self.dim*self.dim
        self.dim12 = int(self.dim/2)
        # hoppings
        self.Vdps = Vdps
        self.Vdpp = Vdpp
        self.Vd2s = Vd2s
        self.Vd2p = Vd2p
        self.Vd2d = Vd2d
        self.Vp2s = Vp2s
        self.Vp2p = Vp2p
        # onsite energy
        self.Ed = Ed
        self.Ep1 = Ep1
        self.Ep0 = Ep0
        self.diag = np.tile(np.array([self.Ed, self.Ed, self.Ed,
                                      self.Ep1, self.Ep0, self.Ep1]),2)
        # intrinsic spin-orbit
        self.lm = lm/au.Eh
        self.lx2 = lx2/au.Eh
        self.l_diag = np.array([-self.lm, 0.,  self.lm, -self.lx2/2., 0.,  self.lx2/2., 
                                 self.lm, 0., -self.lm,  self.lx2/2., 0., -self.lx2/2.])

    def update_parameters(self,  Vdps, Vdpp, Vd2s, Vd2p, Vd2d, Vp2s, Vp2p, Ed, Ep1, Ep0, lm, lx2):
         # hoppings
        self.Vdps = Vdps
        self.Vdpp = Vdpp
        self.Vd2s = Vd2s
        self.Vd2p = Vd2p
        self.Vd2d = Vd2d
        self.Vp2s = Vp2s
        self.Vp2p = Vp2p
        # onsite energy
        self.Ed = Ed
        self.Ep1 = Ep1
        self.Ep0 = Ep0
        self.diag = np.tile(np.array([self.Ed, self.Ed, self.Ed,
                                      self.Ep1, self.Ep0, self.Ep1]),2)
        self.lm = lm
        self.lx2 = lx2
        self.l_diag = np.array([-self.lm, 0.,  self.lm, -self.lx2/2., 0.,  self.lx2/2., 
                                 self.lm, 0., -self.lm,  self.lx2/2., 0., -self.lx2/2.])


class TMDCmaterial3:
    """ Class containing lattice model parameters.

    Attributes
    ----------
    a0 : float
        MoS2 lattice constant (in nanometers)
    dim : int
        Hilbert space (sub)dimension: no of orbitals x spin degree = 3 x 2,
        dimension of the whole state-space = dim*N, where N is a no of lattice nodes
    dim2 : int
        squared dim: dim2 = dim*dim
    dim12 : int
        halved dim: dim12 = dim/2
    t.. : float
        tight-binding hopping parameters
    e. : float
        tight-binding onsite energies
    lso : float
        intinsic spin-orbit energy (in meV)
    """
    def __init__(self, a0, t0, t1, t2, t11, t12, t22, e0, e1, e2, lso):
        self.a0 = a0/au.Ah
        self.dim = 6
        self.dim2 = self.dim*self.dim
        self.dim12 = int(self.dim/2)
        # hoppings
        self.t0 = t0/au.Eh
        self.t1 = t1/au.Eh
        self.t2 = t2/au.Eh
        self.t11 = t11/au.Eh
        self.t12 = t12/au.Eh
        self.t22 = t22/au.Eh
        # onsite energy
        self.e0 = e0/au.Eh
        self.e1 = e1/au.Eh
        self.e2 = e2/au.Eh
        self.diag = np.array([self.e0,self.e1,self.e2,self.e0,self.e1,self.e2])
        # intrinsic spin-orbit
        self.lso = lso/au.Eh

    def update_parameters(self, t0, t1, t2, t11, t12, t22, e0, e1, e2, lso):
         # hoppings
        self.t0 = t0
        self.t1 = t1
        self.t2 = t2
        self.t11 = t11
        self.t12 = t12
        self.t22 = t22
        # onsite energy
        self.e0 = e0
        self.e1 = e1
        self.e2 = e2
        self.diag = np.array([self.e0,self.e1,self.e2,self.e0,self.e1,self.e2])
        self.lso = lso


class Lattice:

    def __init__(self):
        lattice_vectors = np.array([[1.,0.], [-.5, np.sqrt(3.)/2.]])
        #self.K_points = [np.array([np.pi*4./3., 0.]), np.array([np.pi*4./6., np.pi*2./np.sqrt(3.)])]
        self.K_points = [np.array([np.pi*4./3.,np.pi*4./np.sqrt(3.)]), np.array([np.pi*2./3,np.pi*2./np.sqrt(3.)])]
        RB1 = np.array([0.,-1.])
        RB2 = np.array([np.sqrt(3.),1.])/2.
        RB3 = np.array([-np.sqrt(3.),1.])/2.
        #
        RA1 = RB1 - RB3
        RA2 = RB2 - RB3
        RA3 = RB2 - RB1
        RA4 = RB3 - RB1
        RA5 = RB3 - RB2
        RA6 = RB1 - RB2
        self.hoppingsMX = [RB1, RB2, RB3]
        self.hoppingsMM = [RA1, RA2, RA3, RA4, RA5, RA6]
        #
        K = self.K_points[0][0]
        M = K*3./2
        G = K*np.sqrt(3.)/2.
        dk = (M+G)/120.  # magic number to get exactly 120 points at the path
        self.critical_points = [(r'$\Gamma$', 0.), ('K', K), ('M', M), (r'$\Gamma$', M+G)]
        self.critical_points_w_names = {"gamma_1": 0., "K": K, "M": M, "gamma_2": M+G}
        k_GK = [[x, 0.] for x in np.arange(0, K, dk)] # k varying from Gamma to K point within the BZ
        k_KM = [[x, 0.] for x in np.arange(K, M, dk)] # k varying from K to M point within the BZ
        k_MG = [[M, y]  for y in np.linspace(0, G, num=int(G/dk), endpoint=True)] # k varying from M to Gamma point within the BZ
        self.BZ_path = np.concatenate((k_GK, k_KM, k_MG)) # full path within the BZ
        self.k_points = None
    
    def select_k_indices(self, distance=5):
        """ 
        select points along BZ path:
        [G,Q,K-2d,K-d,K,K+d,K+2d,M]
        with neighbor points taken at some distance
        """
        d = distance
        self.k_indices = [0,25,51-d*4,51-d*3,51-d*2,51-d,51,51+d,51+d*2,51+d*3,51+d*4,77]  # G = 0, Q = 25, K = 51, M = 77 
        self.ks_indices = [51-d,51+d]

class BandModel:

    def __init__(self, parameters, lattice):
        self.m = parameters
        self.l = lattice
        self.hoppingsMM = [h*self.m.a0/np.sqrt(3.) for h in self.l.hoppingsMM]
        self.hoppingsMX = [h*self.m.a0/np.sqrt(3.) for h in self.l.hoppingsMX]
        self.BZ_path = self.l.BZ_path/self.m.a0
        self.critical_points = [(p[0], p[1]/self.m.a0) for p in self.l.critical_points]
        self.K_points = [p/self.m.a0 for p in self.l.K_points]

    def hopping_matrix_(self, x, y, x1, y1, linkstype):
        """
        create 6x6 hopping matrix that represents hopping integral within the tight-binding model

        orbitals basis = {Dm2, D0, Dp2, PEm1, PE0, PEp1}

        """
        m = self.m
        hh_m=np.zeros((m.dim,m.dim), dtype=np.complex128)
        if linkstype == 1:
        # M-M hoppings:
            L = (x1-x)/m.a0
            M = (y1-y)/m.a0
            hh_m[0,0] = (3.*m.Vd2s + 4.*m.Vd2p + m.Vd2d)/8.
            hh_m[0,1] = (np.sqrt(3./2.)/4.)*(1.j*M+L)**2*(m.Vd2d-m.Vd2s)
            hh_m[0,2] = (1.j*M+L)**4*(3.*m.Vd2s - 4.*m.Vd2p + m.Vd2d)/8.
            hh_m[1,0] = (np.sqrt(3./2.)/4.)*(-1.j*M+L)**2*(m.Vd2d-m.Vd2s)
            hh_m[1,1] = (3.*m.Vd2d+m.Vd2s)/4.
            hh_m[1,2] = (np.sqrt(3./2.)/4.)*(1.j*M+L)**2*(m.Vd2d-m.Vd2s)
            hh_m[2,0] = (-1.j*M+L)**4*(3.*m.Vd2s - 4.*m.Vd2p + m.Vd2d)/8.
            hh_m[2,1] = (np.sqrt(3./2.)/4.)*(-1.j*M+L)**2*(m.Vd2d-m.Vd2s)
            hh_m[2,2] = (3.*m.Vd2s + 4.*m.Vd2p + m.Vd2d)/8.
        elif linkstype == 2:
        # X2-X2 hoppings
            L = (x1-x)/m.a0
            M = (y1-y)/m.a0
            hh_m[3,3] = (m.Vp2s+m.Vp2p)/2.
            hh_m[3,4] = 0.
            hh_m[3,5] = -1.*(1.j*M+L)**2*(m.Vp2s-m.Vp2p)/2.  # -1 = Maciek correction
            hh_m[4,3] = 0.
            hh_m[4,4] = m.Vp2p
            hh_m[4,5] = 0.
            hh_m[5,3] = -1.*(-1.j*M+L)**2*(m.Vp2s-m.Vp2p)/2.  # -1 = Maciek correction
            hh_m[5,4] = 0.
            hh_m[5,5] = (m.Vp2s+m.Vp2p)/2.
        else:
        # M-X2 or X2-M hoppings
            L = (x1-x)/m.dd
            M = (y1-y)/m.dd
            if linkstype == 4:
            # X2-M hoppings, T(-R) = T^\dag(R)
                L *= -1
                M *= -1
            hh_m[0,3] = (1.j*M+L)*(np.sqrt(3.)/2.*m.Vdps*((m.dp/m.dd)**2-1.)-m.Vdpp*((m.dp/m.dd)**2+1.))/np.sqrt(2.)
            hh_m[0,4] = -(1.j*M+L)**2*(m.dp/m.dd)*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)/2.
            hh_m[0,5] = -(1.j*M+L)**3*(np.sqrt(3.)/2.*m.Vdps-m.Vdpp)/np.sqrt(2.)*(-1.)  # (-1) = Maciek correction
            hh_m[1,3] = -(-1.j*M+L)*((3.*(m.dp/m.dd)**2-1.)*m.Vdps-2.*np.sqrt(3.)*(m.dp/m.dd)**2*m.Vdpp)/2.
            hh_m[1,4] = -(m.dp/m.dd)*((3.*(m.dp/m.dd)**2-1.)*m.Vdps-2.*np.sqrt(3.)*((m.dp/m.dd)**2-1.)*m.Vdpp)/np.sqrt(2.)
            hh_m[1,5] = -( 1.j*M+L)*((3.*(m.dp/m.dd)**2-1.)*m.Vdps-2.*np.sqrt(3.)*(m.dp/m.dd)**2*m.Vdpp)/2.*(-1.)  # (-1) = Maciek correction
            hh_m[2,3] = -(-1.j*M+L)**3*(np.sqrt(3.)/2.*m.Vdps-m.Vdpp)/np.sqrt(2.)
            hh_m[2,4] = -(-1.j*M+L)**2*(m.dp/m.dd)*(np.sqrt(3.)*m.Vdps-2.*m.Vdpp)/2.
            hh_m[2,5] = (-1.j*M+L)*(np.sqrt(3.)/2.*m.Vdps*((m.dp/m.dd)**2-1.)-m.Vdpp*((m.dp/m.dd)**2+1.))/np.sqrt(2.)*(-1.)  # (-1) = Maciek correction
            if linkstype == 4:
            # X2-M hoppings, T(-R) = T^\dag(R)
                hh_m[3:6,0:3] = np.conjugate(hh_m[0:3,3:6]).transpose()
                hh_m[0:3,3:6] = 0.
        # spin-down block is the same:
        hh_m[m.dim12:,m.dim12:] = hh_m[:m.dim12,:m.dim12]
        return hh_m

    def build_tb_hamiltonian(self, kx, ky):
        hh_m = np.zeros((self.m.dim,self.m.dim), dtype=np.complex128)
        diagonal = self.m.diag.copy()
        # intrinistic spin-orbit coupling -- diagonal part:
        diagonal += self.m.l_diag
        np.fill_diagonal(hh_m, diagonal)
        # hoppings
        for h in self.hoppingsMX:
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 3)*np.exp(1.j*(kx*h[0]+ky*h[1]))
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 4)*np.exp(1.j*(kx*h[0]+ky*h[1]))
        for h in self.hoppingsMM:
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 1)*np.exp(1.j*(kx*h[0]+ky*h[1]))
            hh_m += self.hopping_matrix_(0., 0., h[0], h[1], 2)*np.exp(1.j*(kx*h[0]+ky*h[1]))
        return hh_m
    
    def hopping_matrix_6(self, x, y, x1, y1, parameters):
        """
        create 6x6 hopping matrix that represents hopping integral within the tight-binding model
        """
        m = parameters 
        hh_m=np.zeros((m.dim,m.dim), dtype=np.complex128)
        # which of hopping vector R1...R6??
        # see e.g.: Phys. Rev. B 91, 155410 (2015) or Phys. Rev. B 88, 085433 (2013).
        # R1 / R6
        if abs(y - y1) < 1.e-8:
            if x1 > x: R=1
            if x1 < x: R=6
        # R2 / R3
        if abs((x1 - x) - 0.5 *m.a0) < 1.e-8:
            if y1 > y: R=3
            if y1 < y: R=2
        # R4 / R5
        if abs((x1 - x) + 0.5 *m.a0) < 1.e-8:
            if y1 > y: R=4
            if y1 < y: R=5
        if R == 1:
            hh_m[0,0] = m.t0;  hh_m[0,1] = m.t1;   hh_m[0,2] = m.t2
            hh_m[1,0] = -m.t1; hh_m[1,1] = m.t11;  hh_m[1,2] = m.t12
            hh_m[2,0] = m.t2;  hh_m[2,1] = -m.t12; hh_m[2,2] = m.t22
        if R == 6:
            hh_m[0,0] = m.t0;  hh_m[0,1] = -m.t1;  hh_m[0,2] = m.t2
            hh_m[1,0] = m.t1;  hh_m[1,1] = m.t11;  hh_m[1,2] = -m.t12
            hh_m[2,0] = m.t2;  hh_m[2,1] = m.t12;  hh_m[2,2] = m.t22
        if R == 2:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = 0.5*m.t1 - np.sqrt(3.0)/2*m.t2;                       hh_m[0,2] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = -0.5*m.t1 - np.sqrt(3.0)/2*m.t2; hh_m[1,1] = 0.25*m.t11 + 0.75*m.t22;                              hh_m[1,2] = -np.sqrt(3.0)/4*m.t11 - m.t12 + np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2;  hh_m[2,1] = -np.sqrt(3.0)/4*m.t11 + m.t12 + np.sqrt(3.0)/4*m.t22; hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        if R == 4:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = -0.5*m.t1 - np.sqrt(3.0)/2*m.t2;                      hh_m[0,2] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = 0.5*m.t1 - np.sqrt(3.0)/2*m.t2;  hh_m[1,1] = 0.25*m.t11 + 0.75*m.t22;                              hh_m[1,2] = -np.sqrt(3.0)/4*m.t11 + m.t12 + np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2; hh_m[2,1] = -np.sqrt(3.0)/4*m.t11 - m.t12 + np.sqrt(3.0)/4*m.t22; hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        if R == 3:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = 0.5*m.t1 + np.sqrt(3.0)/2*m.t2;                       hh_m[0,2] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = -0.5*m.t1 + np.sqrt(3.0)/2*m.t2; hh_m[1,1] = 0.25*m.t11 + 0.75* m.t22;                             hh_m[1,2] = np.sqrt(3.0)/4*m.t11 - m.t12 - np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2; hh_m[2,1] = np.sqrt(3.0)/4*m.t11 + m.t12 - np.sqrt(3.0)/4*m.t22;  hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        if R == 5:
            hh_m[0,0] = m.t0;                            hh_m[0,1] = -0.5*m.t1 + np.sqrt(3.0)/2*m.t2;                      hh_m[0,2] = -np.sqrt(3.0)/2*m.t1 - 0.5*m.t2
            hh_m[1,0] = 0.5*m.t1 + np.sqrt(3.0)/2*m.t2;  hh_m[1,1] = 0.25*m.t11 + 0.75*m.t22;                              hh_m[1,2] = np.sqrt(3.0)/4*m.t11 + m.t12 - np.sqrt(3.0)/4.0*m.t22
            hh_m[2,0] = np.sqrt(3.0)/2*m.t1 - 0.5*m.t2;  hh_m[2,1] = np.sqrt(3.0)/4*m.t11 - m.t12 - np.sqrt(3.0)/4*m.t22;  hh_m[2,2] = 3.0/4.0*m.t11 + 1.0/4.0*m.t22
        # spin-down block is the same:
        hh_m[m.dim12:,m.dim12:] = hh_m[:m.dim12,:m.dim12]
        return hh_m  
    
    def build_tb_hamiltonian3(self, kx, ky):
        hh_m = np.zeros((self.m.dim,self.m.dim), dtype=np.complex128)
        np.fill_diagonal(hh_m, self.m.diag)  
        # intrinistic spin-orbit coupling
        hh_m[1,2] = 1.j*self.m.lso
        hh_m[2,1] = -1.j*self.m.lso
        hh_m[4,5] = -1.j*self.m.lso
        hh_m[5,4] = 1.j*self.m.lso
        # hoppings
        for h in self.hoppingsMM:
            hh_m += self.hopping_matrix_6(0., 0., h[0], h[1], self.m)*np.exp(1.j*(kx*h[0]+ky*h[1]))   
        return hh_m
    

class EigenSolver:

    def __init__(self, model):
        self.model = model

    def solve_k(self, k, get_spin=False):
        hamiltonian = self.model.build_tb_hamiltonian(k[0],k[1])
        if get_spin is False:
            return eigh(hamiltonian, eigvals_only=True)
        else:
            val, vec = eigh(hamiltonian, eigvals_only=False)
            vec2 = np.real(np.conjugate(vec)*vec)
            no_bands = int(vec2.shape[0]/2)
            vec2 = vec2.reshape((2,no_bands,-1))
            spin = np.sum(vec2[0,:,:], axis=0)-np.sum(vec2[1,:,:], axis=0)
            comp = np.sum(vec2, axis=0)
            return val, spin, comp
        
    def solve_at_points(self, k_points, get_spin=False):
        if get_spin is False:
            return np.array([self.solve_k(k) for k in k_points])
        else:
            vals = []
            spins = []
            comps = []
            for k in k_points:
                val, spin, comp = self.solve_k(k, get_spin=True)
                vals.append(val)
                spins.append(spin)
                comps.append(comp)
            return np.array(vals), np.array(spins), np.array(comps)

    def solve_BZ_path(self, get_spin=False):
        return self.solve_at_points(self.model.BZ_path, get_spin=get_spin)


class Plotting:
    """ Plotting utils.

    Attributes
    ----------
    grid_k : List[List]
        2d list containing full path within the BZ
    critical_points : List[Tuple]
        list of tuples containing critical points`s names and their coordinates
    """
    def __init__(self, model, directory=None):
        self.grid_k = model.BZ_path
        self.critical_points = model.critical_points
        if directory:
            self.directory = os.path.join('./', directory)
            os.makedirs(directory, exist_ok=True)
        else:
            self.directory = './'

    def plot_Ek(self, Ek, x_label='k (nm$^{-1}$)', y_label='E (meV)'):
        """ Plots dispersion relation.

        Parameters
        ----------
        Ek : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        _, ax = plt.subplots()
        ax.axes.set_aspect(.0035)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # plot dispersion relation
        Ek = np.array(Ek)
        for band_idx in range(Ek.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah ,Ek[:,band_idx]*au.Eh, label='Band' + str(band_idx))

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 100))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = 'ek.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_Ek_output_target(self, Ek_target, Ek_output1, plot_name, Ek_output2=None):
        """ Plots dispersion relations for
        two given lists of bands.

        Parameters
        ----------
        Ek_. : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(3.5)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')


        # plot dispersion relation
        Ek_target = np.array(Ek_target)
        Ek_output1 = np.array(Ek_output1)
        if Ek_output2 is not None:
            Ek_output2 = np.array(Ek_output2)
        for band_idx in range(Ek_target.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx], color='green', label='Target band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output1[:,band_idx], '--', color='blue', label='Fitted band')
            if Ek_output2 is not None:
                ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output2[:,band_idx], color='red', label='Decoder output')


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = f'{plot_name}.png'
        plt.savefig(os.path.join(self.directory, "plots", filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_Ek_output_target_s(self, target, output, plot_name):
        """ Plots dispersion relations for
        two given lists of bands.

        Parameters
        ----------
        Ek_. : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        pointsize=5.
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(3.5)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')
        ax.set_ylim([-4,3])

        # plot dispersion relation
        Ek_target = np.array(target[0])
        spin_target = np.array(target[1])
        Ek_output = np.array(output[0])
        spin_output = np.array(output[1])
        for band_idx in range(Ek_target.shape[1]):
            ax.scatter((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx], s=pointsize, marker='.', c='k', cmap='bwr',label='Target band')
            ax.scatter((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output[:,band_idx], s=pointsize, marker='.', c=spin_output[:,band_idx], cmap='bwr', label='Fitted band')

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = f'{plot_name}.png'
        plt.savefig(os.path.join(self.directory, "plots", filename), bbox_inches='tight', dpi=300)
        plt.close()
        
    def plot_Ek_output_target0(self, Ek_target, Ek_output1, Ek_output2, Ek_output3):
        """ Plots dispersion relations for
        two given lists of bands.

        Parameters
        ----------
        Ek_. : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(10.5)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')
        
        
        # plot dispersion relation
        Ek_target = np.array(Ek_target)
        Ek_output1 = np.array(Ek_output1)
        if Ek_output2 is not None:
            Ek_output2 = np.array(Ek_output2)
        for band_idx in range(Ek_target.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx], color='green', label='Target band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output1[:,band_idx], color='orange', label='Fitted best')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output2[:,band_idx], color='blue', label='Fitted band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output3[:,band_idx], color='red', label='Decoder output')
            
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01         
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = 'ek_target0.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()

    def plot_Ek_output_target1(self, Ek_target, Ek_output1, Ek_output2=None):
        """ Plots dispersion relations for
        two given lists of bands.

        Parameters
        ----------
        Ek_. : List[array]
            List of arrays of eigenvalues
        x_label : string
            label of x-axis
        y_label : string
            label of y-axis
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axes.set_aspect(10.5)
        ax.set_xlabel('k (nm$^{-1}$)')
        ax.set_ylabel('E (eV)')

        # plot dispersion relation
        Ek_target = np.array(Ek_target)
        Ek_output1 = np.array(Ek_output1)
        if Ek_output2 is not None:
            Ek_output2 = np.array(Ek_output2)
        for band_idx in range(Ek_target.shape[1]):
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_target[:,band_idx], color='green', label='Target band')
            ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output1[:,band_idx], color='blue', label='Fitted band')
            if Ek_output2 is not None:
                ax.plot((self.grid_k[:,0]+self.grid_k[:,1])/au.Ah,Ek_output2[:,band_idx], color='red', label='Decoder output')


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        text_shift_x = (ax.get_xlim()[1] - ax.get_xlim()[0])*0.01
        plot_max_y = ax.get_ylim()[1]

        for (name, position) in self.critical_points:
             position_k=position/au.Ah
             ax.annotate(name, xy=(position_k-text_shift_x, plot_max_y), xytext=(position_k-text_shift_x, plot_max_y + 0.1))
             ax.axvline(x=position_k, linestyle='--', color='black')
        filename = 'ek_target1.png'
        plt.savefig(os.path.join(self.directory, filename), bbox_inches='tight', dpi=200)
        plt.close()


def load_data(filename):
    return np.roll(np.loadtxt(filename)[::-1,1].reshape(101,12)[:,::2], 2, axis=0)  # roll to move CB minimum into K point


def load_np(filename):
    return np.load(filename)


def normalize(bands, parameters=None):
    min = np.amin(bands)
    max = np.amax(bands)
    if parameters is not None:
        parameters[-3:] -= min  # shift diagonals
        parameters /= max-min  # then scale
    return (bands-min)/(max-min)
