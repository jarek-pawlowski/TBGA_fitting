import numpy as np
import utils_tb

from utils_tb_k import Flake, PlottingOnFlake
    
#material = utils_tb.TMDCmaterial3(0.388, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)
material = utils_tb.TMDCmaterial11(0.388, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)

lattice = utils_tb.Lattice()
model = utils_tb.BandModel11(material, lattice)
eigen_solver = utils_tb.EigenSolver(model)
plotting = utils_tb.Plotting(eigen_solver.model)

# define flake    
flake_edge_size = [300,300]  # [800,800] 
flake = Flake(material.a0, flake_edge_size, shape='rhombus', find_neighbours=False)
flake.set_bfield(0.)  # in teslas

# build reciprocal lattice
flake.set_lattice_parameters(lattice)
flake.create_reciprocal_lattice()

parameters = np.array([0.5303333767767483, 1.1139367006174299, -0.22029626391997187, 0.63512133230208, 0.2357301545544615, 0.2965495264896972, -0.1176609724634411, -0.15780539528891308, -0.14252978069227137, -0.25214770221155436, 1.2997392133678685, -3.1852554573522593])
parameters = np.array([-0.5652248235471982, -1.0230287615236027, -0.004144542454125683, 0.2436480099993278, 0.17519028277983584, 0.5512095930801296, -0.12507917536939017, 0.005127585989217813, -0.3688908301602572, -0.2977195613062045, 1.7874994849771866, -2.3006786044193563])
parameters = np.array([-1.8238384477373688, -1.0089858116474357, 0.12352674166638113, 0.021410759993732657, -1.1106091991084863, 0.18907145481738855, 0.2147975359099036, -2.3554756963922023, 1.1581041926697713, -4.999999440525821, 1.9365082609871396, -0.36707593571303426])

material.update_parameters(*parameters)
material.set_e_field(1.)

bands, vecs, spins  = eigen_solver.solve_at_points(k_points=flake.nodes_k, get_spin=True, get_vec=True)
np.save('bands.npy', bands)
np.save('vecs.npy', vecs)
np.save('spins.npy', spins)
#bands = np.load('bands.npy')
#vecs = np.load('vecs.npy')  # to index eigenstates: [no_k_point,:,no_eigenstate]
#spins = np.load('spins.npy')

flake.center_reciprocal_lattice()
berry = []
no_band = 10
for plaq in flake.plaquettes:
    F = np.vdot(vecs[plaq[0],:,no_band], vecs[plaq[3],:,no_band])
    F *= np.vdot(vecs[plaq[1],:,no_band], vecs[plaq[0],:,no_band])
    F *= np.vdot(vecs[plaq[2],:,no_band], vecs[plaq[1],:,no_band])
    F *= np.vdot(vecs[plaq[3],:,no_band], vecs[plaq[2],:,no_band])    
    F = np.angle(F)
    kx = flake.nodes_k[plaq[0],0]+flake.nodes_k[plaq[1],0]+flake.nodes_k[plaq[2],0]+flake.nodes_k[plaq[3],0]
    ky = flake.nodes_k[plaq[0],1]+flake.nodes_k[plaq[1],1]+flake.nodes_k[plaq[2],1]+flake.nodes_k[plaq[3],1]
    berry.append([kx,ky,F])
berry = np.array(berry)
berry[np.abs(berry[:,2])>1.]=0.

plot = PlottingOnFlake(flake, directory='plots')
plot.plot_berry_flake(berry)

bands, spins, _  = eigen_solver.solve_BZ_path(get_spin=True)
plotting.plot_Ek_output_target_s([bands, spins], [bands, spins], plot_name='berry_ee')