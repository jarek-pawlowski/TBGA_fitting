import numpy as np

import utils_evo
import utils_tb

def residuum(real_bands, fitted_bands):
    #return np.sum(((real_bands-fitted_bands)**2)[45:55,1:5])
    #return np.sum(((real_bands-fitted_bands)**2)[45:60,7:11])
    return np.sum(((real_bands-fitted_bands)**2)[:,4:5])
    
#material = utils_tb.TMDCmaterial3(0.388, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)
material = utils_tb.TMDCmaterial(0.388, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)
lattice = utils_tb.Lattice()
model = utils_tb.BandModel(material, lattice)
eigen_solver = utils_tb.EigenSolver(model)
#real_bands = utils_tb.load_np('./wsi2sb4_DFT.npy')
#real_bands = utils_tb.load_np('./wsi2sb4_DFT_2_4.npy')
real_bands = utils_tb.load_np('./wsi2sb4_DFT_3_3.npy')
plotting = utils_tb.Plotting(eigen_solver.model)
#evo_parameters = [-0.2547142200062724, -0.06426584756038467, 0.4078916351816453, 0.2626703076819392, -0.0897745227635216, 0.3218540829733574, -0.6801509591747495, 1.1104338643267875, 0.7445131072661981, -0.33240411336967646]
#evo_parameters = [-0.3421798810922879, 0.9039927843265207, 0.6557782418317614, -0.15345236352256908, 0.6104292253712387, -0.558454298267775, 0.023089321716619936, -0.2506652841776824, -1.2296887465682935, -2.3460932425176053, -1.330281330667991, -0.11443550957539698]
evo_parameters = np.array([0.5303333767767483, 1.1139367006174299, -0.22029626391997187, 0.63512133230208, 0.2357301545544615, 0.2965495264896972, -0.1176609724634411, -0.15780539528891308, -0.14252978069227137, -0.25214770221155436, 1.2997392133678685, -3.1852554573522593])
evo_parameters = np.array([-0.5652248235471982, -1.0230287615236027, -0.004144542454125683, 0.2436480099993278, 0.17519028277983584, 0.5512095930801296, -0.12507917536939017, 0.005127585989217813, -0.3688908301602572, -0.2977195613062045, 1.7874994849771866, -2.3006786044193563])
evo_parameters = np.array([-1.8238384477373688, -1.0089858116474357, 0.12352674166638113, 0.021410759993732657, -1.1106091991084863, 0.18907145481738855, 0.2147975359099036, -2.3554756963922023, 1.1581041926697713, -4.999999440525821, 1.9365082609871396, -0.36707593571303426])

best_fit = evo_parameters
best_err = 100.  # 1.

"""
for i in range(100):
    modifiers = np.random.randn(12)*.05 + 1.  # 10
    new_params = best_fit*modifiers
    material.update_parameters(*new_params)
    fitted_bands, _ = eigen_solver.solve_BZ_path()
    err = residuum(real_bands, fitted_bands)
    if err < best_err:
        best_fit = new_params
        best_err = err
    print(i, best_err)
print(best_fit)
"""

material.update_parameters(*best_fit)
fitted_bands, fitted_spins, fitted_comps  = eigen_solver.solve_BZ_path(get_spin=True)
breakpoint()
real_spins = np.ones_like(fitted_spins)
plotting.plot_Ek_output_target_s([real_bands, real_spins], [fitted_bands, fitted_spins], "fit")