import numpy as np
import utils_tb

from utils_evo import SlicedBands
from sklearn.metrics import mean_squared_error

from scipy.optimize import minimize, dual_annealing, basinhopping, differential_evolution, shgo

lattice_const = 3.323
# --- Slater-Koster layers parameters
# layer up: MoSe2
Ed_up          = -0.09
Ep1_up         = -5.01
Ep0_up         = -5.30
Vdp_sigma_up   = -3.08
Vdp_pi_up      =  1.08
Vdd_sigma_up   = -0.94
Vdd_pi_up      =  0.75
Vdd_delta_up   =  0.13
Vpp_sigma_up   =  1.39
Vpp_pi_up      = -0.45
Ep1_odd_up     =  Ep1_up
Ep0_odd_up     =  Ep0_up
Ed_odd_up      =  Ed_up
lambda_M_up    =  0.186/2.  # 0.15  # it should be: 0.186d0/2.0d0
lambda_X2_up   =  0.200/2.  # 0.15  # it should be: 0.200d0/2.0d0
# layer down: WSe2
Ed_down        = -0.12
Ep1_down       = -4.17
Ep0_down       = -4.52
Vdp_sigma_down = -3.31
Vdp_pi_down    =  1.16
Vdd_sigma_down = -1.14
Vdd_pi_down    =  0.72
Vdd_delta_down =  0.17
Vpp_sigma_down =  1.16
Vpp_pi_down    = -0.25
Ep1_odd_down   =  Ep1_down
Ep0_odd_down   =  Ep0_down
Ed_odd_down    =  Ed_down 
lambda_M_down  =  0.472/2.  # 0.35  # it should be: 0.472d0/2.0d0
lambda_X2_down =  -0.390/2.  # -0.20  # it should be: -0.390d0/2.0d0
# interlayer
Vpp_sigma_inter =  0.5  # positive
Vpp_pi_inter    = -0.5  # negative
Vdd_sigma_inter = -0.3  # negative
Vdd_pi_inter    =  0.3  # positive
Vdd_delta_inter = -0.6  # negative

material = utils_tb.Newmaterial(lattice_const,
                    Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,Ep0_odd_up,Ed_odd_up,lambda_M_up,lambda_X2_up,
                    Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,lambda_M_down,lambda_X2_down,
                    Vpp_sigma_inter,Vpp_pi_inter,Vdd_sigma_inter,Vdd_pi_inter,Vdd_delta_inter, 0.)

k_path = utils_tb.load_k_path('kpointsDFT.dat')
lattice = utils_tb.Lattice(BZ_path=k_path)
lattice.select_k_indices(distance=4)
model = utils_tb.BandModel(material, lattice)
eigen_solver = utils_tb.EigenSolver(model)
plotting = utils_tb.Plotting(eigen_solver.model)
#real_bands = utils_tb.load_data_tomek('./even_DFT.txt')[:, ::-1]
real_bands, real_spins = utils_tb.load_data_kasia('./ElectronicStructureInfo.dat')
real_bands += 1.897  # VB max to zero

evo_parameters = np.array([0.001167388164884331, 0.18009870232129319, 0.018709663056286185, -0.07722657524912134, 0.19961295858746758, -0.08157683673132583, -0.11073960008881399, 0.07046468732906268, 0.1169077198529775, 0.12384528456879772, -0.030956136429465334, -0.07588637019092144, -0.12285700486937311, 0.19357608324373224, 0.17084236745357528, 0.08151081377766772, 0.19933040507880695, -0.17109331385254958, 0.06625137749184233, -0.017411913060875816, -0.1684433637034547, 0.01703766352343621, 0.15736941757551404, -0.19734114624880528, -0.19996678122952397, -0.1979093111675889, 1.2565623286568979, -1.0625058707012418, -0.7159558182014585, -0.7922816562019475, 0.1290933491848918, -0.4448624573013966])
evo_parameters = np.array([-0.08828459419459263, -0.1065741944844086, 0.09919704816807498, -0.07416195494171325, 0.1952221266318892, -0.03034493211602889, 0.10992051153337323, 0.07087152083916033, -0.12965666068887788, 0.1480397947726102, -0.19997061728491722, -0.08851103211364558, 0.010812216552237653, 0.19997661722899673, -0.06526532788243998, -0.16923684769237257, 0.1994052437841276, -0.17865912437931702, 0.056517616399663066, 0.012717312059901831, -0.07005458127537055, 0.18801211157673056, -0.013558255136722864, -0.19987520318938803, -0.19999794876221774, -0.19994044135041536, -0.6692022745228159, 0.3443433736406849, 0.36447015572031205, 0.9909950157777585, -0.07228095551111346, -0.33175426823773035])
evo_parameters = np.array([0.11604795391793897, 0.015800549851649092, -0.19696927850714924, 0.03177534285142216, 0.19263049124595086, -0.03220597698178247, 0.006565735527119587, 0.03643741833963885, -0.04546239586238985, -0.017896484704495674, 0.06285651989434995, -0.19191296197177093, -0.14822243567459573, 0.142941185171015, 0.18132093039840133, 0.07642484423675402, 0.19978595530862564, -0.16707676282750916, 0.03664970868749551, -0.08004472223598941, 0.13485317991267318, 0.17108150108542045, -0.08988653800857443, 0.15088320746670034, -0.19994856718649154, -0.1993331811133499, 0.32320883116186705, -0.7474801418060057, 0.05004647049900246, 1.388539586062283, -0.2157945374817689, -0.3207502786282179])
evo_parameters = np.array([0.030411386321679234, -0.06768325945268493, -0.13985445526998835, 0.10354798970965703, 0.09511473252137233, 0.017192102201147058, 0.0009731466399938161, 0.09503724114298144, 0.006517343991442069, -0.01765029376948105, -0.139991453790443, 0.022210650586162457, -0.1399665270159982, 0.1396462854912086, -0.01873315173927232, -0.13965675237116856, 0.0479373707384002, -0.1399412022063997, -0.08320015408731994, 0.13994380829292616, 0.06816633552643377, -0.0807977909651734, -0.11578177646634323, -0.13999999976336608, -0.1395760909761852, -0.1398140425616252, -1.1006155026388882, -0.1546785496916816, -0.4117617630474648, 1.8318187602187703, -0.329984241003078, -0.21562176129413912])


def residuum(eigen_solver, real_bands, real_spins):
    #predicted_bands = self.eigen_solver.solve_BZ_path()
    predicted_bands = eigen_solver.solve_at_points(eigen_solver.model.BZ_path[lattice.k_indices])
    real_bands_slice, predicted_bands_slice = SlicedBands(
        real_bands=real_bands[lattice.k_indices],
        predicted_bands=predicted_bands,
        bands="coductance_reduced"
    ).get_sliced_bands().get_bands()
    spin_err = 0. 
    comp_err = 0.
    for ks_idx in lattice.ks_indices:
        _, spins, comps = eigen_solver.solve_k(eigen_solver.model.BZ_path[ks_idx], get_spin=True)
        #spin_err += sqrt(mean_squared_error(self.real_spins[ks_idx,7:15], spins[13:21]))
        spin_err += np.sqrt(mean_squared_error(real_spins[ks_idx,9:13], spins[15:19]))
        # to be implemented:
        #comp_err += 1. if np.sum(np.array([comps[0,4]+comps[2,4], comps[1,5], comps[1,6], comps[0,7]+comps[2,7]]) - np.array([1., 1., 1., 1.])*compostition_loss) < 0. else 0.
    sqrt = np.sqrt(mean_squared_error(real_bands_slice, predicted_bands_slice))*10.
    print(sqrt, spin_err)
    return sqrt + spin_err + comp_err

idx_to_modify = [0, 4, 7, 13, 16, 17, 19, 20, 29, 30, 31]  # 31
idx_to_modify = [29, 30, 31]  # 31

def _residuum(x, eigen_solver, real_bands, real_spins):
    new_params = evo_parameters.copy()
    new_params[idx_to_modify] = x
    material.update_parameters(*new_params)
    return residuum(eigen_solver, real_bands, real_spins)

material.update_parameters(*evo_parameters)
print(residuum(eigen_solver, real_bands, real_spins))
fitted_bands, fitted_spins, _ = eigen_solver.solve_BZ_path(get_spin=True)
plotting.plot_Ek_output_target_ss([real_bands, real_spins], [[fitted_bands, fitted_spins]], "fit_starting")

res = dual_annealing(_residuum, 
               bounds = [(-2.,2.) for _ in range(len(idx_to_modify))],
               x0=evo_parameters[idx_to_modify], 
               args=(eigen_solver, real_bands, real_spins))
print(_residuum(res.x, eigen_solver, real_bands, real_spins))
new_params = evo_parameters.copy()
new_params[idx_to_modify] = res.x
material.update_parameters(*new_params)
fitted_bands, fitted_spins, _ = eigen_solver.solve_BZ_path(get_spin=True)
real_spins = np.ones_like(fitted_spins)
plotting.plot_Ek_output_target_ss([real_bands, real_spins], [[fitted_bands, fitted_spins]], "fit_mc_refined")
print(new_params)