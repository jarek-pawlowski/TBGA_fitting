import utils_evo
import utils_tb
import datetime
from numpy.random import uniform

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
#real_bands = utils_tb.load_data_tomek('./even_DFT.txt')[:, ::-1]
real_bands, real_spins = utils_tb.load_data_kasia('./ElectronicStructureInfo.dat')
real_bands += 1.897  # VB max to zero

evo_search = utils_evo.EvoSearch(
    material=material,
    k_indices=lattice.k_indices,
    ks_indices=lattice.ks_indices,
    eigen_solver=eigen_solver,
    real_bands=real_bands,
    real_spins=real_spins,
    bounds=[2.,],
    bands=["coductance_and_valence", "coductance_and_valence_reduced"],
    compostition_loss=[.5],  #[.5, .2, .1],
    inter_intra_ratio_bounds=[0.09,0.07,0.05],
    strategy=["best1bin","best1exp",],
    pop_size=[503],  # 500
    tol=0.0001,
    mutation=[(0.05, 0.15)],
    recombination=[.6,],  # .7 | seems like >0.7 is doing worse
    metrics=["rmse"],
    max_iter=1000,
    workers=8,  # -1 = all cores
    no_params=[26,6],
    results_path=f'./results/results_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.txt',
    plots_path=None
)

#print(evo_search._calculate_residuals([0.5303333767767483, 1.1139367006174299, -0.22029626391997187, 0.63512133230208, 0.2357301545544615, 0.2965495264896972, -0.1176609724634411, -0.15780539528891308, -0.14252978069227137, -0.25214770221155436, 1.2997392133678685, -3.1852554573522593], "coductance_and_valence", "rmse" ))
#print(evo_search._calculate_residuals([-0.5652248235471982, -1.0230287615236027, -0.004144542454125683, 0.2436480099993278, 0.17519028277983584, 0.5512095930801296, -0.12507917536939017, 0.005127585989217813, -0.3688908301602572, -0.2977195613062045, 1.7874994849771866, -2.3006786044193563], "coductance_and_valence", "rmse" ))

evo_search.search_for_best()
