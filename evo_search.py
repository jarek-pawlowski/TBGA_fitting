import utils_evo
import utils_tb
import datetime
from numpy.random import uniform
import numpy as np

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
lambda_X2_down =  0.200/2. # -0.390/2.  # -0.20  # it should be: -0.390d0/2.0d0
# interlayer
Vpp_sigma_inter =  0.5  # positive
Vpp_pi_inter    = -0.5  # negative
Vdd_sigma_inter = -0.3  # negative
Vdd_pi_inter    =  0.3  # positive
Vdd_delta_inter = -0.6  # negative
Vdp_sigma_inter =  0.0
Vpd_sigma_inter =  0.0
Vdp_pi_inter =  0.0
Vpd_pi_inter =  0.0 

material = utils_tb.Newmaterial(lattice_const,
                    Ed_up,Ep1_up,Ep0_up,Vdp_sigma_up,Vdp_pi_up,Vdd_sigma_up,Vdd_pi_up,Vdd_delta_up,Vpp_sigma_up,Vpp_pi_up,Ep1_odd_up,Ep0_odd_up,Ed_odd_up,lambda_M_up,lambda_X2_up,
                    Ed_down,Ep1_down,Ep0_down,Vdp_sigma_down,Vdp_pi_down,Vdd_sigma_down,Vdd_pi_down,Vdd_delta_down,Vpp_sigma_down,Vpp_pi_down,Ep1_odd_down,Ep0_odd_down,Ed_odd_down,lambda_M_down,lambda_X2_down,
                    Vpp_sigma_inter,Vpp_pi_inter,Vdd_sigma_inter,Vdd_pi_inter,Vdd_delta_inter,Vdp_sigma_inter,Vpd_sigma_inter,Vdp_pi_inter,Vpd_pi_inter)

k_path = utils_tb.load_k_path('kpointsDFT.dat')
lattice = utils_tb.Lattice(BZ_path=k_path)
lattice.select_k_indices(distance=4)
model = utils_tb.BandModel(material, lattice, interlayer_type='minimal')
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
    bounds=[2.,5.],
    bands=["coductance_and_valence_reduced", "coductance_and_valence"],
    compostition_loss=[.5, .2],
    inter_intra_ratio_bounds=[0.05, 0.07, 0.09],
    strategy=["best1bin","best1exp",],
    pop_size=[534],  # 500
    tol=0.0001,
    mutation=[(0.05, 0.15)],
    recombination=[.6,],  # .7 | seems like >0.7 is doing worse
    metrics=["rmse"],
    max_iter=500,
    workers=8,  # -1 = all cores
    no_params=[26,9],
    results_path=f'./results1/results_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.txt',
    plots_path="plots1"
)

#print(evo_search._calculate_residuals([0.5303333767767483, 1.1139367006174299, -0.22029626391997187, 0.63512133230208, 0.2357301545544615, 0.2965495264896972, -0.1176609724634411, -0.15780539528891308, -0.14252978069227137, -0.25214770221155436, 1.2997392133678685, -3.1852554573522593], "coductance_and_valence", "rmse" ))
#print(evo_search._calculate_residuals([-0.5652248235471982, -1.0230287615236027, -0.004144542454125683, 0.2436480099993278, 0.17519028277983584, 0.5512095930801296, -0.12507917536939017, 0.005127585989217813, -0.3688908301602572, -0.2977195613062045, 1.7874994849771866, -2.3006786044193563], "coductance_and_valence", "rmse" ))

parameters = np.array([-5.958861367832076E-002, -5.07768325945268, -5.43985445526999, -2.97645201029034, 1.17511473252137, -0.922807897798853, 0.750973146639994, 0.225037241142981, 1.39651734399144, -0.467650293769481, -5.21767471324313, -5.41764380468383, -0.199555140694319, 
                       1.964628549120861E-002, -4.18873315173927, -4.65965675237117, -3.26206262926160, 1.02005879779360, -1.22320015408732, 0.859943808292926, 0.238166335526434, 1.07920220903483, -0.365781776466343, -4.32873315150264, -4.79923284334735, -0.120167757070417,
                       0., 0., 0., 0., 0., 0.])

# from Kasia
parameters = np.array([-5.958861367832076E-002, -5.07768325945268, -5.43985445526999, -2.97645201029034, 1.17511473252137, -0.922807897798853, 0.750973146639994, 0.225037241142981, 1.39651734399144, -0.467650293769481, -5.21767471324313, -5.41764380468383, -0.199555140694319,     
                        1.964628549120861E-002, -4.18873315173927, -4.65965675237117, -3.26206262926160, 1.02005879779360, -1.22320015408732, 0.859943808292926, 0.238166335526434, 1.07920220903483, -0.365781776466343, -4.32873315150264, -4.79923284334735, -0.120167757070417, 
                       -1.10061550263889, -0.154678549691682, -0.500000000000000, 1.83181876021877, -0.329984241003078, 0., 0., 0., 0.])

material.set_parameters(*parameters)

parameters2 = np.array([-0.24998755827572447, -0.17005314428091503, -0.022919821025013193, 0.2104831697932253, 0.17466364449464572, -0.09354299555441778, -0.00914801256882744, 0.15135245018185967, 0.006216498983297014, 0.03826851846383894, -0.0724397359513074, -0.249695612098196, -0.24883397651837594, 
                       -0.04289658669470314, 0.062283559164763524, -0.14943193100982877, 0.04854128866309959, -0.24958568156502298, -0.07757581053010329, 0.24949296795927373, -0.06256542749566582, -0.1898572454900231, -0.10157010536962033, -0.24967265695393462, -0.2499078340659163, -0.24999523407767876, 
                       0.00021367075437650485, 0.5084574455873137, -0.013432654045392733, 0.01495566922888214, 0.18059766998185123, -0.07819324737746769, -0.07819324737746769, 0., 0.])
parameters = parameters2
material.update_parameters(*parameters)

# parameters = np.array([-0.09983598013125389, 0.04654703317547791, 0.09999176727770528, 0.08980310914053277, 0.054209020059768374, -0.07583695391399352, 0.08795713489725772, 0.09954923118056039, 0.09999520819063411, 0.06420849123419707, -0.09998993502045736, -0.09999275121501275, -0.09999744776075556, 
#                        -0.09999946263609742, -0.011012367063698472, -0.09977497067888252, 0.09999203489059871, -0.09999599084089607, -0.02450668782266034, 0.099999594497945, 0.01696419885310237, 0.09999901225285218, 0.09997243730346958, -0.03773067293983372, 0.012537944739452755, -0.09987616291611341, 
#                        0.09786747200360431, -1.5806129936095958, 0.04223882902123956, 0.6304004044093547, 0.37407900006562195, 0.5728972486613119])
# material.update_parameters(*parameters)

evo_search.search_for_best()
