import utils_evo
import utils_tb
import datetime
from numpy.random import uniform

# setup t-b model and eigensolver
#material = utils_tb.TMDCmaterial(0.316, 0.158,
#                                 -3.390, 1.100, -1.100, 0.760, 0.270, 1.190, -0.830,
#                                 -0.030, -3.360, -4.780)  # in eV

material = utils_tb.TMDCmaterial(0.388, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.)

bound = 5.
#material.update_parameters(*uniform(-bound, bound, 10))
lattice = utils_tb.Lattice()
lattice.select_k_indices(distance=3)
model = utils_tb.BandModel(material, lattice)
eigen_solver = utils_tb.EigenSolver(model)
#real_bands = utils_tb.load_data('./even_DFT.txt')[:, ::-1]
real_bands = utils_tb.load_np('./wsi2sb4_DFT_3_3.npy')
real_bands_sliced = real_bands[lattice.k_indices]

evo_search = utils_evo.EvoSearch(
    material=material,
    k_indices=lattice.k_indices,
    ks_indices=lattice.ks_indices,
    eigen_solver=eigen_solver,
    real_bands=real_bands,    
    bounds=[5.,],
    bands=["conductance", "coductance_and_valence", "all"],
    compostition_loss=[.5, .25, .0],
    strategy=["best1bin","best1exp",],
    pop_size=[503],  # 500
    tol=0.0001,
    mutation=[(0.05, 0.15)],
    recombination=[.7,],  # .7 | seems like >0.7 is doing worse
    metrics=["rmse"],
    max_iter=5000,
    workers=8,  # -1 = all cores
    no_params=12,
    results_path=f'./results/results_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.txt',
    plots_path=None
)

#print(evo_search._calculate_residuals([0.5303333767767483, 1.1139367006174299, -0.22029626391997187, 0.63512133230208, 0.2357301545544615, 0.2965495264896972, -0.1176609724634411, -0.15780539528891308, -0.14252978069227137, -0.25214770221155436, 1.2997392133678685, -3.1852554573522593], "coductance_and_valence", "rmse" ))
#print(evo_search._calculate_residuals([-0.5652248235471982, -1.0230287615236027, -0.004144542454125683, 0.2436480099993278, 0.17519028277983584, 0.5512095930801296, -0.12507917536939017, 0.005127585989217813, -0.3688908301602572, -0.2977195613062045, 1.7874994849771866, -2.3006786044193563], "coductance_and_valence", "rmse" ))
#print(evo_search._calculate_residuals([-1.8238384477373688, -1.0089858116474357, 0.12352674166638113, 0.021410759993732657, -1.1106091991084863, 0.18907145481738855, 0.2147975359099036, -2.3554756963922023, 1.1581041926697713, -4.999999440525821, 1.9365082609871396, -0.36707593571303426], "coductance_and_valence", "rmse" ))

evo_search.search_for_best()
