import json
import os.path
import glob
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from numpy.random import uniform

from utils_tb import AtomicUnits, Plotting

au = AtomicUnits()

class Methods:
    def __init__(self, material, eigensolver, real_bands, bounds):
        self.material = material
        self.eigensolver = eigensolver
        self.real_bands = real_bands
        self.bounds = bounds

    def evo(self, residuals, pop_size):
        return differential_evolution(residuals,
                                      bounds=self.bounds,
                                      strategy='best1bin',
                                      args=(self.material, self.eigensolver, self.real_bands),
                                      popsize=pop_size,
                                      mutation=(.5, 1.5),
                                      recombination=0.7,  # .7
                                      maxiter=10000,  # 10000
                                      init='random',
                                      workers=-1,  # 8
                                      disp=True)


class SlicedBands:
    def __init__(self, real_bands, predicted_bands, bands):
        self.real_bands = real_bands
        self.predicted_bands = predicted_bands
        
        self.bands = bands

    def get_sliced_bands(self):
        if self.bands == "coductance":
            self.real_bands = self.real_bands[:, 11:15]
            self.predicted_bands = self.predicted_bands[:, 17:21]
            return self
        if self.bands == "coductance_reduced":
            self.real_bands = self.real_bands[:, 11:13]
            self.predicted_bands = self.predicted_bands[:, 17:19]
            return self
        if self.bands == "coductance_and_valence":
            self.real_bands = self.real_bands[:, 7:15]
            self.predicted_bands = self.predicted_bands[:, 13:21]
            return self
        if self.bands == "coductance_and_valence_reduced":
            self.real_bands = self.real_bands[:, 9:13]
            self.predicted_bands = self.predicted_bands[:, 15:19]
            return self
        if self.bands == "valence":
            self.real_bands = self.real_bands[:, 7:11]
            self.predicted_bands = self.predicted_bands[:, 13:17]
            return self
        if self.bands == "all":
            return self
        return self

    def get_bands(self):
        return self.real_bands, self.predicted_bands


class EvoSearch:
    def __init__(self, material, k_indices, ks_indices, eigen_solver, real_bands, real_spins, bounds=None, bands='all', compostition_loss=None, inter_intra_ratio_bounds=None,
                 strategy='best1bin', pop_size=None, tol=0.01, mutation=None, recombination=None, max_iter=1000,
                 workers=-1, disp=True, no_params=10, results_path="results.txt", plots_path="plots", metrics=None):
        self.material = material
        self.k_indices = k_indices
        self.ks_indices = ks_indices
        self.eigen_solver = eigen_solver
        self.real_bands = real_bands
        self.real_spins = real_spins
        self.bounds = bounds
        self.bands = bands
        self.compostition_loss = compostition_loss
        self.inter_intra_ratio_bounds = inter_intra_ratio_bounds
        self.strategy = strategy
        self.pop_size = pop_size
        self.tol = tol
        self.mutation = mutation
        self.recombination = recombination
        self.max_iter = max_iter
        self.workers = workers
        self.disp = disp
        self.results_path = results_path
        self.plots_path = plots_path
        self.metrics = metrics
        self.no_params = no_params
        self.already_searched = self.load_already_searched()
        self.plotting = Plotting(self.eigen_solver.model, self.plots_path)
        self._init_results_file()

    def search_for_best(self):
        if self.bounds is None:
            print("no bounds specified")
            return
        for s in self.strategy:
            for rb in self.inter_intra_ratio_bounds:
                for b in self.bounds:
                    bounds = [(-b*rb, b*rb) for _ in range(self.no_params[0])] + [(-b, b) for _ in range(self.no_params[1])]
                    for p in self.pop_size:
                        for m in self.mutation:
                            for r in self.recombination:
                                for bands in self.bands:
                                    for c in self.compostition_loss:
                                        for metric in self.metrics:
                                            self.material.update_parameters(*np.concatenate([uniform(-b*rb, b*rb, self.no_params[0]),uniform(-b, b, self.no_params[1])]))
                                            name = f"ek_target_evo_{s}_{p}_{m}_{r}_bands_{bands}_bound_{b}_ratio_bound_{rb}_comp_{c}_metric_{metric}"
                                            if name not in self.already_searched:
                                                print(f"searching for: {name}")
                                                output = self.evo(
                                                    bounds=bounds,
                                                    strategy=s,
                                                    pop_size=p,
                                                    tol=self.tol,
                                                    mutation=m,
                                                    recombination=r,
                                                    bands=bands,
                                                    compostition_loss=c,
                                                    metric=metric,
                                                )
                                                self.material.update_parameters(*output.x)
                                                fitted_bands, fitted_spins, _  = self.eigen_solver.solve_BZ_path(get_spin=True)
                                                tb_params_str = ", ".join(str(p) for p in output.x)
                                                sliced_real_bands, sliced_predicted_bands = SlicedBands(
                                                    real_bands=self.real_bands,
                                                    predicted_bands=fitted_bands,
                                                    bands=b,
                                                ).get_sliced_bands().get_bands()
                                                result = {
                                                    "name": name,
                                                    "rmse": self._calculate_rmse(self.real_bands, fitted_bands),
                                                    "fit_rmse": self._calculate_rmse(sliced_real_bands, sliced_predicted_bands),
                                                    "tb_params": f"[{tb_params_str}]",
                                                }
                                                self._append_result_to_file(result)
                                                #self.plotting.plot_Ek_output_target(self.real_bands, fitted_bands, name)
                                                real_spins = np.ones_like(fitted_spins)
                                                self.plotting.plot_Ek_output_target_s([self.real_bands, real_spins], [fitted_bands, fitted_spins], name)
                                            else:
                                                print(f'skipped search for: {name}')
        self._close_results_file()

    def evo(self, bounds, strategy, pop_size, tol, mutation, recombination, bands="valence", compostition_loss=.5, metric="mae"):
        return differential_evolution(
            self._calculate_residuals,
            bounds=bounds,
            strategy=strategy,
            args=(bands, compostition_loss, metric),
            popsize=pop_size,
            tol=tol,
            mutation=mutation,
            recombination=recombination,  # .7
            maxiter=self.max_iter,  # 10000
            init='random',
            workers=self.workers,  # 8
            disp=self.disp
        )

    def _calculate_residuals(self, parameters, bands, compostition_loss, metric):
        self.material.update_parameters(*parameters)
        #predicted_bands = self.eigen_solver.solve_BZ_path()
        predicted_bands = self.eigen_solver.solve_at_points(self.eigen_solver.model.BZ_path[self.k_indices])
        real_bands, predicted_bands = SlicedBands(
            real_bands=self.real_bands[self.k_indices],
            predicted_bands=predicted_bands,
            bands=bands
        ).get_sliced_bands().get_bands()
        spin_err = 0. 
        comp_err = 0.
        for ks_idx in self.ks_indices:
            _, spins, comps = self.eigen_solver.solve_k(self.eigen_solver.model.BZ_path[ks_idx], get_spin=True)
            #spin_err += sqrt(mean_squared_error(self.real_spins[ks_idx,7:15], spins[13:21]))
            spin_err += sqrt(mean_squared_error(self.real_spins[ks_idx,9:13], spins[15:19]))
            # to be implemented:
            #comp_err += 1. if np.sum(np.array([comps[0,4]+comps[2,4], comps[1,5], comps[1,6], comps[0,7]+comps[2,7]]) - np.array([1., 1., 1., 1.])*compostition_loss) < 0. else 0.
        if metric == "rmse":
            return sqrt(mean_squared_error(real_bands, predicted_bands)) + spin_err + comp_err
        if metric == "mae":
            return mean_absolute_error(real_bands, predicted_bands) + spin_err + comp_err
        return np.sum(np.abs(real_bands - predicted_bands))

    def _calculate_rmse(self, real_bands, predicted_bands):
        return sqrt(mean_squared_error(real_bands, predicted_bands))

    def _calculate_mae(self, real_bands, predicted_bands):
        return mean_absolute_error(real_bands, predicted_bands)

    def _init_results_file(self):
        with open(self.results_path, mode="w") as f:
            f.write("[\n")

    def _close_results_file(self):
        with open(self.results_path, mode="a") as f:
            f.write("\n]")

    def _append_result_to_file(self, result):
        with open(self.results_path, mode="a") as f:
            json.dump(result, f)
            f.write(",\n")

    def generate_bands_array(self, x):
        grid_k = self.eigen_solver.model.BZ_path
        target = np.array(self.real_bands)
        output = np.array(x)
        y_t, y_o, x_axis = None, None, None
        for band_idx in range(target.shape[1]):
            x_axis = (grid_k[:, 0] + grid_k[:, 1]) / au.Ah
            y_t = target[:, band_idx]
            y_o = output[:, band_idx]
        return y_t, y_o, x_axis

    def load_already_searched(self):
        return [name.split(".png")[0].replace(f"./plots", "") for name in glob.glob(glob.escape(f"./plots")+ "/*.png")]
