import numpy as np
import numba as nb
from typing import Any, Dict, List, Tuple

class CMA_ES_algorithm:
    def __init__(self, population_size:int, parameters_size:int, elite_size:int, mean:float, sigma:float, mean_max:float=None, mean_min:float=None, sigma_max:float=None, sigma_min:float=None, alpha_cov:float=2.0, is_clipped:bool=False, extra_info:Dict[str, Any] = None):  

        self.N:int = parameters_size
        self.lambda_:int = population_size
        self.population_size:int = population_size
        self.parameters_size:int = parameters_size
        self.elite_size:int = np.floor(min(elite_size, self.lambda_)).astype(np.int64)
        self.generations:int = 0
        self.is_clipped:bool = is_clipped

        # 1 - Dynamic parameters (main) will be updated/optimized during the algorithm
        self.mean:np.ndarray = np.full(self.N, mean, dtype=np.float32)
        self.sigma:np.ndarray = np.full(self.N, sigma, dtype=np.float32)
        self.sigma_init:np.ndarray = np.full(self.N, sigma, dtype=np.float32)

        self.B:np.ndarray = np.identity(self.N, dtype=np.float32)
        self.D:np.ndarray = np.ones(self.N, dtype=np.float32)
        self.gaussian:np.ndarray = np.random.randn(self.lambda_, self.N).astype(np.float32) 
        self.covariance:np.ndarray = self.B @ np.diag(self.D**2) @ self.B.T
        self.eigeneval:int = 0 # from paper

        # 1.1 - Dynamic parameters (intermediary)
        self.covariance_path:np.ndarray = np.zeros(self.N) # p_c
        self.sigma_path:np.ndarray = np.zeros((1, self.N), dtype=np.float32) # p_sigma
        self.heaviside_sigma:int = 0 # not finished


        # 2 - Static parameters
        self.mean_max:float = mean_max
        self.mean_min:float = mean_min
        self.sigma_max:float = sigma_max
        self.sigma_min:float = sigma_min
        self.expectation_norm = self.__expected_norm(num_samples=10_000, dim=self.N) # E[||N(0, I)||]

        # 3 - Selection parameters
        self.mu:int = np.floor(min(elite_size, self.lambda_)).astype(np.int32) # mu = min(elite_size, lambda)
        self.weighted_sum:np.ndarray = np.log(self.mu + 1/2) - np.log(np.arange(1, self.mu + 1)) # w_i = log(mu + 1/2) - log(1:mu)
        self.weighted_sum:np.ndarray = np.array([self.weighted_sum / np.sum(self.weighted_sum)], dtype=np.float32) # w_i = w_i / sum(w_i)
        self.weighted_sum_T:np.ndarray = self.weighted_sum.T
        self.mu_eff:float = np.sum(self.weighted_sum)**2 / np.sum(self.weighted_sum**2) # equivalent to mu_eff = 1 / sum(w_i^2) if sum(w_i) = 1
        # print("weighted_sum:", self.weighted_sum, "sum:", np.sum(self.weighted_sum))
        # print("mu_eff:", self.mu_eff)

        # 4 - Adaptation parameters setting:
        self.alpha_cov:float = alpha_cov # choosed from paper -> can be possibly decrease on noisy functions/problems
        self.mean_coef:float = 1 # c_m = 1
        self.cumulation_coef:float = (4 + self.mu_eff/self.N) / (self.N + 4 + 2 * self.mu_eff/self.N) # c_c = (4 + mu_eff/N) / (N + 4 + 2 * mu_eff/N)
        self.sigma_coef:float = (self.mu_eff + 2) / (self.N + self.mu_eff + 5) # c_s = (mu_eff + 2) / (N + mu_eff + 5)
        self.rank_one_coef:float = alpha_cov / ((self.N + 1.3)**2 + self.mu_eff) # c_1 = 2 / ((N + 1.3)^2 + mu_eff)
        self.rank_mu_coef:float = min(1 - self.rank_one_coef, alpha_cov * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.N + 2)**2 + alpha_cov * self.mu_eff/2)) # c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((N + 2)^2 + 2 * mu_eff/2))
        self.dampings_coef:float = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.N + 1)) - 1) + self.sigma_coef # d = 1 + 2 * max(0, sqrt((mu_eff - 1) / (N + 1)) - 1) + c_s
        # print("cumulation_coef:", self.cumulation_coef, "sigma_coef:", self.sigma_coef, "rank_one_coef:", self.rank_one_coef, "rank_mu_coef:", self.rank_mu_coef, "dampings_coef:", self.dampings_coef)

        # 5 - popluation parameters
        self.population_param:np.ndarray = np.zeros((self.lambda_, self.N), dtype=np.float32)
        self.population_param = self.__update_parameters(self.mean, self.sigma, self.covariance, self.gaussian) # x_i = m + sigma * N(0, C) (Eq. 5)

        # 6 - Extra info
        self.best_fitness:float = None
        self.best_param:np.ndarray = None
        self.current_best_fitness:float = None
        self.current_best_param:np.ndarray = None

    def get_parameters(self) -> np.ndarray:
        return self.population_param

    def result(self) -> Tuple[np.ndarray, float, np.ndarray, float]:
        return self.best_param, self.best_fitness, self.current_best_param, self.current_best_fitness
    
    def __update_best(self, elites_indexes:np.ndarray, fitnesses:np.ndarray):
        self.current_best_fitness = fitnesses[elites_indexes[0]]
        self.current_best_param = self.population_param[elites_indexes[0]]
        if self.best_fitness is None or self.current_best_fitness > self.best_fitness:
            self.best_fitness = self.current_best_fitness
            self.best_param = self.current_best_param
        


    # @np.errstate(divide="raise", invalid="raise") # raise NumPy errors (on divide and invalid)
    def update(self, elites_indexes:np.ndarray, fitnesses:np.ndarray) -> np.ndarray:

        self.generations += 1
        elites_param:np.ndarray = self.population_param[elites_indexes[:self.mu]] # x_i... (elites population parameters)[:mu]
        self.__update_best(elites_indexes, fitnesses)

        # 1 - Update mean
        mean_new:np.ndarray = self.__update_mean(self.mean, self.mean_coef, self.weighted_sum_T, elites_param) # m^(g+1) = m + cm * sum(w_i * (x_i - m)) (Eq. 9)
        
        # 2 - Update cumulation path
        self.sigma_path, self.heaviside_sigma, self.covariance_path = self.__update_cumulation_path(
                                                                                                # 1 paths
                                                                                                self.sigma_path,
                                                                                                self.covariance_path,
                                                                                                # 2 coefficients
                                                                                                self.mean_coef,
                                                                                                self.sigma_coef,
                                                                                                self.cumulation_coef,
                                                                                                # 3 - mean
                                                                                                self.mean,
                                                                                                mean_new,
                                                                                                # 4 - covariance
                                                                                                self.B,
                                                                                                self.D,
                                                                                                self.sigma,
                                                                                                # 5 - others
                                                                                                self.parameters_size,
                                                                                                self.mu_eff,
                                                                                                self.generations,
                                                                                                self.lambda_,
                                                                                                self.expectation_norm,
                                                                                                )
        
        # 3 - Update step size σ
        self.sigma = self.__update_step_size_sigma(
                                                    self.sigma, 
                                                    self.sigma_coef, 
                                                    self.sigma_path, 
                                                    self.dampings_coef, 
                                                    self.expectation_norm,
                                                    ) # σ^(g+1) = σ * exp(cs/d * (||p_sigma^(g+1)|| / E[||N(0, I)||] - 1)) (Eq. 37)
        
        # 4 - Clip mean and sigma (optional)
        if self.is_clipped == True:
            mean_new, self.sigma = self.__clip_mean_sigma(mean_new, self.sigma, self.mean_max, self.mean_min, self.sigma_max, self.sigma_min)

        # 5 - Update covariance matrix C
        self.covariance:np.ndarray = self.__update_covariance_matrix_paper(
                                                    self.covariance,
                                                    self.covariance_path,
                                                    self.mean,
                                                    self.sigma,
                                                    self.heaviside_sigma,
                                                    self.cumulation_coef,
                                                    self.rank_mu_coef,
                                                    self.rank_one_coef,
                                                    self.weighted_sum,
                                                    elites_param,
                                                    self.parameters_size,
                                                    ) # C^(g+1) = (1 - c1 - cm) * C^g + c1 * p_c^(g+1) * p_c^(g+1)^T + cm * sum(w_i * (x_i - m) * (x_i - m)^T) (Eq. 30)

        
        # 6 - Update parameters
        self.mean:np.ndarray = mean_new
        self.population_param = self.__update_parameters(self.mean, self.sigma, self.covariance, self.gaussian) # x_i = m^(g+1) + σ^(g+1) * N(0, C^(g+1)) (Eq. 5)
        self.gaussian:np.ndarray = np.random.randn(self.lambda_, self.N).astype(np.float32) 

        # 7 - Update eigendecomposition
        # if self.generations - self.eigeneval > self.lambda_ / (self.rank_mu_coef * self.N / 10):
        if self.generations - self.eigeneval > self.N / 2:
            self.eigeneval = self.generations
            self.covariance = np.triu(self.covariance) + np.triu(self.covariance, 1).T # C = triu(C) + triu(C, 1)^T
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance) # eigh est pour les matrices symétriques/hermitiennes
            eigenvalues[eigenvalues<0.000001] = 0.000001 # avoid negative or null eigenvalues for sqrt
            self.D = np.sqrt(eigenvalues)
            self.B = eigenvectors

        # 8 - Escape flat fitness
        if np.max(fitnesses) - np.min(fitnesses) < 1e-10 or np.mean(self.sigma) < 0.01:
            self.sigma *= np.exp(0.2 + self.sigma_coef / self.dampings_coef)
            # self.sigma *= 1.2

        return self.population_param

    @staticmethod
    def __clip_mean_sigma(mean:np.ndarray, sigma:np.ndarray, mean_max:float, mean_min:float, sigma_max:float, sigma_min:float) -> Tuple[np.ndarray, np.ndarray]:
        # I did this in case you want only clip one side of the mean or sigma (e.g just a min or max or both)
        # if mean_max is not None: mean[mean > mean_max] = mean_max
        # if mean_min is not None: mean[mean < mean_min] = mean_min
        # if sigma_max is not None: sigma[sigma > sigma_max] = sigma_max
        # if sigma_min is not None: sigma[sigma < sigma_min] = sigma_min
        np.clip(mean, mean_min, mean_max, out=mean)
        np.clip(sigma, sigma_min, sigma_max, out=sigma)
        # print("mean",mean, "sigma", sigma, "mean_max:", mean_max, "mean_min:", mean_min, "sigma_max:", sigma_max, "sigma_min:", sigma_min)
        return mean, sigma

    @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def __update_parameters(mean:np.ndarray, sigma:np.ndarray, covariance:np.ndarray, gaussian:np.ndarray) -> np.ndarray:
        return mean + sigma * np.dot(gaussian, covariance) # x = m + sigma * N(0, C) (Eq. 5)
    
    def __expected_norm(self, num_samples:int=10000, dim:int=2):
        samples:np.ndarray = np.random.randn(num_samples, dim)  # Génère des échantillons à partir de N(0,I)
        norms:np.ndarray = np.linalg.norm(samples, axis=1)      # Calcule la norme de chaque échantillon
        return norms.mean() 

    @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def __update_mean(mean:np.ndarray, mean_coef:float, weigthed_sum:np.ndarray, param_population:np.ndarray) -> np.ndarray:
        return mean + mean_coef * np.sum(weigthed_sum * (param_population - mean), axis=0) # m + cm * sum(w_i * (x_i - m)) (Eq. 9) -> current generation mean (not next generation mean)
        
    @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def __update_cumulation_path(
                            # 1 paths
                            sigma_path:np.ndarray, 
                            covariance_path:np.ndarray,
                            # 2 coefficients
                            mean_coef:float,
                            sigma_coef:float,
                            cumulation_coef:float,
                            # 3 - mean
                            mean:np.ndarray,
                            mean_new:np.ndarray,
                            # 4 - covariance
                            B:np.ndarray,
                            D:np.ndarray,
                            sigma:np.ndarray,
                            # 5 - others
                            parameters_size:int,
                            mu_eff:float,
                            generation:int,
                            lambda_:int,
                            expectation_norm:float,
                            ):
        D[D==0] = 1e-10 # avoid division by 0
        C_minus_sqrt:np.ndarray = np.dot(np.dot(B, D**-1), B.T)
        z_mean:np.ndarray = ((mean_new - mean) / (mean_coef * sigma))

        # 1 - Update sigma path
        # from paper Eq 31
        sigma_path:np.ndarray = (1 - sigma_coef) * sigma_path + np.sqrt(sigma_coef * (2 - sigma_coef) * mu_eff) * (C_minus_sqrt * z_mean)
        

        # 2 - Update heaviside sigma
        heaviside_sigma:int = np.linalg.norm(sigma_path) / np.sqrt(1-(1-sigma_coef)**(2*generation/lambda_)) / expectation_norm < (1.4 + 2/parameters_size+1)
        if heaviside_sigma < 0 or heaviside_sigma > 1: raise Exception("heaviside_sigma must be 0 or 1")

        # 3 - Update covariance path
        # from paper Eq 24
        covariance_path:np.ndarray = (1 - cumulation_coef) * covariance_path + heaviside_sigma * np.sqrt(cumulation_coef*(2-cumulation_coef)*mu_eff) * z_mean

        return sigma_path, heaviside_sigma, covariance_path

    @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def __update_step_size_sigma(
                                    sigma:np.ndarray, # σ
                                    sigma_coef:float, # cs
                                    sigma_path:np.ndarray, # p_sigma
                                    dampings_coef:float, # d_sigma,
                                    expectation_norm:float # E[||N(0, I)||]
                                 ) -> np.ndarray:

        return sigma * np.exp((sigma_coef / dampings_coef) * ((np.linalg.norm(sigma_path) / expectation_norm) - 1)) # σ * exp(cs/d_sigma * (||p_sigma^(g+1)|| / E[||N(0, I)||] - 1)) (Eq. 37)
        
    @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def __update_covariance_matrix_paper( 
                                    cov_matrix:np.ndarray, 
                                    cov_path, # p_c,
                                    mean:np.ndarray, # m[:mu]
                                    sigma:np.ndarray, # σ
                                    heaviside_sigma:int, # h_σ
                                    cumulation_coef:float, # c_c
                                    rank_mu_coef:float, # c_mu
                                    rank_one_coef:float, # c1
                                    weigthed_sum:np.ndarray, # w_i...
                                    elites_param:np.ndarray, # x_i... (elites population parameters)[:mu]
                                    parameters_size:int, # N
                                   ) -> np.ndarray:

        y_i:np.ndarray = (elites_param - mean) / sigma # (x_i - m) / sigma (Eq. 11)
        weighted_sum_T:np.ndarray = weigthed_sum.T

        # 1 - Rank-one update
        rank_one_update:np.ndarray = rank_one_coef * np.outer(cov_path, cov_path)

        # 2 - Rank-mu update
        weighted_y_i:np.ndarray = np.zeros((len(weigthed_sum), parameters_size, parameters_size))
        for i in range(len(weigthed_sum)):
            weighted_y_i[i] = weighted_sum_T[i][0] * np.outer(y_i[i], y_i[i].T)
        rank_mu_update:np.ndarray = rank_mu_coef * np.sum(weighted_y_i, axis=0)

        # 3 - Covariance matrix update
        delta_heaviside_sigma:float = (1 - heaviside_sigma) * cumulation_coef * (2 - cumulation_coef) # <= 1
        covariance_history_info:np.ndarray = (1 + (rank_one_coef * delta_heaviside_sigma) - rank_one_coef - (rank_mu_coef * np.sum(weighted_sum_T, axis=0)[0])) * cov_matrix # exponential smoothing (1 - c1 - cm) * C^g (Eq. 29)

        # print(covariance_history_info.dtype, rank_one_update.dtype, rank_mu_update.dtype)
        # exit()

        return covariance_history_info +  rank_one_update + rank_mu_update # C^(g+1) = (1 - c1 - cm) * C^g + c1 * p_c^(g+1) * p_c^(g+1)^T + cm * sum(w_i * (x_i - m) * (x_i - m)^T) (Eq. 30)

    @staticmethod
    # @nb.njit(cache=True, fastmath=True, nogil=True)
    def __cma_es_all(
                    # 1 - Covariance parameters
                    elites_param:np.ndarray,
                    lambda_:int,
                    parameters_size:int,
                    B:np.ndarray,
                    D:np.ndarray,
                    
                    # dynamic parameters
                    covariance:np.ndarray,
                    mean:np.ndarray, 
                    sigma:np.ndarray,
                    covariance_path:np.ndarray,
                    sigma_path:np.ndarray,
                    

                    # coefficients
                    mean_coef:float,
                    sigma_coef:float,
                    rank_mu_coef:float,
                    cumulation_coef:float,
                    dampings_coef:float,
                    rank_one_coef:float,

                    # others
                    weighted_sum:np.ndarray,
                    mu_eff:float,
                    generation:int,
                    expectation_norm:float,
                    ):

        # 1 - Update mean
        mean_new = mean + mean_coef * np.sum(weighted_sum * (elites_param - mean), axis=0)

        # 2 - Update cumulation path
        C_minus_sqrt:np.ndarray = np.dot(np.dot(B, D**-1), B.T)
        z_mean:np.ndarray = ((mean_new - mean) / (mean_coef * sigma))

        # 2.1 - Update sigma path
        # from paper Eq 31
        sigma_path:np.ndarray = (1 - sigma_coef) * sigma_path + np.sqrt(sigma_coef * (2 - sigma_coef) * mu_eff) * (C_minus_sqrt * z_mean)
        

        # 2.2 - Update heaviside sigma
        heaviside_sigma:int = np.linalg.norm(sigma_path) / np.sqrt(1-(1-sigma_coef)**(2*generation/lambda_)) / expectation_norm < (1.4 + 2/parameters_size+1)
        if heaviside_sigma < 0 or heaviside_sigma > 1: raise Exception("heaviside_sigma must be 0 or 1")

        # 2.3 - Update covariance path
        # from paper Eq 24
        covariance_path:np.ndarray = (1 - cumulation_coef) * covariance_path + heaviside_sigma * np.sqrt(cumulation_coef*(2-cumulation_coef)*mu_eff) * z_mean

        # 3 - Update step size σ
        sigma = sigma * np.exp((sigma_coef / dampings_coef) * ((np.linalg.norm(sigma_path) / expectation_norm) - 1)) # σ * exp(cs/d_sigma * (||p_sigma^(g+1)|| / E[||N(0, I)||] - 1)) (Eq. 37)
        

        # 4 - Update covariance matrix C
        y_i:np.ndarray = (elites_param - mean) / sigma # (x_i - m_old) / sigma (Eq. 11)
        # weighted_sum_T:np.ndarray = weigthed_sum.T

        # 4.1 - Rank-one update
        rank_one_update:np.ndarray = rank_one_coef * np.outer(covariance_path, covariance_path)

        # 4.2 - Rank-mu update
        weighted_y_i:np.ndarray = np.zeros((len(weighted_sum), parameters_size, parameters_size))
        for i in range(weighted_sum.size):
            weighted_y_i[i] = weighted_sum[i][0] * np.outer(y_i[i], y_i[i].T)
        rank_mu_update:np.ndarray = rank_mu_coef * np.sum(weighted_y_i, axis=0)

        # 4.3 - Covariance matrix update
        delta_heaviside_sigma:float = (1 - heaviside_sigma) * cumulation_coef * (2 - cumulation_coef) # <= 1
        covariance_history_info:np.ndarray = (1 + (rank_one_coef * delta_heaviside_sigma) - rank_one_coef - (rank_mu_coef * np.sum(weighted_sum, axis=0)[0])) * covariance # exponential smoothing (1 - c1 - cm) * C^g (Eq. 29)

        # print(covariance_history_info.dtype, rank_one_update.dtype, rank_mu_update.dtype)
        # exit()
        covariance:np.ndarray = covariance_history_info +  rank_one_update + rank_mu_update # C^(g+1) = (1 - c1 - cm) * C^g + c1 * p_c^(g+1) * p_c^(g+1)^T + cm * sum(w_i * (x_i - m) * (x_i - m)^T) (Eq. 30)

        # population_parameter = mean + sigma * np.dot(gaussian, covariance) # x = m + sigma * N(0, C) (Eq. 5)
        return covariance, mean_new, sigma, covariance_path, sigma_path

    def reset_from_current_mean(self, sigma_init_coef:float=2.0):
        self.mean = self.mean
        self.sigma = np.full(self.N, self.sigma_init*sigma_init_coef, dtype=np.float32) # self.sigma_init * sigma_init_coef, for the first generation
        self.B = np.identity(self.N, dtype=np.float32)
        self.D = np.ones(self.N, dtype=np.float32)
        self.gaussian = np.random.randn(self.lambda_, self.N).astype(np.float32) 
        self.covariance = self.B @ np.diag(self.D**2) @ self.B.T
        self.eigeneval = 0 # from paper

        self.covariance_path:np.ndarray = np.zeros(self.N) # p_c
        self.sigma_path:np.ndarray = np.zeros((1, self.N), dtype=np.float32) # p_sigma
        self.heaviside_sigma:int = 0 # not finished