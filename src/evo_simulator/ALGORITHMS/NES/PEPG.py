import numpy as np
from evo_simulator.ALGORITHMS.NES.Optimizer import Optimizer, Adam, SGD, BasicSGD
from evo_simulator.ALGORITHMS.NES.fitness_shaping import compute_centered_fitness_ranks, compute_parameters_decay
from typing import Tuple

# adopted from:
#https://github.com/hardmaru/estool/blob/master/es.py

class PEPG:
  def __init__(self,
               # 1. Major params
               population_size:int,               # population size
               num_params:int,                    # number of model parameters
               elite_ratio:float = 0.0,           # if > 0, then ignore learning_rate
               # 2. Mu params
               mu_init:float=0.0,                 # initial start point
               # 2. Sigma params
               sigma_init:float=0.10,             # initial standard deviation
               sigma_alpha:float=0.20,            # learning rate for standard deviation
               sigma_decay:float=0.999,           # anneal standard deviation
               sigma_limit:float=0.01,            # stop annealing if less than this
               sigma_max_change:float=0.2,        # clips adaptive sigma to 20%
               # 3. Learning rate params
               learning_rate:float=0.01,          # learning rate for standard deviation
               learning_rate_decay:float= 0.9999, # annealing the learning rate
               learning_rate_limit:float= 0.01,   # stop annealing learning rate
               # 4. Other params
               average_baseline:bool=True,        # set baseline to average of batch
               parameters_decay:float=0.01,       # solution parameters decay coefficient
               rank_fitness:bool=True,            # use rank rather than fitness numbers
               forget_best:bool=True,             # don't keep the historical best solution
               optimizer_name:str = "Adam"        # choose optimizer
               ):            

    self.population_size:int = population_size
    self.num_params:int = num_params

    self.sigma_init:float = sigma_init
    self.sigma_alpha:float = sigma_alpha
    self.sigma_decay:float = sigma_decay
    self.sigma_limit:float = sigma_limit
    self.sigma_max_change:float = sigma_max_change

    self.learning_rate:float = learning_rate
    self.learning_rate_decay:float = learning_rate_decay
    self.learning_rate_limit:float = learning_rate_limit

    self.average_baseline:bool = average_baseline
    if self.average_baseline == True:
      if (self.population_size % 2 == 1): raise "Population size must be even"
      self.batch_size = int(self.population_size / 2)
    else:
      if (self.population_size % 2 == 0): raise "Population size must be odd"
      self.batch_size = int((self.population_size - 1) / 2)

    # option to use greedy es method to select next mu, rather than using drift param
    self.elite_ratio:float = elite_ratio
    self.elite_popsize:int = int(self.population_size * self.elite_ratio)
    self.use_elite:bool = False if self.elite_popsize <= 0 else True

    self.rank_fitness:bool = rank_fitness # use rank (linear fitness) rather than (raw) fitness (fitness shaping) - supposed to avoid dominance of a few good solutions
    self.forget_best = True if self.rank_fitness == True else forget_best # always forget the best one if rank is used

    self.batch_fitness:np.ndarray = np.zeros(self.batch_size * 2)
    self.mu:np.ndarray = np.full(self.num_params, mu_init)
    self.sigma:np.ndarray = np.ones(self.num_params) * self.sigma_init
    self.curr_best_mu:np.ndarray = np.zeros(self.num_params)
    self.best_mu:np.ndarray = np.zeros(self.num_params)
    self.best_fitness:float = 0
    self.first_interation:bool = True
    self.parameters_decay:float = parameters_decay

    # choose optimizer
    self.optimizer_dict = {"Adam": Adam, "SGD": SGD, "BasicSGD": BasicSGD}
    if optimizer_name not in self.optimizer_dict: raise ValueError("Optimizer name not found. Choose from: ", self.optimizer_dict.keys())
    self.optimizer:Optimizer = self.optimizer_dict[optimizer_name](self, learning_rate)


  def get_parameters(self):
    '''returns a list of parameters'''
    if self.first_interation: return self.__update_population()
    return self.solutions

  def update(self, elites_indexes:np.ndarray, fitnesses:np.ndarray) -> np.ndarray:
    # input must be a numpy float array
    assert(len(fitnesses) == self.population_size), "Inconsistent fitnesses size reported."

    # 1 - Fitness shaping
    fitnesses:np.ndarray = self.__fitness_shaping(fitnesses)

    # 2 - Get baseline
    baseline, fitnesses = self.__update_baseline(fitnesses)
      
    # 3 - Get population indexes will be used to update parameters (mean and sigma)
    elites_indexes:np.ndarray = np.argsort(fitnesses)[::-1][0:self.elite_popsize] if self.use_elite else np.argsort(fitnesses)[::-1]

    # 4 - Update best
    self.__update_best(elites_indexes, fitnesses, baseline)

    # 5 - Update mean
    self.mu = self.__update_mean(elites_indexes, fitnesses)

    # 6 - Update sigma
    self.sigma = self.__update_sigma(fitnesses, baseline)
    
    # 7 - Update learning rate
    if (self.learning_rate_decay < 1 and self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay
    
    return self.__update_population()

  def __update_population(self):
    # 1 - Antithetic sampling -> Positive and negative perturbations of the same magnitude
    self.epsilon:np.ndarray = np.random.randn(self.batch_size, self.num_params) * self.sigma.reshape(1, self.num_params)
    self.epsilon_full:np.ndarray = np.concatenate([self.epsilon, - self.epsilon]) # to have GOOD/better symmetry around mu
    
    # 2 - positive standard deviation
    if self.average_baseline:
      epsilon:np.ndarray = self.epsilon_full
    else:
      # first population is mu, then positive epsilon, then negative epsilon
      epsilon:np.ndarray = np.concatenate([np.zeros((1, self.num_params)), self.epsilon_full])

    # 3 - Get the solutions from the current mu and sigma distribution
    self.solutions = self.mu.reshape(1, self.num_params) + epsilon # episilon is the perturbation added to the mean
    return self.solutions
  
  def __update_mean(self, elites_indexes:np.ndarray, fitnesses:np.ndarray) -> np.ndarray:
    if self.use_elite: # move mean to the average of the elites idx means
      self.mu += self.epsilon_full[elites_indexes].mean(axis=0)
    else:
      rT = (fitnesses[:self.batch_size] - fitnesses[self.batch_size:])
      # print("fitnesses[:self.batch_size]:\n", fitnesses[:self.batch_size], "shape:", fitnesses[:self.batch_size].shape)
      # print("fitnesses[self.batch_size:]:\n", fitnesses[self.batch_size:], "shape:", fitnesses[self.batch_size:].shape)
      # print("rT:\n", rT, "shape:", rT.shape)
      change_mu:np.ndarray = np.dot(rT, self.epsilon) # Gradient of the mean (mu) J = rT * epsilon
      # print("self.epsilon:\n", self.epsilon, "shape:", self.epsilon.shape)
      # print("change_mu:\n", change_mu, "shape:", change_mu.shape)
      # exit()
      self.optimizer.stepsize:float = self.learning_rate
      update_ratio:float = self.optimizer.update(-change_mu) # adam, rmsprop, momentum, etc.
      #self.mu += (change_mu * self.learning_rate) # normal SGD method
    return self.mu

  def __update_sigma(self, fitnesses:np.ndarray, baseline) -> np.ndarray:
    if (self.sigma_alpha > 0):
      stdev_fitness:float = 1.0
      if self.rank_fitness == False:
        stdev_fitness = fitnesses.std()
      S = ((self.epsilon * self.epsilon - (self.sigma * self.sigma).reshape(1, self.num_params)) / self.sigma.reshape(1, self.num_params))
      fitness_avg:np.ndarray = (fitnesses[:self.batch_size] + fitnesses[self.batch_size:]) / 2.0
      rS:np.ndarray = fitness_avg - baseline
      delta_sigma = (np.dot(rS, S)) / (2 * self.batch_size * stdev_fitness)

      # adjust sigma according to the adaptive sigma calculation
      # for stability, don't let sigma move more than 10% of orig value
      change_sigma = self.sigma_alpha * delta_sigma
      change_sigma = np.minimum(change_sigma, self.sigma_max_change * self.sigma)
      change_sigma = np.maximum(change_sigma, - self.sigma_max_change * self.sigma)
      self.sigma += change_sigma

    if (self.sigma_decay < 1):
      self.sigma[self.sigma > self.sigma_limit] *= self.sigma_decay
    
    return self.sigma

  def __update_baseline(self, fitnesses:np.ndarray) -> Tuple[float, np.ndarray]:
    fitness_offset:int = 1
    if self.average_baseline:
      baseline:float = np.mean(fitnesses) # mean fitness baseline
      fitness_offset:int = 0
    else:
      baseline:float = fitnesses[0] # best fitness baseline
    fitnesses:np.ndarray = fitnesses[fitness_offset:] # in order to remove the baseline, the first fitnesses are not used if average_baseline is False
    return baseline, fitnesses

  def __fitness_shaping(self, fitnesses:np.ndarray) -> np.ndarray:
    if self.rank_fitness: # use rank rather than actual fitness (fitness shaping) - supposed to avoid dominance of a few good solutions
      fitnesses:np.ndarray = compute_centered_fitness_ranks(fitnesses)
    
    if self.parameters_decay > 0:
      l2_decay:np.ndarray = compute_parameters_decay(self.parameters_decay, self.solutions)
      fitnesses += l2_decay
    return fitnesses

  def __update_best(self, elites_indexes:np.ndarray, fitnesses:np.ndarray, baseline) -> np.ndarray:
    best_fitness = fitnesses[elites_indexes[0]]

    if (best_fitness > baseline or self.average_baseline):
      best_mu = self.mu + self.epsilon_full[elites_indexes[0]]
      best_fitness = fitnesses[elites_indexes[0]]
    else:
      best_mu = self.mu
      best_fitness = baseline

    self.curr_best_fitness = best_fitness
    self.curr_best_mu = best_mu

    if self.first_interation:
      self.sigma = np.ones(self.num_params) * self.sigma_init
      self.first_interation = False
      self.best_fitness = self.curr_best_fitness
      self.best_mu = best_mu
    else:
      if self.forget_best or (self.curr_best_fitness > self.best_fitness):
        self.best_mu = best_mu
        self.best_fitness = self.curr_best_fitness

  def rms_stdev(self):
    sigma = self.sigma
    return np.mean(np.sqrt(sigma*sigma))

  def current_param(self):
    return self.curr_best_mu

  def set_mu(self, mu):
    self.mu = np.array(mu)
  
  def best_param(self):
    return self.best_mu

  def result(self): # return best params so far, along with historically best_ever fitness, curr fitness, sigma
    return (self.best_mu, self.best_fitness, self.curr_best_fitness, self.sigma)
