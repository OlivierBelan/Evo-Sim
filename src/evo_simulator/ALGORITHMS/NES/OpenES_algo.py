import numpy as np
from evo_simulator.ALGORITHMS.NES.Optimizer import Optimizer, Adam, SGD, BasicSGD
from evo_simulator.ALGORITHMS.NES.fitness_shaping import compute_centered_fitness_ranks, compute_parameters_decay

# adopted from:
#https://github.com/hardmaru/estool/blob/master/es.py

class OpenES_algo:
  ''' Basic Version of OpenAI Evolution Strategies.'''
  def __init__(self,
              # 0. Major params
               population_size:int,               # population size
               num_params:int,                    # number of model parameters
               # 1. Mu params
               mu_init:float=0.0,                 # initial start point
               # 2. Sigma params
               sigma_init:float=0.1,              # initial standard deviation
               sigma_decay:float=0.999,           # anneal standard deviation
               sigma_limit:float=0.01,            # stop annealing if less than this
               # 3. Learning rate params
               learning_rate:float=0.01,          # learning rate for standard deviation
               learning_rate_decay:float= 0.9999, # annealing the learning rate
               learning_rate_limit:float= 0.001,  # stop annealing learning rate
               # 4. Other params
               antithetic:bool=False,             # whether to use antithetic sampling
               parameters_decay:float=0.01,       # parameters decay coefficient
               rank_fitness:bool=True,            # use rank rather than fitness numbers
               forget_best:bool=True,             # forget historical best
               optimizer_name:str = "Adam"        # choose optimizer
               ):            

    self.population_size:int = population_size
    self.num_params:int = num_params

    self.sigma:float = sigma_init
    self.sigma_decay:float = sigma_decay
    self.sigma_init:float = sigma_init
    self.sigma_limit:float = sigma_limit

    self.learning_rate:float = learning_rate
    self.learning_rate_decay:float = learning_rate_decay
    self.learning_rate_limit:float = learning_rate_limit

    self.antithetic:bool = antithetic
    if self.antithetic == True:
      if (self.population_size % 2 == 1): raise "Population size must be even"
      self.half_popsize = int(self.population_size / 2)

    self.reward:np.ndarray = np.zeros(self.population_size)
    # self.mu:np.ndarray = np.zeros(self.num_params)
    self.mu:np.ndarray = np.full(self.num_params, mu_init)
    self.best_mu:np.ndarray = np.zeros(self.num_params)
    self.best_fitness:float = 0.0
    self.first_interation:bool = True
    self.forget_best:bool = forget_best
    self.parameters_decay:float = parameters_decay

    self.rank_fitness = rank_fitness
    if self.rank_fitness == True: 
      self.forget_best = True # always forget the best one if we rank

    # choose optimizer
    self.optimizer_dict = {"Adam": Adam, "SGD": SGD, "BasicSGD": BasicSGD}
    if optimizer_name not in self.optimizer_dict: raise ValueError("Optimizer name not found. Choose from: ", self.optimizer_dict.keys())
    self.optimizer:Optimizer = self.optimizer_dict[optimizer_name](self, learning_rate)


  def get_parameters(self):
    if self.first_interation: return self.__update_population()
    return self.solutions


  def update(self, elites_indexes:np.ndarray, fitnesses:np.ndarray):
    # input must be a numpy float array
    assert(len(fitnesses) == self.population_size), "Inconsistent fitnesses size reported."

    # 0 - Fitness shaping    
    fitnesses:np.ndarray = self.__fitness_shaping(fitnesses)

    # 1 - Update best parameters
    self.__update_best(fitnesses)

    # 2 - Update mean
    self.mu = self.__update_mean(fitnesses)

    # 3 - Update sigma
    self.sigma = self.__update_sigma()

    # 4 - Update learning rate
    self.learning_rate = self.__update_learning_rate()

    return self.__update_population()

  def __update_population(self):
    '''returns a list of parameters'''
    # antithetic sampling
    if self.antithetic == True:
      self.epsilon_half:np.ndarray = np.random.randn(self.half_popsize, self.num_params)
      self.epsilon:np.ndarray = np.concatenate([self.epsilon_half, - self.epsilon_half])
    else:
      self.epsilon:np.ndarray = np.random.randn(self.population_size, self.num_params)

    self.solutions:np.ndarray = self.mu.reshape(1, self.num_params) + self.epsilon * self.sigma

    return self.solutions

  def __update_mean(self, fitnesses:np.ndarray) -> np.ndarray:
    # main bit:
    # standardize the rewards to have a gaussian distribution
    normalized_fitnesses = (fitnesses - np.mean(fitnesses)) / np.std(fitnesses)
    change_mu = 1./(self.population_size*self.sigma)*np.dot(self.epsilon.T, normalized_fitnesses)
    
    #self.mu += self.learning_rate * change_mu # Normal SGD

    self.optimizer.stepsize = self.learning_rate
    update_ratio = self.optimizer.update(-change_mu) # Adam, SGD, BasicSGD ....
    return self.mu

  def __update_sigma(self) -> float:
    # adjust sigma according to the adaptive sigma calculation
    if (self.sigma > self.sigma_limit):
      self.sigma *= self.sigma_decay
    return self.sigma

  def __update_learning_rate(self) -> float:
    if (self.learning_rate > self.learning_rate_limit):
      self.learning_rate *= self.learning_rate_decay
    return self.learning_rate
      
  def __fitness_shaping(self, fitnesses:np.ndarray) -> np.ndarray:
    if self.rank_fitness:
      fitnesses = compute_centered_fitness_ranks(fitnesses)
    
    if self.parameters_decay > 0:
      l2_decay = compute_parameters_decay(self.parameters_decay, self.solutions)
      fitnesses += l2_decay
    return fitnesses

  def __update_best(self, fitnesses:np.ndarray):
    idx = np.argsort(fitnesses)[::-1]

    best_fitness = fitnesses[idx[0]]
    best_mu = self.solutions[idx[0]]

    self.curr_best_fitness = best_fitness
    self.curr_best_mu = best_mu

    if self.first_interation:
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

  def result(self): # return best params so far, along with historically best reward, curr reward, sigma
    return (self.best_mu, self.best_fitness, self.curr_best_fitness, self.sigma)
