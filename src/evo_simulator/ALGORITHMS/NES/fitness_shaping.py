import numpy as np

def compute_ranks(x:np.ndarray) -> np.ndarray:
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks:np.ndarray = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_fitness_ranks(fitnesses_raw:np.ndarray) -> np.ndarray:
  """
  FITNESS SHAPING: transformes raw fitness into a centered rank (linear fitness)
  from: https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  fitnesses_ranked:np.ndarray = compute_ranks(fitnesses_raw.ravel()).reshape(fitnesses_raw.shape).astype(np.float32)
  fitnesses_ranked /= (fitnesses_raw.size - 1)
  fitnesses_ranked -= .5
  return fitnesses_ranked

def compute_parameters_decay(parameters_decay_coef:float, parameters:np.ndarray) -> np.ndarray:
  return - parameters_decay_coef * np.mean(parameters * parameters, axis=1)
