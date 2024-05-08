from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Genome import Genome_NN
from problem.RL.ENVIRONNEMENT import Environment
class Problem:
    def __init__(self) -> None:
        pass

    def run(self, population:Population, run_nb:int, generation:int, seed:int = None) -> Population:  
        raise Exception("The run function must be implemented in the problem class")
    
    def run_render(self, genome:Genome_NN, env:Environment, seed:int) -> None:
        raise Exception("The run_render function must be implemented in the problem class")