from evo_simulator.GENERAL.Genome import Genome_NN
from evo_simulator.GENERAL.Distance import Distance
from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Population import Population_NN as Specie
from evo_simulator.GENERAL.Index_Manager import get_new_population_id

from typing import List, Set, Any, Dict
import numpy as np
import numba as nb
import TOOLS
import sys


class Specie_Manager():
    def __init__(self, config_path_file:str):
        
        # Mutatable Variables
        self.population:Dict[int, Genome_NN] = None
        self.species:Dict[int, Specie] = {}
        self.mean_distance:float = 0
        self.config_path_file:str = config_path_file
        self.config_specie:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, ["NEAT", "Specie", "Distance", "NEURO_EVOLUTION"])
        self.history_species_previous_fitness:Dict[int, Dict[str, float]] = {}
        self.pop_size:int = int(self.config_specie["NEAT"]["pop_size"])

        # Distance Variables
        self.distance_threshold_ratio:float = float(self.config_specie["Distance"]["distance_threshold_ratio"])
        self.distance_threshold_min:float = float(self.config_specie["Distance"]["distance_threshold_min"])

        # Constants Variables
        self.species_elitism:int = int(self.config_specie["Specie"]["species_elitism"]) # Number of species to keep
        # self.specie_pop_min_size:int = int(self.config_specie["Specie"]["specie_pop_min_size"]) # Minimum size of a specie
        self.stagnation_threshold:int = int(self.config_specie["Specie"]["stagnation_threshold"]) # Number of generations without improvement to consider a specie extincted
        self.keep_best_species:bool = True if self.config_specie["Specie"]["keep_best_species"] == "True" else False 
        self.delete_genomes_with_specie:bool = True if self.config_specie["Specie"]["delete_genomes_with_specie"] == "True" else False
        self.delete_genomes_elite_with_specie:bool = True if self.config_specie["Specie"]["delete_genomes_elite_with_specie"] == "True" else False
        self.stagnation_differenciation:str = self.config_specie["Specie"]["stagnation_differenciation"]
        if self.stagnation_differenciation not in ["best", "mean"]: raise Exception("stagnation_differenciation must be 'best' or 'mean'")
        self.optimization_type:str = self.config_specie["NEURO_EVOLUTION"]["optimization_type"]

        # Parameters to mutate
        self.distance:Distance = Distance(config_path_file)
        self.represantative_specie_to_genome:Dict[int, int] = {}
        self.represantative_genome_to_specie:Dict[int, int] = {}


    def speciation(self, population:Population) -> None:
        
        self.population:Dict[int, Genome_NN] = population.population
        self.distance.reset_cache()

        # 1- Update age + stagnation
        self.__species_update_information()

        # 2- Remove old species
        self.__species_remove_old()

        # 3- Update Representative and reset species genome list
        unspeciated_genomes:Set[int] = set(self.population.keys())
        self.__species_update_representative_genome(unspeciated_genomes)
    
        # 4- Speciation or create new species
        self.__species_speciation_2(unspeciated_genomes)

        # 5- Size adjustement of species for reproduction
        self.__species_size_adjustement_explicit_fitness_sharing_based()

        population.population = self.population
        self.population = {}


    def __species_update_information(self) -> None:
        self.new_history_species_previous_fitness:Dict[int, Dict[str, float]] = {}
        for specie_id, specie in self.species.items():

            # 1 - Update Info specie
            specie.update_info()

            # 2 - Update Age
            specie.age += 1

            previous_fitness:Dict[str, float] = self.history_species_previous_fitness.get(specie_id, None)
            if previous_fitness is not None:
                # 3 - Update Stagnation
                if self.stagnation_differenciation == "best":
                    best_previous_fitness:float = previous_fitness["best"]
                    if self.optimization_type == "maximize" and specie.fitness.score <= best_previous_fitness:
                        specie.stagnation += 1
                    elif self.optimization_type == "minimize" and specie.fitness.score >= best_previous_fitness:
                        specie.stagnation += 1
                    elif self.optimization_type == "closest_to_zero" and abs(specie.fitness.score) >= abs(best_previous_fitness):
                        specie.stagnation += 1
                    else:
                        specie.stagnation = 0
                        self.new_history_species_previous_fitness[specie_id] = {"best": specie.fitness.score, "mean": specie.fitness.mean}

                elif self.stagnation_differenciation == "mean":
                    mean_previous_fitness:float = previous_fitness["mean"]
                    if self.optimization_type == "maximize" and specie.fitness.mean <= mean_previous_fitness:
                        specie.stagnation += 1
                    elif self.optimization_type == "minimize" and specie.fitness.mean >= mean_previous_fitness:
                        specie.stagnation += 1
                    elif self.optimization_type == "closest_to_zero" and abs(specie.fitness.mean) >= abs(mean_previous_fitness):
                        specie.stagnation += 1
                    else:
                        specie.stagnation = 0
                        self.new_history_species_previous_fitness[specie_id] = {"best": specie.fitness.score, "mean": specie.fitness.mean}
            else:
                self.new_history_species_previous_fitness[specie_id] = {"best": specie.fitness.score, "mean": specie.fitness.mean}
        self.history_species_previous_fitness.update(self.new_history_species_previous_fitness)

    
    def __species_remove_old(self) -> None:
        species_ranked:List[Specie] = sorted(self.species.values(), key=lambda specie: specie.fitness.score, reverse=True)
        if self.keep_best_species == True: species_ranked = species_ranked[self.species_elitism:]
        if len(species_ranked) == 0: return

        genomes_to_remove:Set[float] = set()
        species_to_remove:Set[float] = set()
        for specie in species_ranked:
            if specie.stagnation > self.stagnation_threshold:

                print("Specie extincted:", specie.id)
                species_to_remove.add(specie.id)

                if self.delete_genomes_with_specie == True:
                    for genome_id in specie.population.keys():
                        if genome_id in self.population:
                            if self.delete_genomes_elite_with_specie == False and self.population[genome_id].info["is_elite"] == True:
                                continue
                            else:
                                genomes_to_remove.add(genome_id)

        # Remove genomes and species
        elites:List[Genome_NN] = []
        for genome_id in genomes_to_remove:
            if self.population[genome_id].info["is_elite"] == True:
                elites.append(self.population[genome_id])
            del self.population[genome_id]
        for specie_id in species_to_remove:
            del self.species[specie_id]

        if len(self.population) <= 0:
            for elite in elites:
                self.population[elite.id] = elite

    def __species_update_representative_genome(self, unspeciated:Set[int]) -> None:
        self.represantative_specie_to_genome:Dict[int, int] = {}
        self.represantative_genome_to_specie:Dict[int, int] = {}
        for id, specie in self.species.items():
            # Get closest genome
            if specie.extra_info["representative_genome"] in self.population:
                if specie.extra_info["representative_genome"] in unspeciated and len(unspeciated) == 1:
                    closest_genome:int = specie.best_genome.id    
                else:
                    self.distance.distance_genomes_list([specie.extra_info["representative_genome"]], unspeciated, self.population, reset_cache=False)
                    closest_genome:int = self.distance.distances_min_genomes_cache[specie.extra_info["representative_genome"]]["id_global"]
            else: # representative_genome not in population cause it has been removed during the reproduction process (cause representative_genome can be not an elite)
                closest_genome:int = specie.best_genome.id
                if closest_genome not in unspeciated: # cause the best genome from this specie can have been previously choosen as representative_genome of another specie (because of the distance), so it can be speciated (and so not in unspeciated)
                    self.distance.distance_genomes_list([closest_genome], unspeciated, self.population, reset_cache=False)
                    closest_genome:int = self.distance.distances_min_genomes_cache[closest_genome]["id_global"]

            # Update representative genome
            specie.population:Dict[int, Genome_NN] = {}
            specie.population[closest_genome] = self.population[closest_genome]
            specie.extra_info["representative_genome"] = closest_genome
            self.represantative_specie_to_genome[id] = closest_genome
            self.represantative_genome_to_specie[closest_genome] = id

            # remove closest_genome (new_representative) from unspeciated
            unspeciated.remove(closest_genome)

    def __species_speciation(self, unspeciated:Set[int]) -> float:

        # 0 - Compute distance between genomes (representative_genome, unspeciated)
        self.distance.distance_genomes_list(self.represantative_specie_to_genome.values(), unspeciated, self.population, reset_cache=False)

        self.mean_distance:float = self.distance.mean_distance["global"]
        self.mean_distance_cumulative:float = 0
        
        genome_cum:int = 0
        while unspeciated:
            candidate_specie_id:int = None
            candidate_distance:float = sys.float_info.max
            genome_to_speciate:int = unspeciated.pop()

            # Get candidate species
            for specie_id, specie in self.species.items():
                # 1 - Get distance between genomes
                distance_between_genomes:float = self.distance.distance_genomes(self.population[genome_to_speciate], self.population[specie.extra_info["representative_genome"]])

                # 2 - Custom compatibility distance. By percentage of mean distance
                if self.mean_distance == 0:
                    self.mean_distance_cumulative += distance_between_genomes
                    genome_cum += 1
                    self.distance_threshold_used:float = ((self.mean_distance_cumulative/genome_cum) + ((self.mean_distance_cumulative/genome_cum) * self.distance_threshold_ratio))
                    self.distance_threshold_used:float = max(self.distance_threshold_used, self.distance_threshold_min)
                    if distance_between_genomes <= self.distance_threshold_used and distance_between_genomes < candidate_distance:
                        candidate_distance = distance_between_genomes
                        candidate_specie_id = specie_id
                else:
                    self.distance_threshold_used:float = (self.mean_distance + (self.mean_distance * self.distance_threshold_ratio))
                    self.distance_threshold_used:float = max(self.distance_threshold_used, self.distance_threshold_min)
                    if distance_between_genomes <= self.distance_threshold_used and distance_between_genomes < candidate_distance:
                        candidate_distance = distance_between_genomes
                        candidate_specie_id = specie_id

            # 3 - Speciate genome
            if candidate_specie_id is not None: # Specie found
                self.species[candidate_specie_id].population[genome_to_speciate] = self.population[genome_to_speciate]
            else: # otherwise create new specie
                new_specie_id:int = get_new_population_id()
                self.species[new_specie_id] = Specie(new_specie_id, genome_to_speciate)
                self.represantative_specie_to_genome[new_specie_id] = genome_to_speciate
                self.represantative_genome_to_specie[genome_to_speciate] = new_specie_id

        # print("self.species", self.species)
        print("self.represantative_specie_to_genome", self.represantative_specie_to_genome)
        print("self.represantative_genome_to_specie", self.represantative_genome_to_specie)
        print("mean_distance", self.mean_distance)
        # # print("min_distance", min_distance)
        
        # 4 - Define mean distance
        self.mean_distance = self.mean_distance if self.mean_distance != 0 else self.mean_distance_cumulative/genome_cum
        return self.mean_distance

    def __species_speciation_2(self, unspeciated:Set[int]) -> None:

        min_distance:float = sys.float_info.max
        genome_to_speciate:int = 0
        candidate_specie_id:int = -1
        candidate_id:int = -1

        while unspeciated:
            genome_to_speciate:int = unspeciated.pop()

            # 1 - Compute distance between genomes (genome_to_speciate, representative_genome)
            self.distance.distance_genomes_list([genome_to_speciate], self.represantative_specie_to_genome.values(), self.population, reset_cache=False)

            # 2 - Set mean distance, min distance and distance threshold
            self.mean_distance:float = self.distance.mean_distance["global"]
            min_distance:float = self.distance.distances_min_genomes_cache[genome_to_speciate]["distance_global"] if self.mean_distance != 0 else sys.float_info.max 
            self.distance_threshold_used:float = (self.mean_distance + (self.mean_distance * self.distance_threshold_ratio))
            self.distance_threshold_used:float = max(self.distance_threshold_used, self.distance_threshold_min)

            # 3.1 - Speciate genome
            if min_distance <= self.distance_threshold_used:
                candidate_id:int = self.distance.distances_min_genomes_cache[genome_to_speciate]["id_global"]
                candidate_specie_id:int = self.represantative_genome_to_specie[candidate_id]
                self.species[candidate_specie_id].population[genome_to_speciate] = self.population[genome_to_speciate]
            else: # 3.2 otherwise create new specie
                new_specie_id:int = get_new_population_id()
                self.species[new_specie_id] = Specie(new_specie_id, self.config_path_file, extra_info={"representative_genome":genome_to_speciate})
                self.species[new_specie_id].population[genome_to_speciate] = self.population[genome_to_speciate]
                self.represantative_specie_to_genome[new_specie_id] = genome_to_speciate
                self.represantative_genome_to_specie[genome_to_speciate] = new_specie_id

        # print("self.represantative_specie_to_genome", self.represantative_specie_to_genome)
        # print("self.represantative_genome_to_specie", self.represantative_genome_to_specie)
        # print("mean_distance", self.mean_distance)
        # print("min_distance", min_distance)        

    def __species_size_adjustement_explicit_fitness_sharing_based(self) -> None:

        # 1 - Get population fitnesses and Check if best fitness is the max or the min
        if self.optimization_type == "maximize":
            population_fitnesses:np.ndarray = np.array([genome.fitness.score for genome in self.population.values() if genome.fitness.score != None], dtype=np.float32)
        elif self.optimization_type == "minimize":
            population_fitnesses:np.ndarray = np.array([-genome.fitness.score for genome in self.population.values() if genome.fitness.score != None], dtype=np.float32)
        elif self.optimization_type == "closest_to_zero":
            population_fitnesses:np.ndarray = np.array([-abs(genome.fitness.score) for genome in self.population.values() if genome.fitness.score != None], dtype=np.float32)

        # 2 - Compute mean normalized population fitness
        if population_fitnesses.size == 0: return
        population_fitnesses_normalized_mean:float = np.mean(self.normalize_array_jit(population_fitnesses, population_fitnesses.max(), population_fitnesses.min()))

        # 3- Compute species (ajusted) fitnesses and new sizes according to explicit fitness sharing
        species_new_sizes:List[float] = []    
        for specie in self.species.values():
            specie_ajusted_fitnesses:np.ndarray = self.__species_ajusted_fitness_simplified(specie, population_fitnesses.max(), population_fitnesses.min())
            species_new_sizes.append(np.sum(specie_ajusted_fitnesses) / population_fitnesses_normalized_mean)

        # 4- repartion species size (has to keep the order of species!!!)
        species_new_sizes:np.ndarray = self.species_size_repartion_jit(np.array(species_new_sizes), self.pop_size)
        if int(np.sum(species_new_sizes)) != self.pop_size: raise Exception("species_new_sizes sum is not equal to pop_size", int(np.sum(species_new_sizes)), self.pop_size)

        # 5- update species size & remove empty species
        species_copy = list(self.species.values())
        for i, specie in enumerate(species_copy):
            specie.reproduction_size = int(species_new_sizes[i])
            if specie.reproduction_size <= 0:
                for genome_id in specie.population.keys():
                    del self.population[genome_id]
                del self.species[specie.id]

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def species_size_repartion_jit(species_new_sizes:np.ndarray, pop_size:int) -> np.ndarray:
        # 3- repartion species size
        species_new_sizes = np.ceil((species_new_sizes / np.sum(species_new_sizes)) * pop_size)
        species_new_sizes_id_sorted = np.argsort(species_new_sizes)

        # 4- ajust species size in order to keep the same number of genomes
        while int(np.sum(species_new_sizes)) != pop_size:
            for id in species_new_sizes_id_sorted:
                if np.sum(species_new_sizes) > pop_size and species_new_sizes[id] >= 1:
                    species_new_sizes[id] -= 1

                elif np.sum(species_new_sizes) < pop_size and species_new_sizes[id] >= 1:
                    species_new_sizes[id] += 1

                if int(np.sum(species_new_sizes)) == pop_size: break
        return species_new_sizes

    @staticmethod
    @nb.njit(cache=True, fastmath=True, nogil=True)
    def normalize_array_jit(array:np.ndarray, max:float, min:float) -> np.ndarray:
        if np.all((array == min)) and (max - min) == 0 or (max - min) == 0:
            return (array - min) + 1
        return (array - min) / (max - min)

    def __species_ajusted_fitness_simplified(self, specie:Specie, max_fitness:float, min_fitness:float) -> np.ndarray:
        if self.optimization_type == "maximize":
            specie_fitnesses:np.ndarray = np.array([genome.fitness.score for genome in specie.population.values() if genome.fitness.score != None], dtype=np.float32)
        elif self.optimization_type == "minimize":
            specie_fitnesses:np.ndarray = np.array([-genome.fitness.score for genome in specie.population.values() if genome.fitness.score != None], dtype=np.float32)
        elif self.optimization_type == "closest_to_zero":
            specie_fitnesses:np.ndarray = np.array([-abs(genome.fitness.score) for genome in specie.population.values() if genome.fitness.score != None], dtype=np.float32)
        return self.normalize_array_jit(specie_fitnesses, max_fitness, min_fitness) / specie_fitnesses.size