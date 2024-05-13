from evo_simulator.GENERAL.Population import Population
from evo_simulator.GENERAL.Genome import Genome
from evo_simulator.GENERAL.Index_Manager import get_new_population_id, get_new_niche_status_index, reset_niche_status_index
from typing import Any, Dict, List, Tuple, Callable, Set
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from io import TextIOWrapper
import TOOLS
import pickle
import os
import random
import time

class Niche():
    def __init__(self, centroid:np.ndarray, status_index:int, genome_id:int, fitness:float, description:np.ndarray) -> None:
        # General
        self.status_index:int = status_index
        self.genome_ids:Set[int] = set()
        self.genome_ids.add(genome_id)
        self.centroid:np.ndarray = centroid
        self.description:np.ndarray = description
        self.fitness:float = fitness
        self.try_to_update_counter:int = 0
        self.improve_counter:int = 0

        # Specific to novelty search
        self.novelty_score:float = 0.0
        self.competition_score:float = 0.0
        
        # Extra info
        self.info:Dict[str, Any] = {}

    def update(self, genome_id:int, fitness:float, description:np.ndarray):
        self.fitness:float = fitness
        self.genome_ids.add(genome_id)
        self.description:np.ndarray = description
        self.improve_counter += 1

class Archive:
    def __init__(self, config_section_name:str, config_path_file:str, genome_builder_function:Callable) -> None:

        # 0 - Config parameters
        self.config_path_file:str = config_path_file
        self.genome_builder_function:Callable = genome_builder_function
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path_file, [config_section_name, "NEURO_EVOLUTION"])
        self.folder_path:str = self.config[config_section_name]["folder_path"]
        if self.folder_path != "" and self.folder_path[-1] != "/": self.folder_path += "/"
        reset_niche_status_index()


        # 1 - Archive parameters
        self.niches_info:Dict[Any, Niche] = {} # key: centroid, value: niche
        self.archive_genomes:Dict[Any, Genome] = {} # key: centroid, value: genome
        self.description_name:str = self.config[config_section_name]["description_name"] # name of the description
        self.update_criteria:str = self.config[config_section_name]["update_criteria"] # fitness, novelty, competitions
        self.optimization_type:str = self.config[config_section_name]["optimization_type"] # maximize, minimize, closest_to_zero
        self.archive_dimensions:int = int(self.config[config_section_name]["dimensions"]) # dimension of the description space
        self.name:str = self.config["NEURO_EVOLUTION"]["algo_name"] + "_" + config_section_name + "_" + self.description_name


        # 2 - CVT parameters
        self.niches_nb:int = int(self.config[config_section_name]["niches_nb"]) # number of niches
        self.cvt_samples:int = int(self.config[config_section_name]["cvt_samples"]) # # more of this -> higher-quality CVT (for the KMeans algorithm)
        self.cvt_use_cache:bool = True if self.config[config_section_name]["cvt_use_cache"] == "True" else False # do we cache the result of CVT and reuse?
        
        # 3 - Utilities parameters
        self.best_population:Population = Population(get_new_population_id(), self.config_path_file)
        self.best_population_size:int = int(self.config[config_section_name]["best_population_size"])

        self.generation:int = 0
        self.checkpoint_period:int = int(self.config[config_section_name]["checkpoint_period"]) # number of generations before saving the archive
        self.checkpoint_counter:int = 0
        self.improvement_archives:int = 0 # number of times the archives have been improved

        # 4 - Niches parameters
        self.niches_status:np.ndarray = np.zeros(self.niches_nb, dtype=np.int32) # 0: empty, 1: filled
        self.fitness_niches:np.ndarray = np.zeros(self.niches_nb, dtype=np.float32) # fitness of the niches
        self.description_niches:np.ndarray = np.zeros((self.niches_nb, self.archive_dimensions), dtype=np.float32) # description of the niches
        self.centroid_niches:np.ndarray = np.zeros((self.niches_nb, self.archive_dimensions), dtype=np.float32) # description of the niches

        # 5 - Build the CVT (niches)
        self.kdt:KDTree = self.__init_cvt(self.niches_nb, self.archive_dimensions, self.cvt_samples, self.cvt_use_cache) # made for fast nearest-neighbor queries (find the closest niche to a given individual)
        self.checkpoint_period_counter:int = 0
        self.check_point_time:float = time.time()



    def __init_cvt(self, niches_nb:int, map_dim:int, cvt_samples:int, cvt_use_cache:bool=False) -> KDTree:
        # 0 - Build the folder for the archives
        self.__build_folder() # build the folder for the archives
        
        # 1 - Check if we have cached values
        fname:str = self.__centroids_filename(niches_nb, map_dim)
        if cvt_use_cache == True:
            if Path(fname).is_file() == True:
                print("WARNING: using cached CVT:", fname)
                return np.loadtxt(fname)
        # otherwise, compute cvt
        print("Build CVT map...:", fname)

        # 2 - Create the CVT with KMeans algorithm
        random_distributed_data_points:np.ndarray = np.random.rand(cvt_samples, map_dim) # more samples -> better CVT: random points uniformly distributed in the description space
        k_means:KMeans = KMeans(init='k-means++', n_clusters=niches_nb, n_init=1, verbose=0)#,algorithm="full") # Build the k-means object
        k_means.fit(random_distributed_data_points) # Create the 'niches_nb' CVT
        self.__write_centroids(k_means.cluster_centers_) # save the CVT - cluster centers is the CVT (centroids)

        return KDTree(k_means.cluster_centers_, leaf_size=30, metric='euclidean') # made for fast nearest-neighbor queries (find the closest niche to a given individual)
  
    def get_random_genome_from_archive(self, size:int) -> Population:
        population:Population = Population(get_new_population_id(), self.config_path_file)
        population_dict:Dict[int, Genome] = population.population
        niches_keys:List = list(self.niches_info.keys())
        if len(niches_keys) < size: raise Exception("ERROR: not enough niches in the archive")
        centroid_list:List[Any] = random.sample(niches_keys, size)

        # 1 - Load the genomes
        for centroid in centroid_list:
            # 1.1 Check if the centroid is in the archive
            if self.niches_info.get(centroid) == None: raise Exception("ERROR: centroid not found in archive")
            
            # 1.2 Load the genome
            file_name:str = self.__path_archive_genomes + 'genome_' + str(centroid) + '.pkl'
            with open(file_name, 'rb') as file:
                genome:Genome = pickle.load(file)
                population_dict[genome.id] = genome
        return population

    def get_new_random_genome_not_from_archive(self, size:int) -> Population:
        population:Population = Population(get_new_population_id(), self.config_path_file)
        population_dict:Dict[int, Genome] = population.population
        while len(population_dict) < size:
            new_genome:Genome = self.genome_builder_function()
            population_dict[new_genome.id] = new_genome
        return population

    def get_archive_filled_ratio(self) -> float:
        return len(self.niches_info) / self.niches_nb

    def get_archive_filled_nb(self) -> int:
        return len(self.niches_info)

    def get_best_population(self) -> Population:
        return self.best_population

    def get_niches_info(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_niches_indexes:np.ndarray = np.where(self.niches_status == 1)[0]
        return self.fitness_niches[active_niches_indexes], self.description_niches[active_niches_indexes], self.centroid_niches[active_niches_indexes]

    def load_genome_from_archive(self, centroid:Any) -> Genome:
        # 0 - Check if the centroid is in the archive
        if self.niches_info.get(centroid) == None: raise Exception("ERROR: centroid not found in archive")
        
        # 1 - Load the genome
        file_name:str = self.__path_archive_genomes + 'genome_' + str(centroid) + '.pkl'
        with open(file_name, 'rb') as file:
            genome:Genome = pickle.load(file)
        return genome

    def __update_niches_status(self, index:int, fitness:float, description:np.ndarray, centroid:np.ndarray):
        # 1 - Update the niche status
        self.niches_status[index] = 1

        # 2 - Update the fitness of the niche
        self.fitness_niches[index] = fitness

        # 3 - Update the description of the niche
        self.description_niches[index] = description

        # 4 - Update the centroid of the niche
        self.centroid_niches[index] = centroid

    def update(self, population:Population, update_criteria:str=None):
        population_dict:Dict[int, Genome] = population.population
        first_genome:Genome = population_dict[next(iter(population_dict))]
        self.archive_genomes:Dict[Any, Genome] = {}

        # 0 - Check if the genome is well-formed
        if update_criteria != None: self.update_criteria = update_criteria
        if self.update_criteria != "fitness":
            if self.update_criteria not in first_genome.info: raise Exception("Reproduction: criteria ("+self.update_criteria+") does not exist in the genome info")
            if type(first_genome.info[self.update_criteria]) != float: raise Exception("Reproduction: criteria must be a float")
        if first_genome.info[self.description_name] is None: raise Exception("ERROR: genome description name:" + self.description_name + " does not exist")
        if len(first_genome.info[self.description_name]) != self.archive_dimensions: raise Exception("ERROR: genome description dimension is not equal to the archive dimension got", len(first_genome.info[self.description_name]),"expected", self.archive_dimensions)
        if np.all((first_genome.info[self.description_name] >= 0) & (first_genome.info[self.description_name] <= 1)) == False: raise Exception("ERROR: genome description is not in [0, 1] interval")

        for genome in population_dict.values():

            # 1 - Get the niche based on the description (centroid)
            niche_index = self.kdt.query([genome.info[self.description_name]], k=1)[1][0][0]
            centroid:np.ndarray = self.kdt.data[niche_index]
            niche = self.make_hashable(centroid)

            # 2 - Update the archive (if the new genome is better than the one in the archive)
            criteria_score:float = genome.fitness.score if self.update_criteria == "fitness" else genome.info[self.update_criteria]
            if niche in self.niches_info:
                self.niches_info[niche].try_to_update_counter += 1

                if self.criteria_condition(criteria_score, self.niches_info[niche].fitness) == True: # maximize, minimize, closest_to_zero
                    self.improvement_archives += 1
                    genome.info["centroid"] = niche
                    self.niches_info[niche].update(genome.id, criteria_score, genome.info[self.description_name])
                    self.__update_niches_status(self.niches_info[niche].status_index, criteria_score, genome.info[self.description_name], centroid)
                    self.archive_genomes[niche] = genome
                else:
                    self.niches_info[niche].genome_ids.add(genome.id) # for the local competition score (could be remove if we don't use it)

            # 2.1 - If the centroid is not in the archive, add it
            else:
                genome.info["centroid"] = niche
                self.niches_info[niche] = Niche(niche, get_new_niche_status_index(), genome.id, criteria_score, genome.info[self.description_name])
                self.__update_niches_status(self.niches_info[niche].status_index, criteria_score, genome.info[self.description_name], centroid)
                self.archive_genomes[niche] = genome

        self.__update_best_population(population)
        self.__save_archive_genome()
        self.__checkpoint_save_archive()

    def __save_archive_genome(self):
        # 1 - Dump the genomes in the archive files
        for centroid, genome in self.archive_genomes.items():
            file_name:str = self.__path_archive_genomes + 'genome_' + str(centroid) + '.pkl'
            with open(file_name, 'wb') as file:
                pickle.dump(genome, file)
        # 2 - Reset the archive genomes for the next batch and memory management
        self.archive_genomes:Dict[Any, Genome] = {}

    def criteria_condition(self, criteria_score:float, niche_fitness:float) -> bool:
        if self.optimization_type == "maximize":
            return criteria_score > niche_fitness
        elif self.optimization_type == "minimize":
            return criteria_score < niche_fitness
        elif self.optimization_type == "closest_to_zero":
            return abs(criteria_score) < abs(niche_fitness)
        else:
            raise Exception("ERROR: optimization '" + self.optimization_type +"' type not recognized")


    def __update_best_population(self, population:Population):
        best_population_dict:Dict[int, Genome] = self.best_population.population
        current_population_dict:Dict[int, Genome] = population.population

        # 1 - Concatenate the best population and the current population
        all_genomes_list:List[Genome] = list(best_population_dict.values()) + list(current_population_dict.values())

        # 2 - Sort the Genome by fitness
        if self.optimization_type == "maximize":
            all_genomes_list.sort(key=lambda x: x.fitness.score, reverse=True)
        elif self.optimization_type == "minimize":
            all_genomes_list.sort(key=lambda x: x.fitness.score, reverse=False)
        elif self.optimization_type == "closest_to_zero":
            all_genomes_list.sort(key=lambda x: abs(x.fitness.score), reverse=False)

        # 3 - Keep the pop_size best genomes
        best_population_dict:Dict[int, Genome] = {}
        for i in range(self.best_population_size):
            best_population_dict[all_genomes_list[i].id] = all_genomes_list[i]
        self.best_population.population = best_population_dict

    def __checkpoint_save_archive(self):
        self.checkpoint_period_counter += 1
        self.best_population.update_info()
        # 1 - Check if we need to save the archive
        if self.checkpoint_period_counter < self.checkpoint_period: 
            print(self.name+": Generation:", self.generation, "Archive filled:", len(self.niches_info), "/", self.niches_nb, "(" + str(round(len(self.niches_info) / self.niches_nb, 3)) + "%); ", "improvement: +" + str(self.improvement_archives) +";", "best fitness:", round(self.best_population.fitness.score, 3), end=";\n")
            self.improvement_archives:int = 0
            self.generation += 1
            return
        # 2 - Print the current state of the archive
        self.checkpoint_period_counter:int = 0
        print(self.name+": Generation:", self.generation, "Archive filled:", len(self.niches_info), "/", self.niches_nb, "(" + str(round(len(self.niches_info) / self.niches_nb, 3)) + "%); ", "improvement: +" + str(self.improvement_archives) +";", "best fitness:", round(self.best_population.fitness.score, 3), end="; ")
        print("Build New check point:", self.checkpoint_counter)
        self.checkpoint_counter += 1
        self.improvement_archives:int = 0
        # print("Build check point:", str(self.checkpoint_total - self.checkpoint_remaining +1) +"/"+ str(self.checkpoint_total), "-->", round(time.time() - self.check_point_time, 3), "s", end="; ")
        # print("Estimated time left:", round(self.checkpoint_remaining * (time.time() - self.check_point_time), 3), "s")
        # self.check_point_time:float = time.time()

        # 3 - Save the archive (write the archive in a file) -> format: fitness, centroid, descriptions, try_to_update_counter, improve_counter
        filename:str = self.__path_archive_checkpoint + 'archive_' + str(self.generation) + '.dat'
        with open(filename, 'w') as f:
            for centroid, niche in self.niches_info.items():
                f.write(str(niche.fitness) + ' ') # -> fitness
                self.__write_array(centroid, f) # -> centroid
                self.__write_array(niche.description, f) # -> description
                f.write(str(niche.try_to_update_counter) + ' ') # -> try_to_update_counter
                f.write(str(niche.improve_counter) + ' ') # -> improve_counter
                f.write(str(len(self.niches_info)) + ' ') # -> current nb niches
                f.write(str(self.niches_nb)+ ' ') # -> maximum niches possible
                f.write(str(len(self.niches_info) / self.niches_nb)) # -> percentage of niches filled
                f.write("\n")
        # active_niches_indexes:np.ndarray = np.where(self.niches_status == 1)[0]
        # print("fitness_max", self.fitness_niches[active_niches_indexes].max(), "fitness_min", self.fitness_niches[active_niches_indexes].min(), "fitness_mean", self.fitness_niches[active_niches_indexes].mean())
        self.generation += 1
    
    def get_centroid_from_description(self, description:np.ndarray) -> Any:
        return self.make_hashable(self.kdt.data[self.kdt.query([description], k=1)[1][0][0]])


    def __write_centroids(self, centroids:np.ndarray):
        niche_nb:int = centroids.shape[0]
        map_dim:int = centroids.shape[1]
        filename:str = self.__centroids_filename(niche_nb, map_dim)
        with open(filename, 'w') as f:
            for p in centroids:
                for item in p:
                    f.write(str(item) + ' ')
                f.write('\n')

    def __write_array(self, a, f:TextIOWrapper):
        for i in a:
            f.write(str(i) + ' ')

    def make_hashable(self, array) -> Tuple[float]:
        # print("tuple(map(float, array)) == tuple(array)", tuple(map(float, array)) == tuple(array))
        return tuple(map(float, array))

    def __centroids_filename(self, n_niches:int, map_dim:int) -> str:
        return self.__path_archive + 'centroids_' + str(n_niches) + '_' + str(map_dim) + '.dat'

    def __build_folder(self):
        if os.path.exists("./"+ self.folder_path) == False: os.mkdir(""+ self.folder_path)
        self.name:str = self.__init_file(self.folder_path, self.name)
        if os.path.exists("./"+ self.folder_path + self.name) == False: os.mkdir(""+ self.folder_path + self.name)
        if os.path.exists("./"+ self.folder_path + self.name+"/checkpoint") == False: os.mkdir("./"+ self.folder_path + self.name+"/checkpoint")
        if os.path.exists("./"+ self.folder_path + self.name+"/archives_genomes") == False: os.mkdir("./"+ self.folder_path + self.name+"/archives_genomes")

        self.__path_archive:str = "./"+ self.folder_path + self.name+"/"
        self.__path_archive_checkpoint:str = "./"+ self.folder_path + self.name+"/checkpoint/"
        self.__path_archive_genomes:str = "./"+ self.folder_path + self.name+"/archives_genomes/"


    def __init_file(self, config_path, file_name:str) -> str:
        if os.path.exists(config_path + file_name):
            i = 1
            while os.path.exists(config_path + file_name + "_" + str(i)):
                i += 1
            # return file_name + "_" + str(i) + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
            return file_name + "_" + str(i)
        else:
            return file_name

    def __load_genome_from_file(self, centroid:Any) -> Genome:
        # 1.2 Load the genome
        file_name:str = self.__path_archive_genomes + 'genome_' + str(centroid) + '.pkl'
        with open(file_name, 'rb') as file:
            return pickle.load(file)
