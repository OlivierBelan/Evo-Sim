from evo_simulator.GENERAL.Genome import Genome, Genome_NN
from evo_simulator.GENERAL.Population import Population
from evo_simulator import TOOLS
from typing import List, Dict, Any
import os
import shutil
import pickle
import numpy as np

class Record():
    def __init__(self, config_path:str, problem_name:str) -> None:
        self.config:Dict[str, Dict[str, Any]] = TOOLS.config_function(config_path, ["Record", "NEURO_EVOLUTION"])
        self.folder_name:str = self.__init_folder(problem_name)
        self.optimization_type:str = self.config["NEURO_EVOLUTION"]["optimization_type"] # maximize, minimize, closest_to_zero
        self.criterias:List[str] = self.config["Record"]["criteria"].split(" ")
        self.sorted_by:str = self.config["Record"]["sorted_by"]
        self.best_genome:Genome_NN = None
        self.is_record_from_algo:bool = False
        if "record_from_algo" in self.config["Record"]:
            self.record_from_algo:bool = True if self.config["Record"]["record_from_algo"] == "True" else False
        else:
            self.record_from_algo:bool = False

        # 1 - Build folder
        self.__build_folder(self.folder_name, 0)
        # 2 - Save config file in folder
        shutil.copyfile(config_path, self.folder_path + "config")


    def __init_folder(self, folder_name:str) -> str:
        if os.path.exists("./results/" + folder_name):
            i = 1
            while os.path.exists("./results/" + folder_name + "_" + str(i)):
                i += 1
            return folder_name + "_" + str(i)
        else:
            return folder_name


    def __build_folder(self, folder_name:str, run_nb:int) -> str:
        # 1 - Check if folder exists
        folder_path:str = "./results/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path) # Build folder results
        if not os.path.exists(folder_path + folder_name):
            os.makedirs(folder_path + folder_name) # Build folder results/folder_name
        if not os.path.exists(folder_path + folder_name + "/" + str(run_nb)):
            os.makedirs(folder_path + folder_name + "/" + str(run_nb)) # Build folder results/folder_name/run_nb
        self.folder_path:str = folder_path + folder_name + "/"
        self.folder_run_path:str = folder_path + folder_name + "/" + str(run_nb)


    def save_info(self, population:Population, run_nb:int) -> None:
        self.__build_folder(self.folder_name, run_nb)
        population_dict:Dict[int, Genome] = population.population
        # 1 - Build info structure to save
        info_to_save:Dict[str, Any] = {}
        for criteria in self.criterias:
            info_to_save[criteria] = []

        # 2 - Collect info
        for genome in population_dict.values():
            for criteria in self.criterias:
                if criteria == "fitness":
                    info_to_save[criteria].append(genome.fitness.score)
                else:
                    info_to_save[criteria].append(genome.info[criteria])
        
        # 2.1 - Sort info
        if self.sorted_by in info_to_save:
            if self.optimization_type ==   "maximize":       sorted_indexes:np.ndarray = np.argsort(info_to_save[self.sorted_by])[::-1]
            elif self.optimization_type == "minimize":       sorted_indexes:np.ndarray = np.argsort(info_to_save[self.sorted_by])
            elif self.optimization_type == "closest_to_zero":sorted_indexes:np.ndarray = np.argsort(np.abs(info_to_save[self.sorted_by]))
            else:raise Exception("Unknown optimization type -> " + self.optimization_type)
            for criteria in self.criterias:
                info_to_save[criteria] = np.array(info_to_save[criteria])[sorted_indexes].tolist()
        
        # 3 - Save info
        for criteria in self.criterias:
            # sorted_info = sorted(info_to_save[criteria], reverse=True if self.optimization_type == "maximize" else False)
            with open(self.folder_run_path + "/" + criteria + ".txt", "a") as f:
                # f.write(str(sorted_info) + "\n")
                f.write(str(info_to_save[criteria]) + "\n")
        
        # 4 - Save best genome
        population.update_info()
        self.save_best(population.best_genome)
    
    def save_best(self, genome:Genome_NN, path:str=None) -> None:
        if self.best_genome == None:
            self.best_genome = genome
            self.__save_best(path)
        elif self.optimization_type == "maximize" and genome.fitness.score > self.best_genome.fitness.score:
            self.best_genome = genome
            self.__save_best(path)
        elif self.optimization_type == "minimize" and genome.fitness.score < self.best_genome.fitness.score:
            self.best_genome = genome
            self.__save_best(path)
        elif self.optimization_type == "closest_to_zero" and abs(genome.fitness.score) < abs(self.best_genome.fitness.score):
            self.best_genome = genome
            self.__save_best(path)
        
    def __save_best(self, path:str=None) -> None:
        if path == None:
            path:str = self.folder_run_path
        with open(path + "/best_genome.pickle", "wb") as f:
            pickle.dump(self.best_genome, f)

    @staticmethod
    def load_best(path:str) -> Genome_NN:
        with open(path, "rb") as f:
            genome:Genome_NN = pickle.load(f)
        return genome