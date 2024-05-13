genome_index = -1
population_index = -1
niche_status_index = -1

@staticmethod
def get_new_niche_status_index() -> int:
    global niche_status_index
    niche_status_index += 1
    return niche_status_index

@staticmethod
def get_new_genome_id() -> int:
    global genome_index
    genome_index += 1
    return genome_index

@staticmethod
def get_new_population_id() -> int:
    global population_index
    population_index += 1
    return population_index


@staticmethod
def reset_niche_status_index():
    global niche_status_index
    niche_status_index = -1

@staticmethod
def reset_population_id():
    global population_index
    population_index = -1

@staticmethod
def reset_genome_id():
    global genome_index
    genome_index = -1
