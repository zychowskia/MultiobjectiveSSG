import numpy as np
import random
from algorithms import WCD_EA, LeaderStrategy
from mo_utils import calculate_nadir_point, worst_case_set, pareto_prune

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.core.individual import Individual
from pymoo.decomposition.tchebicheff import Tchebicheff

class NSGA3_Solver(WCD_EA):
    def __init__(self, game, pop_size=20, generations=20):
        super().__init__(game, pop_size, generations)
        
        if game.n_objectives == 2:
            n_partitions = pop_size - 1
            if n_partitions < 1: n_partitions = 1
        elif game.n_objectives == 3:
            n_partitions = int(np.sqrt(2 * pop_size))
            if n_partitions < 1: n_partitions = 1
        else:
            n_partitions = max(1, int(pop_size ** (1.0 / (game.n_objectives - 1))))
            
        self.ref_dirs = get_reference_directions("das-dennis", game.n_objectives, n_partitions=n_partitions)
        self.survival = ReferenceDirectionSurvival(self.ref_dirs)

    def run(self):
        population = self.initialize_population()
        
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")
            offspring = []
            
            while len(offspring) < self.pop_size:
                parents = random.sample(population, 2)
                child = self.crossover(parents[0], parents[1])
                self.mutate(child)
                self.evaluate_individual(child)
                offspring.append(child)
                
            combined = population + offspring
            
            pymoo_pop = Population.empty(len(combined))
            for i, ind in enumerate(combined):
                pt = ind.nadir if ind.nadir is not None else np.full(self.game.n_objectives, -np.inf)
                # Maximize payoffs -> minimize negative payoffs
                pymoo_pop[i].set("F", -pt)
                pymoo_pop[i].set("X", i) 
                
            from pymoo.core.problem import Problem
            class MockProblem(Problem):
                def __init__(self, n_obj):
                    super().__init__(n_var=1, n_obj=n_obj, n_ieq_constr=0)
                    
            surviving = self.survival.do(MockProblem(self.game.n_objectives), pymoo_pop, n_survive=self.pop_size)
            
            next_pop = []
            for ind in surviving:
                orig_idx = ind.get("X")
                next_pop.append(combined[orig_idx])
                
            population = next_pop
            
        return population

class MOEAD_Solver(WCD_EA):
    def __init__(self, game, pop_size=20, generations=20, n_neighbors=20):
        super().__init__(game, pop_size, generations)
        
        if game.n_objectives == 2:
            n_partitions = pop_size - 1
            if n_partitions < 1: n_partitions = 1
        elif game.n_objectives == 3:
            n_partitions = int(np.sqrt(2 * pop_size))
            if n_partitions < 1: n_partitions = 1
        else:
            n_partitions = max(1, int(pop_size ** (1.0 / (game.n_objectives - 1))))
            
        self.weights = get_reference_directions("das-dennis", game.n_objectives, n_partitions=n_partitions)
        self.actual_pop_size = len(self.weights)
        self.n_neighbors = min(n_neighbors, self.actual_pop_size)
        
        self.neighbors = np.zeros((self.actual_pop_size, self.n_neighbors), dtype=int)
        for i, weight in enumerate(self.weights):
            distances = np.linalg.norm(self.weights - weight, axis=1)
            self.neighbors[i] = np.argsort(distances)[:self.n_neighbors]
            
        self.decomposition = Tchebicheff()
        self.ideal_point = np.full(game.n_objectives, np.inf)

    def run(self):
        old_pop_size = self.pop_size
        self.pop_size = self.actual_pop_size
        population = self.initialize_population()
        self.pop_size = old_pop_size
        
        for ind in population:
            pt = ind.nadir if ind.nadir is not None else np.full(self.game.n_objectives, -np.inf)
            self.ideal_point = np.minimum(self.ideal_point, -pt)
            
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")
            
            for i in range(self.actual_pop_size):
                mating_pool = self.neighbors[i]
                parent_indices = random.sample(list(mating_pool), 2)
                
                p1 = population[parent_indices[0]]
                p2 = population[parent_indices[1]]
                
                child = self.crossover(p1, p2)
                self.mutate(child)
                self.evaluate_individual(child)
                
                child_pt = child.nadir if child.nadir is not None else np.full(self.game.n_objectives, -np.inf)
                child_f = -child_pt
                self.ideal_point = np.minimum(self.ideal_point, child_f)
                
                for neighbor_idx in mating_pool:
                    neighbor_ind = population[neighbor_idx]
                    neighbor_pt = neighbor_ind.nadir if neighbor_ind.nadir is not None else np.full(self.game.n_objectives, -np.inf)
                    neighbor_f = -neighbor_pt
                    
                    weight = self.weights[neighbor_idx]
                    
                    val_neighbor = self.decomposition.do(neighbor_f.reshape(1,-1), weights=weight, ideal_point=self.ideal_point)[0]
                    val_child = self.decomposition.do(child_f.reshape(1,-1), weights=weight, ideal_point=self.ideal_point)[0]
                    
                    if val_child < val_neighbor:
                        population[neighbor_idx] = child
                        
        return population
