import numpy as np
import random
from mo_utils import pareto_prune, worst_case_set, calculate_nadir_point, non_dominated_sort_leader

class LeaderStrategy:
    def __init__(self, strategies=None, probabilities=None):
        if strategies is None:
            self.strategies = []
            self.probabilities = []
        else:
            self.strategies = strategies
            self.probabilities = probabilities
        
        self.wc_set = None
        self.nadir = None

    def normalize(self):
        total = sum(self.probabilities)
        if total > 0:
            self.probabilities = [p / total for p in self.probabilities]
        else:
            n = len(self.strategies)
            self.probabilities = [1.0/n] * n

class WCD_EA:
    def __init__(self, game, pop_size=20, generations=20):
        self.game = game
        self.pop_size = pop_size
        self.generations = generations
        self.pure_strat_indices = list(range(len(self.game.pure_strategies)))
        
        print(f"Precomputing payoff matrix for {len(self.pure_strat_indices)}^2 interactions...")
        self.payoff_matrix_L = np.zeros((len(self.pure_strat_indices), len(self.pure_strat_indices), game.n_objectives))
        self.payoff_matrix_F = np.zeros_like(self.payoff_matrix_L)
        
        for r_idx in self.pure_strat_indices:
            for c_idx in self.pure_strat_indices:
                u_l, u_f = self.game.play(self.game.pure_strategies[r_idx], self.game.pure_strategies[c_idx])
                self.payoff_matrix_L[r_idx, c_idx] = u_l
                self.payoff_matrix_F[r_idx, c_idx] = u_f

    def evaluate_individual(self, individual):
        weights = np.array(individual.probabilities)
        idxs = individual.strategies
        
        sub_matrix_F = self.payoff_matrix_F[idxs] 
        expected_F = np.average(sub_matrix_F, axis=0, weights=weights)
        
        rat_indices = pareto_prune(expected_F)
        
        sub_matrix_L = self.payoff_matrix_L[idxs]
        expected_L = np.average(sub_matrix_L, axis=0, weights=weights)
        
        wc_set = worst_case_set(expected_L, rat_indices)
        
        individual.wc_set = wc_set
        individual.nadir = calculate_nadir_point(wc_set)

    def initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            strat_idx = random.choice(self.pure_strat_indices)
            ind = LeaderStrategy([strat_idx], [1.0])
            self.evaluate_individual(ind)
            pop.append(ind)
        return pop

    def crossover(self, p1, p2):
        new_map = {}
        
        for s, p in zip(p1.strategies, p1.probabilities):
            new_map[s] = new_map.get(s, 0) + (p / 2)
            
        for s, p in zip(p2.strategies, p2.probabilities):
            new_map[s] = new_map.get(s, 0) + (p / 2)
            
        strategies = []
        probs = []
        
        if not new_map:
            best_s = random.choice(p1.strategies)
            new_map[best_s] = 1.0
            
        best_s = max(new_map, key=new_map.get)
        
        for s, p in new_map.items():
            if s == best_s:
                strategies.append(s)
                probs.append(p)
            else:
                prob_remove = (1.0 - p)**2
                if random.random() > prob_remove:
                    strategies.append(s)
                    probs.append(p)
                    
        child = LeaderStrategy(strategies, probs)
        child.normalize()
        return child

    def mutate(self, ind):
        if random.random() < 0.2:
            if hasattr(self.game, 'actions') and hasattr(self.game, 'rounds'):
                idx_in_mix = random.randint(0, len(ind.strategies) - 1)
                old_strat_idx = ind.strategies[idx_in_mix]
                old_strat = self.game.pure_strategies[old_strat_idx]
                
                t_cut = random.randint(0, self.game.rounds - 1)
                
                new_suffix = [random.choice(self.game.actions) for _ in range(self.game.rounds - t_cut)]
                new_strat_tuple = old_strat[:t_cut] + tuple(new_suffix)
                
                try:
                    new_strat_idx = self.game.pure_strategies.index(new_strat_tuple)
                    ind.strategies[idx_in_mix] = new_strat_idx
                    self._consolidate_strategy(ind)
                except ValueError:
                    pass
            else:
                 idx_in_mix = random.randint(0, len(ind.strategies) - 1)
                 ind.strategies[idx_in_mix] = random.choice(self.pure_strat_indices)
                 self._consolidate_strategy(ind)

    def _consolidate_strategy(self, ind):
        mapping = {}
        for s, p in zip(ind.strategies, ind.probabilities):
            mapping[s] = mapping.get(s, 0) + p
        ind.strategies = list(mapping.keys())
        ind.probabilities = list(mapping.values())

    def run(self):
        population = self.initialize_population()
        pop_ranks = {i: 0 for i in range(self.pop_size)}
        pop_distances = {i: 0.0 for i in range(self.pop_size)}
        
        fronts = non_dominated_sort_leader(population)
        for r, front in enumerate(fronts):
            distances = self._get_crowding_distances(population, front)
            for idx in front:
                pop_ranks[idx] = r
                pop_distances[idx] = distances[idx]
        
        for gen in range(self.generations):
            print(f"Generation {gen+1}/{self.generations}")
            offspring = []
            
            while len(offspring) < self.pop_size:
                i, j = random.sample(range(self.pop_size), 2)
                p1 = population[i] if pop_ranks[i] < pop_ranks[j] or (pop_ranks[i] == pop_ranks[j] and pop_distances[i] > pop_distances[j]) else population[j]
                
                i, j = random.sample(range(self.pop_size), 2)
                p2 = population[i] if pop_ranks[i] < pop_ranks[j] or (pop_ranks[i] == pop_ranks[j] and pop_distances[i] > pop_distances[j]) else population[j]
                
                child = self.crossover(p1, p2)
                self.mutate(child)
                self.evaluate_individual(child)
                offspring.append(child)
                
            combined = population + offspring
            
            fronts = non_dominated_sort_leader(combined)
            
            next_pop = []
            next_ranks = {}
            next_distances = {}
            for r, front in enumerate(fronts):
                distances = self._get_crowding_distances(combined, front)
                if len(next_pop) + len(front) <= self.pop_size:
                    for idx in front:
                        next_ranks[len(next_pop)] = r
                        next_distances[len(next_pop)] = distances[idx]
                        next_pop.append(combined[idx])
                else:
                    needed = self.pop_size - len(next_pop)
                    sorted_front = sorted(front, key=lambda i: distances[i], reverse=True)
                    for i in range(needed):
                        idx = sorted_front[i]
                        next_ranks[len(next_pop)] = r
                        next_distances[len(next_pop)] = distances[idx]
                        next_pop.append(combined[idx])
                    break
            
            population = next_pop
            pop_ranks = next_ranks
            pop_distances = next_distances
            
        return population

    def _get_crowding_distances(self, population, front_indices):
        l = len(front_indices)
        distances = {idx: 0.0 for idx in front_indices}
        if l == 0: return distances
        
        points = []
        for idx in front_indices:
            pt = population[idx].nadir
            if pt is None:
                pt = np.full(self.game.n_objectives, -np.inf)
            points.append(pt)
        points = np.array(points)
        
        for m in range(self.game.n_objectives):
            sorted_indices = np.argsort(points[:, m])
            
            distances[front_indices[sorted_indices[0]]] = float('inf')
            distances[front_indices[sorted_indices[-1]]] = float('inf')
            
            m_min = points[sorted_indices[0], m]
            m_max = points[sorted_indices[-1], m]
            diff = m_max - m_min
            
            if diff == 0: continue
            
            for i in range(1, l - 1):
                idx = front_indices[sorted_indices[i]]
                prev_val = points[sorted_indices[i-1], m]
                next_val = points[sorted_indices[i+1], m]
                distances[idx] += (next_val - prev_val) / diff
                
        return distances

    def _crowding_distance_sort(self, population, front_indices):
        distances = self._get_crowding_distances(population, front_indices)
        return sorted(front_indices, key=lambda i: distances[i], reverse=True)


class WCD_CoEA:
    def __init__(self, game, pop_size=20, epochs=20, gen_per_epoch=10):
        self.game = game
        self.pop_size = pop_size
        self.epochs = epochs
        self.gen_per_epoch = gen_per_epoch
        self.pure_strat_indices = list(range(len(self.game.pure_strategies)))
        
        print("Precomputing payoff matrix...")
        self.payoff_matrix_L = np.zeros((len(self.pure_strat_indices), len(self.pure_strat_indices), game.n_objectives))
        self.payoff_matrix_F = np.zeros_like(self.payoff_matrix_L)
        
        for r_idx in self.pure_strat_indices:
            for c_idx in self.pure_strat_indices:
                u_l, u_f = self.game.play(self.game.pure_strategies[r_idx], self.game.pure_strategies[c_idx])
                self.payoff_matrix_L[r_idx, c_idx] = u_l
                self.payoff_matrix_F[r_idx, c_idx] = u_f

    def initialize_populations(self):
        P_L = []
        for _ in range(self.pop_size):
            strat_idx = random.choice(self.pure_strat_indices)
            P_L.append(LeaderStrategy([strat_idx], [1.0]))
            
        P_F = []
        for _ in range(self.pop_size):
            P_F.append(random.choice(self.pure_strat_indices))
            
        return P_L, P_F

    def evaluate_leader(self, population_L, population_F):
        for ind in population_L:
            weights = np.array(ind.probabilities)
            idxs = ind.strategies
            
            sub_matrix_L = self.payoff_matrix_L[idxs][:, population_F]
            expected_L = np.average(sub_matrix_L, axis=0, weights=weights)
            
            sub_matrix_F = self.payoff_matrix_F[idxs][:, population_F]
            expected_F = np.average(sub_matrix_F, axis=0, weights=weights)
            
            rat_indices_local = pareto_prune(expected_F)
            
            wc_set = worst_case_set(expected_L, rat_indices_local)
            
            ind.wc_set = wc_set
            ind.nadir = calculate_nadir_point(wc_set)
            
    def evaluate_follower(self, population_F, population_L):
        fitness_values = []
        
        for f_idx in population_F:
            avg_u_f = np.zeros(self.game.n_objectives)
            
            for l_ind in population_L:
                weights = np.array(l_ind.probabilities)
                idxs = l_ind.strategies
                
                sub_vecs_F = self.payoff_matrix_F[idxs, f_idx]
                
                u_f = np.average(sub_vecs_F, axis=0, weights=weights)
                avg_u_f += u_f
            
            avg_u_f /= len(population_L)
            fitness_values.append(avg_u_f)
            
        return fitness_values

    def crossover_leader(self, p1, p2):
         new_map = {}
         for s, p in zip(p1.strategies, p1.probabilities):
            new_map[s] = new_map.get(s, 0) + (p / 2)
         for s, p in zip(p2.strategies, p2.probabilities):
            new_map[s] = new_map.get(s, 0) + (p / 2)
            
         strategies, probs = [], []
         best_s = max(new_map, key=new_map.get)
         
         for s, p in new_map.items():
            if s == best_s:
                strategies.append(s)
                probs.append(p)
            else:
                if random.random() > (1.0 - p)**2:
                     strategies.append(s)
                     probs.append(p)
                     
         child = LeaderStrategy(strategies, probs)
         child.normalize()
         return child

    def mutate_leader(self, ind):
        if random.random() < 0.2:
             if hasattr(self.game, 'actions') and hasattr(self.game, 'rounds'):
                 idx_in_mix = random.randint(0, len(ind.strategies) - 1)
                 old_strat_idx = ind.strategies[idx_in_mix]
                 old_strat = self.game.pure_strategies[old_strat_idx]
                 t_cut = random.randint(0, self.game.rounds - 1)
                 new_suffix = [random.choice(self.game.actions) for _ in range(self.game.rounds - t_cut)]
                 new_strat_tuple = old_strat[:t_cut] + tuple(new_suffix)
                 try:
                     new_strat_idx = self.game.pure_strategies.index(new_strat_tuple)
                     ind.strategies[idx_in_mix] = new_strat_idx
                     mapping = {}
                     for s, p in zip(ind.strategies, ind.probabilities):
                         mapping[s] = mapping.get(s, 0) + p
                     ind.strategies = list(mapping.keys())
                     ind.probabilities = list(mapping.values())
                 except ValueError: pass
             else:
                 idx_in_mix = random.randint(0, len(ind.strategies) - 1)
                 ind.strategies[idx_in_mix] = random.choice(self.pure_strat_indices)
                 # Consolidate
                 mapping = {}
                 for s, p in zip(ind.strategies, ind.probabilities):
                     mapping[s] = mapping.get(s, 0) + p
                 ind.strategies = list(mapping.keys())
                 ind.probabilities = list(mapping.values())

    def crossover_follower(self, f_idx1, f_idx2):
        if hasattr(self.game, 'actions') and hasattr(self.game, 'rounds'):
            s1 = self.game.pure_strategies[f_idx1]
            s2 = self.game.pure_strategies[f_idx2]
            
            common_indices = [i for i, (a, b) in enumerate(zip(s1, s2)) if a == b]
            if not common_indices:
                return f_idx1
                
            swap_point = common_indices[0] + 1
            if swap_point >= len(s1):
                return f_idx1
                
            new_s = s1[:swap_point] + s2[swap_point:]
            try:
                return self.game.pure_strategies.index(new_s)
            except ValueError:
                return f_idx1
        else:
             # Generic crossover: Random parent
             return random.choice([f_idx1, f_idx2])

    def mutate_follower(self, f_idx):
        if random.random() < 0.2:
            if hasattr(self.game, 'actions') and hasattr(self.game, 'rounds'):
                s = self.game.pure_strategies[f_idx]
                t_cut = random.randint(0, self.game.rounds - 1)
                new_suffix = [random.choice(self.game.actions) for _ in range(self.game.rounds - t_cut)]
                new_s = s[:t_cut] + tuple(new_suffix)
                try:
                    return self.game.pure_strategies.index(new_s)
                except ValueError:
                    return f_idx
            else:
                 return random.choice(self.pure_strat_indices)
        return f_idx

    def get_crowding_distance_leader(self, population, front):
        l = len(front)
        distances = {idx: 0.0 for idx in front}
        if l == 0: return distances
        points = []
        for idx in front:
            pt = population[idx].nadir
            if pt is None: pt = np.full(self.game.n_objectives, -np.inf)
            points.append(pt)
        points = np.array(points)
        
        for m in range(self.game.n_objectives):
            sorted_indices = np.argsort(points[:, m])
            distances[front[sorted_indices[0]]] = float('inf')
            distances[front[sorted_indices[-1]]] = float('inf')
            diff = points[sorted_indices[-1], m] - points[sorted_indices[0], m]
            if diff == 0: continue
            for i in range(1, l - 1):
                distances[front[sorted_indices[i]]] += (points[sorted_indices[i+1],m] - points[sorted_indices[i-1],m])/diff
        return distances

    def crowding_distance_leader(self, population, front):
        distances = self.get_crowding_distance_leader(population, front)
        return sorted(front, key=lambda i: distances[i], reverse=True)

    def get_crowding_distance_follower(self, fitness_vectors, front):
        l = len(front)
        distances = {idx: 0.0 for idx in front}
        if l == 0: return distances
        points = np.array([fitness_vectors[i] for i in front])
        
        for m in range(self.game.n_objectives):
            sorted_indices = np.argsort(points[:, m])
            distances[front[sorted_indices[0]]] = float('inf')
            distances[front[sorted_indices[-1]]] = float('inf')
            diff = points[sorted_indices[-1], m] - points[sorted_indices[0], m]
            if diff == 0: continue
            for i in range(1, l - 1):
                 distances[front[sorted_indices[i]]] += (points[sorted_indices[i+1],m] - points[sorted_indices[i-1],m])/diff
        
        return distances

    def crowding_distance_follower(self, fitness_vectors, front):
        distances = self.get_crowding_distance_follower(fitness_vectors, front)
        return sorted(front, key=lambda i: distances[i], reverse=True)

    def run(self):
        from mo_utils import non_dominated_sort_basic
        P_L, P_F = self.initialize_populations()
        
        self.evaluate_leader(P_L, P_F)
        
        ranks_L = {i: 0 for i in range(self.pop_size)}
        dist_L = {i: 0.0 for i in range(self.pop_size)}
        ranks_F = {i: 0 for i in range(self.pop_size)}
        dist_F = {i: 0.0 for i in range(self.pop_size)}
        
        fronts_L = non_dominated_sort_leader(P_L)
        for r, front in enumerate(fronts_L):
            dists = self.get_crowding_distance_leader(P_L, front)
            for idx in front:
                ranks_L[idx] = r
                dist_L[idx] = dists[idx]
                
        fitness_F = self.evaluate_follower(P_F, P_L)
        fronts_F = non_dominated_sort_basic(fitness_F)
        for r, front in enumerate(fronts_F):
            dists = self.get_crowding_distance_follower(fitness_F, front)
            for idx in front:
                ranks_F[idx] = r
                dist_F[idx] = dists[idx]
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            for gen in range(self.gen_per_epoch):
                offspring_F = []
                while len(offspring_F) < self.pop_size:
                    i, j = random.sample(range(self.pop_size), 2)
                    p1 = P_F[i] if ranks_F[i] < ranks_F[j] or (ranks_F[i] == ranks_F[j] and dist_F[i] > dist_F[j]) else P_F[j]
                    
                    i, j = random.sample(range(self.pop_size), 2)
                    p2 = P_F[i] if ranks_F[i] < ranks_F[j] or (ranks_F[i] == ranks_F[j] and dist_F[i] > dist_F[j]) else P_F[j]
                    
                    child = self.crossover_follower(p1, p2)
                    child = self.mutate_follower(child)
                    offspring_F.append(child)
                
                combined_F = P_F + offspring_F
                fitness_F = self.evaluate_follower(combined_F, P_L)
                
                fronts = non_dominated_sort_basic(fitness_F)
                next_F = []
                next_ranks_F = {}
                next_dist_F = {}
                for r, front in enumerate(fronts):
                    dists = self.get_crowding_distance_follower(fitness_F, front)
                    if len(next_F) + len(front) <= self.pop_size:
                        for idx in front:
                            next_ranks_F[len(next_F)] = r
                            next_dist_F[len(next_F)] = dists[idx]
                            next_F.append(combined_F[idx])
                    else:
                        sorted_front = sorted(front, key=lambda i: dists[i], reverse=True)
                        needed = self.pop_size - len(next_F)
                        for i in range(needed):
                            idx = sorted_front[i]
                            next_ranks_F[len(next_F)] = r
                            next_dist_F[len(next_F)] = dists[idx]
                            next_F.append(combined_F[idx])
                        break
                P_F = next_F
                ranks_F = next_ranks_F
                dist_F = next_dist_F
            
            for gen in range(self.gen_per_epoch):
                offspring_L = []
                while len(offspring_L) < self.pop_size:
                    i, j = random.sample(range(self.pop_size), 2)
                    p1 = P_L[i] if ranks_L[i] < ranks_L[j] or (ranks_L[i] == ranks_L[j] and dist_L[i] > dist_L[j]) else P_L[j]
                    
                    i, j = random.sample(range(self.pop_size), 2)
                    p2 = P_L[i] if ranks_L[i] < ranks_L[j] or (ranks_L[i] == ranks_L[j] and dist_L[i] > dist_L[j]) else P_L[j]
                    
                    child = self.crossover_leader(p1, p2)
                    self.mutate_leader(child)
                    offspring_L.append(child)
                    
                combined_L = P_L + offspring_L
                self.evaluate_leader(combined_L, P_F)
                
                fronts = non_dominated_sort_leader(combined_L)
                next_L = []
                next_ranks_L = {}
                next_dist_L = {}
                for r, front in enumerate(fronts):
                    dists = self.get_crowding_distance_leader(combined_L, front)
                    if len(next_L) + len(front) <= self.pop_size:
                         for idx in front:
                             next_ranks_L[len(next_L)] = r
                             next_dist_L[len(next_L)] = dists[idx]
                             next_L.append(combined_L[idx])
                    else:
                        needed = self.pop_size - len(next_L)
                        sorted_front = sorted(front, key=lambda i: dists[i], reverse=True)
                        for i in range(needed):
                            idx = sorted_front[i]
                            next_ranks_L[len(next_L)] = r
                            next_dist_L[len(next_L)] = dists[idx]
                            next_L.append(combined_L[idx])
                        break
                P_L = next_L
                ranks_L = next_ranks_L
                dist_L = next_dist_L
                
        return P_L