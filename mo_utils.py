import numpy as np

def is_dominated(v1, v2):
    return np.all(v2 >= v1) and np.any(v2 > v1)

def pareto_prune(vectors):
    indices = []
    for i, v in enumerate(vectors):
        dominated = False
        for j, other in enumerate(vectors):
            if i != j and is_dominated(v, other):
                dominated = True
                break
        if not dominated:
            indices.append(i)
    return indices

def pareto_min_indices(vectors):
    indices = []
    for i, v in enumerate(vectors):
        dominated = False
        for j, other in enumerate(vectors):
            if i != j:
                if np.all(other <= v) and np.any(other < v):
                    dominated = True
                    break
        if not dominated:
            indices.append(i)
    return indices

def worst_case_set(leader_payoffs, follower_indices_rat):
    relevant_payoffs = [leader_payoffs[i] for i in follower_indices_rat]
    if not relevant_payoffs:
        return []
    
    min_indices = pareto_min_indices(relevant_payoffs)
    return [relevant_payoffs[i] for i in min_indices]

def worst_case_dominance(wc_set_a, wc_set_b):
    if not wc_set_a or not wc_set_b:
        return False
        
    for val_a in wc_set_a:
        is_dominated_by_b = False
        for val_b in wc_set_b:
            if np.all(val_b <= val_a) and np.any(val_b < val_a):
                is_dominated_by_b = True
                break
        if not is_dominated_by_b:
            return False
    return True

def calculate_nadir_point(vector_set):
    if not vector_set:
        return None
    stacked = np.stack(vector_set)
    return np.min(stacked, axis=0)

def non_dominated_sort_leader(population):
    fronts = []
    domination_count = {i: 0 for i in range(len(population))}
    dominated_solutions = {i: [] for i in range(len(population))}
    
    current_front = []
    
    for i in range(len(population)):
        for j in range(len(population)):
            if i == j: continue
            if worst_case_dominance(population[i].wc_set, population[j].wc_set):
                dominated_solutions[i].append(j)
            elif worst_case_dominance(population[j].wc_set, population[i].wc_set):
                domination_count[i] += 1
        
        if domination_count[i] == 0:
            current_front.append(i)
            
    fronts.append(current_front)
    
    idx = 0
    while idx < len(fronts):
        next_front = []
        for i in fronts[idx]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        idx += 1
        
    return fronts

def non_dominated_sort_basic(vectors):
    n = len(vectors)
    fronts = []
    domination_count = {i: 0 for i in range(n)}
    dominated_solutions = {i: [] for i in range(n)}
    
    current_front = []
    
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if is_dominated(vectors[j], vectors[i]): 
                dominated_solutions[i].append(j)
            elif is_dominated(vectors[i], vectors[j]): 
                domination_count[i] += 1
        
        if domination_count[i] == 0:
            current_front.append(i)
            
    fronts.append(current_front)
    
    idx = 0
    while idx < len(fronts):
        next_front = []
        for i in fronts[idx]:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        if not next_front:
            break
        fronts.append(next_front)
        idx += 1
        
    return fronts