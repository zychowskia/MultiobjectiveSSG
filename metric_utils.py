import numpy as np
from scipy.spatial.distance import cdist

def calculate_igd(obtained_points, reference_points):
    """
    Calculates the Inverted Generational Distance (IGD).
    IGD is the average distance from each point in the reference set 
    to its nearest point in the obtained set.
    """
    if len(obtained_points) == 0:
        return float('inf')
        
    obtained_points = np.array(obtained_points)
    reference_points = np.array(reference_points)
    
    # cdist computes distance between each pair of points
    distances = cdist(reference_points, obtained_points)
    min_distances = np.min(distances, axis=1)
    
    return np.mean(min_distances)

def generate_tow_reference_set(dimension, num_points=1000):
    """
    Generates the analytical reference set for the Tug-of-War game.
    For K=2, it's the unit arc in the first quadrant.
    For K=3, it's the unit spherical surface in the positive octant.
    """
    if dimension == 2:
        angles = np.linspace(0, np.pi / 2, num_points)
        return np.column_stack((np.cos(angles), np.sin(angles)))
    elif dimension == 3:
        # Use a grid to generate roughly num_points
        n = int(np.sqrt(num_points))
        theta = np.linspace(0, np.pi/2, n)
        phi = np.linspace(0, np.pi/2, n)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        
        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.sin(phi_grid) * np.sin(theta_grid)
        z = np.cos(phi_grid)
        return np.column_stack((x.flatten(), y.flatten(), z.flatten()))
    else:
        raise ValueError(f"Unsupported dimension for ToW reference set: {dimension}")

def calculate_expected_force(individual, game):
    """
    Extracts the expected force vector for a Leader strategy in ToW.
    """
    expected_force = np.zeros(game.n_objectives)
    for s_idx, prob in zip(individual.strategies, individual.probabilities):
        expected_force += prob * np.array(game.pure_strategies[s_idx])
    return expected_force

def extract_nadir_points(population):
    """
    Extracts the nadir points from the worst-case sets of a population.
    """
    points = []
    for ind in population:
        if ind.nadir is not None:
            points.append(ind.nadir)
    return np.array(points)
