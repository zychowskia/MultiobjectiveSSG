import numpy as np
import math

class TugOfWarGame:
    def __init__(self, discretization="medium", dimension=2):
        self.n_objectives = dimension
        self.rounds = 1 
        
        if discretization == "coarse":
            self.n_angles = 12   
            self.n_force = 5     
        elif discretization == "medium":
            self.n_angles = 36   
            self.n_force = 10    
        elif discretization == "fine":
            self.n_angles = 72   
            self.n_force = 20    
        else:
            raise ValueError("Unknown discretization level")

        self.pure_strategies = self._generate_strategies(dimension)
        
    def _generate_strategies(self, dimension):
        strategies = []
        
        forces = [i / self.n_force for i in range(1, self.n_force + 1)]
        
        if dimension == 2:
            angles = [i * (360.0 / self.n_angles) for i in range(self.n_angles)]
            
            for f in forces:
                for theta_deg in angles:
                    theta_rad = math.radians(theta_deg)
                    fx = f * math.cos(theta_rad)
                    fy = f * math.sin(theta_rad)
                    strategies.append(np.array([fx, fy]))
                    
        elif dimension == 3:
            azimuths = [i * (360.0 / self.n_angles) for i in range(self.n_angles)]
            polars = [i * (180.0 / (self.n_angles // 2)) for i in range(self.n_angles // 2 + 1)]
            
            for f in forces:
                for theta in azimuths:
                    for phi in polars:
                        t_rad = math.radians(theta)
                        p_rad = math.radians(phi)
                        
                        fx = f * math.sin(p_rad) * math.cos(t_rad)
                        fy = f * math.sin(p_rad) * math.sin(t_rad)
                        fz = f * math.cos(p_rad)
                        strategies.append(np.array([fx, fy, fz]))
                        
        return strategies

    def play(self, def_strategy_vec, att_strategy_vec):
        outcome_r = def_strategy_vec - att_strategy_vec
        return outcome_r, -outcome_r