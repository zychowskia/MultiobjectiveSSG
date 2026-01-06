
import argparse
import numpy as np
from tow_env import TugOfWarGame
from algorithms import MO_SSG_Solver, MO_CoSSG_Solver
from mo_utils import non_dominated_sort_leader

def main():
    parser = argparse.ArgumentParser(description="Run Tug-of-War MO-SSG Experiments")
    parser.add_argument("--discretization", type=str, choices=["coarse", "medium", "fine"], default="medium", help="Discretization level")
    parser.add_argument("--dimension", type=int, choices=[2, 3], default=2, help="Problem dimension (K)")
    parser.add_argument("--algo", type=str, choices=["EA", "CoEA"], default="EA", help="Algorithm to run")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--generations", type=int, default=50, help="Generations (for EA)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs (for CoEA)")
    parser.add_argument("--gen_per_epoch", type=int, default=5, help="Generations per epoch (for CoEA)")
    
    args = parser.parse_args()

    print(f"--- Setting up Tug of War Game (Level: {args.discretization}, K={args.dimension}) ---")
    game = TugOfWarGame(discretization=args.discretization, dimension=args.dimension)
    
    print(f"Number of pure strategies per player: {len(game.pure_strategies)}")
    
    final_population = []
    
    if args.algo == "EA":
        print(f"Initializing EA Solver (Pop={args.pop_size}, Gens={args.generations})...")
        solver = MO_SSG_Solver(game, pop_size=args.pop_size, generations=args.generations)
        final_population = solver.run()
        
    elif args.algo == "CoEA":
        print(f"Initializing CoEA Solver (Pop={args.pop_size}, Epochs={args.epochs}, Gen/E={args.gen_per_epoch})...")
        solver = MO_CoSSG_Solver(game, pop_size=args.pop_size, epochs=args.epochs, gen_per_epoch=args.gen_per_epoch)
        final_population = solver.run()
    
    # Analysis of results
    print("\n--- Results Analysis ---")
    fronts = non_dominated_sort_leader(final_population)
    if not fronts:
        print("No non-dominated strategies found.")
        return

    best_indices = fronts[0]
    
    print(f"Number of non-dominated mixed strategies found: {len(best_indices)}")
    
    first_quadrant_count = 0
    max_force_count = 0
    total_strategies = 0
    
    print("\nSample Strategies:")
    for i, idx in enumerate(best_indices):
        ind = final_population[idx]
        
        # Analyze the pure components of this mixed strategy
        for s_idx, prob in zip(ind.strategies, ind.probabilities):
            if prob < 0.01: continue 
            
            total_strategies += 1
            vec = game.pure_strategies[s_idx]
            
            magnitude = np.linalg.norm(vec)
            if args.dimension == 2:
                angle_deg = np.degrees(np.arctan2(vec[1], vec[0]))
                if angle_deg < 0: angle_deg += 360
            else:
                # 3D visualization simplified
                angle_deg = 0 # Placeholder for 3D
            
            if magnitude >= 0.99: 
                max_force_count += 1
            
            if args.dimension == 2 and 0 <= angle_deg <= 90:
                first_quadrant_count += 1
                
            if i < 3 and prob > 0.1: # Print details for first few solutions
                if args.dimension == 2:
                    print(f"  Sol {i+1}: Angle={angle_deg:.1f} deg, Force={magnitude:.2f}, Prob={prob:.2f}")
                else:
                    print(f"  Sol {i+1}: Vec={vec}, Force={magnitude:.2f}, Prob={prob:.2f}")

    print("-" * 30)
    if total_strategies > 0:
        print(f"Components with Max Force (~1.0): {max_force_count}/{total_strategies} ({max_force_count/total_strategies:.1%})")
        if args.dimension == 2:
            print(f"Components in 1st Quadrant (0-90 deg): {first_quadrant_count}/{total_strategies} ({first_quadrant_count/total_strategies:.1%})")
        
if __name__ == "__main__":
    main()