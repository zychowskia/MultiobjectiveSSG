
import argparse
import json
import numpy as np
import os
from flipit_env import FlipItGame
from algorithms import WCD_EA, WCD_CoEA
from baselines import NSGA3_Solver, MOEAD_Solver
from mo_utils import non_dominated_sort_leader
from metric_utils import calculate_igd, extract_nadir_points

def main():
    parser = argparse.ArgumentParser(description="Run MO-SSG Experiments")
    parser.add_argument("--game_file", type=str, required=True, help="Path to the game JSON file")
    parser.add_argument("--objectives", type=int, default=2, help="Number of objectives (2, 4, 8)")
    parser.add_argument("--algo", type=str, choices=["EA", "CoEA", "NSGA3", "MOEAD"], default="EA", help="Algorithm to run")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size")
    parser.add_argument("--generations", type=int, default=20, help="Generations (for EA)")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs (for CoEA)")
    parser.add_argument("--gen_per_epoch", type=int, default=5, help="Generations per epoch (for CoEA)")
    parser.add_argument("--reference_set", type=str, default=None, help="Path to a numpy .npy file containing the reference set (nadir points) for IGD calculation")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.game_file):
        print(f"Error: File '{args.game_file}' not found.")
        return

    with open(args.game_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loading game from {args.game_file}...")
    game = FlipItGame(data, n_objectives=args.objectives)
    print(f"Total pure strategies per player: {len(game.pure_strategies)}")
    
    final_population = []
    
    if args.algo == "EA":
        print(f"Initializing EA Solver (Pop={args.pop_size}, Gens={args.generations})...")
        solver = WCD_EA(game, pop_size=args.pop_size, generations=args.generations)
        final_population = solver.run()
        
    elif args.algo == "CoEA":
        print(f"Initializing CoEA Solver (Pop={args.pop_size}, Epochs={args.epochs}, Gen/E={args.gen_per_epoch})...")
        solver = WCD_CoEA(game, pop_size=args.pop_size, epochs=args.epochs, gen_per_epoch=args.gen_per_epoch)
        final_population = solver.run()
        
    elif args.algo == "NSGA3":
        print(f"Initializing NSGA3 Solver (Pop={args.pop_size}, Gens={args.generations})...")
        solver = NSGA3_Solver(game, pop_size=args.pop_size, generations=args.generations)
        final_population = solver.run()
        
    elif args.algo == "MOEAD":
        print(f"Initializing MOEAD Solver (Pop={args.pop_size}, Gens={args.generations})...")
        solver = MOEAD_Solver(game, pop_size=args.pop_size, generations=args.generations)
        final_population = solver.run()
    
    print("\n--- Results ---")
    print("Final Non-Dominated Leader Strategies:")
    fronts = non_dominated_sort_leader(final_population)
    if not fronts:
        print("No solutions found.")
        return
        
    non_dom_indices = fronts[0]
    
    for i, idx in enumerate(non_dom_indices):
        ind = final_population[idx]
        print(f"Strategy {i+1}:")
        print(f"  Support Size: {len(ind.strategies)}")
        if ind.nadir is not None:
             print(f"  Worst-Case Outcome Set Nadir: {np.round(ind.nadir, 4)}")
        else:
             print("  Worst-Case Outcome Set Nadir: None")
             
        best_comp_idx = np.argmax(ind.probabilities)
        best_strat_idx = ind.strategies[best_comp_idx]
        print(f"  Top Component: {game.pure_strategies[best_strat_idx]} (prob={ind.probabilities[best_comp_idx]:.4f})")
        print("-" * 20)

    print("\n--- IGD Calculation ---")
    if args.reference_set:
        if os.path.exists(args.reference_set):
            reference_set = np.load(args.reference_set)
            
            non_dom_population = [final_population[i] for i in non_dom_indices]
            obtained_points = extract_nadir_points(non_dom_population)
            
            if len(obtained_points) > 0:
                igd_value = calculate_igd(obtained_points, reference_set)
                print(f"IGD: {igd_value:.6f}")
            else:
                print("No valid nadir points found to calculate IGD.")
        else:
            print(f"Error: Reference set file '{args.reference_set}' not found.")
    else:
        print("IGD not calculated. Provide a reference set using --reference_set <file.npy>")

if __name__ == "__main__":
    main()