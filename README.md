# Multiobjective Stackelberg Security Games (MO-SSG)

This repository contains the implementation of Evolutionary Algorithms (EA) and Coevolutionary Algorithms (CoEA) for solving Multiobjective Stackelberg Security Games

## Structure

*   **`algorithms.py`**: Contains implementations:
    *   `WCD_EA`: Standard Evolutionary Algorithm.
    *   `WCD_CoEA`: Coevolutionary Algorithm (Leader vs. Follower populations).
*   **`baselines.py`**: Standard MOEA baselines adapted for MO-SSGs using `pymoo`:
    *   `NSGA3_Solver`: NSGA-III baseline.
    *   `MOEAD_Solver`: MOEA/D baseline.
*   **`flipit_env.py`**: Implementation of the FlipIt game environment.
*   **`tow_env.py`**: Implementation of the Tug-of-War game environment.
*   **`mo_utils.py`**: Utility functions for multiobjective optimization (Pareto dominance, worst-case dominance, non-dominated sorting).
*   **`run_flipit.py`**: Main script for running FlipIt experiments.
*   **`run_tow.py`**: Main script for running Tug-of-War experiments.
*   **`flipit-games/`**: Directory containing benchmark game instances in JSON format.

## Requirements

The baseline algorithms (NSGA-III and MOEA/D) depend on the `pymoo` library. You can install it using pip:
```bash
pip install pymoo
```

## Usage

### Running FlipIt Experiments (`run_flipit.py`)

Use `run_flipit.py` to execute experiments on FlipIt game instances. You can specify the algorithm, population size, and duration.

**Arguments:**
*   `--game_file` (required): Path to the game JSON file.
*   `--algo`: Algorithm to use. Options: `EA` (default), `CoEA`, `NSGA3`, `MOEAD`.
*   `--objectives`: Number of objectives (game dimension).
*   `--pop_size`: Population size.
*   `--generations`: Number of generations (for EA).
*   `--epochs`: Number of epochs (for CoEA).
*   `--gen_per_epoch`: Generations per epoch (for CoEA).

### Running Tug-of-War Experiments (`run_tow.py`)

Use `run_tow.py` to execute experiments on the Tug-of-War game.

**Arguments:**
*   `--discretization`: Level of discretization (`coarse`, `medium`, `fine`).
*   `--dimension`: Problem dimension/objectives (`2` or `3`).
*   `--algo`: Algorithm to use. Options: `EA` (default), `CoEA`, `NSGA3`, `MOEAD`.
*   `--pop_size`: Population size.
*   `--generations`: Number of generations (for EA).
*   `--epochs`: Number of epochs (for CoEA).
*   `--gen_per_epoch`: Generations per epoch (for CoEA).
