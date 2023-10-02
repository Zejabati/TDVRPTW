# TDVRPTW


## Time Dependent Vehicle Routing Problem with Time Window _ Hybrid Genetic Algorithm


This project presents a Python implementation of a solver designed to address the Time-Dependent Vehicle Routing Problem with Time Windows (TDVRPTW).
The solver employs a hybrid algorithm that integrates Genetic Algorithms, Local Search, and Large Neighborhood Search techniques. 

## Key Features

- **Genetic Algorithm (GA):** The solver uses a Genetic Algorithm as the core optimization technique. It iteratively evolves a population of solutions to find better routes.

- **Local Search:** Local search techniques are applied to improve the quality of solutions by exploring the neighborhood of the current solutions.

- **Large Neighborhood Search (LNS):** LNS is employed to perform large-scale explorations of solution neighborhoods, aiding in finding diverse and potentially better solutions.

- **Time-Dependent Travel Times:** Travel times between locations are estimated based on a discrete speed function derived from the research paper "A Hybrid Algorithm for Time-Dependent Vehicle Routing Problem with Time Windows" by Binbin Pan, Zhenzhen Zhang, and Andrew Lim.

- **Performance Evaluation:** The solver tracks and reports the best fitness values, solution (sequence of nodes and departure time from depot for each vehicle), and execution time.


## Data
This repository includes the Solomon benchmark dataset for TDVRPTW, located in Solomon-TDVRPTW.zip. You can use these data files to test and evaluate the solver's performance.


### Algorithm Function

The heart of this repository is the `algorithm` function, which serves as the core solver for the Time-Dependent Vehicle Routing Problem with Time Windows (TDVRPTW). This function orchestrates the entire optimization process and is responsible for finding optimal or near-optimal solutions. Here's an overview of how it works:

```python
def algorithm(Problem_Genetic, v):
    # ... (Initialization and parameter setup)

    best_fit_list = []  # List to track the best fitness values
    q = 3  # Number of top solutions to consider

    # Initial Population
    population = initial_population(Problem_Genetic, size, v, p_cap, p_TW)

    best_q = population[:q]  # Initialize the best 3 solutions list

    for u in range(1, n_iter + 1):
        # ... (Iterate and perform optimization steps)

        # Record the best fitness value in the current generation
        best_fit_list.append(population[0][6])

        # ... (Continue optimization and selection with focus on top 3 solutions found so far)

    # ... (Finalize the optimization process and prepare results)

    return best_fit_list[-1], best_three_fit, best_q[0]


### Key steps of `algorithm` function

- **Initialization:** The function initializes parameters and data structures required for the optimization process, including population size, number of iterations, and constraints such as penalty for vehicle capacity and time windows.

- **Population Initialization:** It generates an initial population of solutions.

- **Main Optimization Loop:** The function enters a loop that iterates for a specified number of iterations. Within each iteration, it performs various optimization steps to evolve and improve the population. The function encompasses genetic operations, local search, and large neighborhood search to improve solution quality.

- **Fitness Tracking:** The best fitness value in each iteration is recorded in the `best_fit_list`, allowing you to track the algorithm's progress.

- **Top `q` Solutions Refinement: It prioritizes the three best solutions found in each iteration and applies local search and Large Neighborhood Search (LNS) techniques to enhance their quality, resulting in potential improvements to the solutions.

- **Result Reporting:** It returns key results, including the fitness value of the best solution found, fitness values of the top three solutions, and the best solution among the top `q` solutions.

These key functionalities collectively make up the core of the TDVRPTW solver, enabling it to find


