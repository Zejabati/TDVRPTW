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


