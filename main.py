__author__ = 'David T. Pocock'


import random
from parser import Parser
from platypus import NSGAII, Problem, Binary, GeneticAlgorithm
from pathlib import Path
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import csv
import sys
import os.path


def main():
    global first_run
    global requirements
    global customers
    global req_costs
    global weight
    global budget
    global runs
    x_mo = []
    y_mo = []
    x_so = []
    y_so = []
    x_rs = []
    y_rs = []
    mo_nsga = []
    so_algorithm = []

    pwd = os.path.abspath(os.path.dirname(__file__))
    classic_dir_path = Path("classic-nrp/")
    realistic_dir_path = Path("realistic-nrp/")
    classic_nrp = pwd / classic_dir_path / "nrp4.txt"
    realistic_nrp = pwd / realistic_dir_path / "nrp-m1.txt"

    filename = 'classic_100_09_05'
    txt_path = os.path.join(pwd, '{}'.format(filename + '.txt'))
    stdout = sys.stdout
    sys.stdout = open(txt_path, 'w')

    parser = Parser(classic_nrp)
    requirements, customers, req_costs = parser.parse_data()

    population_size = 100
    weight = 0.9
    budget = 0.5
    runs = 10000
    first_run = True
    counter = 0

    fig = plt.figure(figsize=(8, 6))

    # Runs with single parameters values
    x_mo, y_mo, mo_nsga = mo_run(population_size)
    x_so, y_so, so_algorithm = so_run(population_size)
    x_rs, y_rs = rs_run(population_size)

    # Runs with multiple parameters values
    # for b in range(5, 11):
    #     budget = b/10
    #     x_mo, y_mo, mo_nsga = mo_run(population_size)
    #
    #     for w in range(1, 10):
    #         counter += 1
    #         print("Run #", counter)
    #         weight = w/10
    #         x_so, y_so, so_algorithm = so_run(population_size)
    #
    #     x_rs, y_rs = rs_run(population_size)

    plt.scatter(x_mo, y_mo, 30, color='red', edgecolors='black')
    plt.scatter(x_so, y_so, 20, marker='s', color='green', edgecolors='black')
    plt.scatter(x_rs, y_rs, 1, color='c')

    plt.title("Comparison - Classic data set - pop_size=100, weight=0.5, budget=0.5")
    plt.xlabel("Score")
    plt.ylabel("-1*Cost")
    plt.legend(('NSGA−II', 'Single-Objective GA', 'Random search'))

    graph_path = os.path.join(pwd, '{}'.format(filename + '.pdf'))
    fig.savefig(graph_path, bbox_inches="tight")

    nsga_results = []
    nsga_zipped_sols = []
    for nsga_sol in mo_nsga.result:
        coupled_nsga = (nsga_sol.objectives[0], nsga_sol.objectives[1] * (-1)), sum(nsga_sol.variables[0]), nsga_sol.variables[0]
        nsga_results.append(coupled_nsga)
        nsga_zipped_sols.append(nsga_sol.variables[0])
    summed = [(i, sum(x)) for i, x in enumerate(zip(*nsga_zipped_sols))]
    sorted_summed = sorted(summed, key=lambda x: x[1], reverse=True)
    print("\nTop Requirements (idx, # times):", sorted_summed)
    sorted_nsga = sorted(nsga_results, key=lambda x: x[0])
    middle_nsga = (len(sorted_nsga) - 1) // 2
    middle_nsga_sol = sorted_nsga[middle_nsga]
    print("\nMiddle NSGA Solution:", middle_nsga_sol)

    soga_results = []
    for soga_sol in so_algorithm.result:
        coupled_soga = (soga_sol.objectives[0], soga_sol.constraints[0] * (-1)), sum(soga_sol.variables[0]), soga_sol.variables[0]
        soga_results.append(coupled_soga)
    sorted_soga = sorted(soga_results, key=lambda x: x[0])
    middle_soga = (len(sorted_soga) - 1) // 2
    middle_soga_sol = sorted_soga[middle_soga]
    print("\nMiddle SO GA Solution:", middle_soga_sol)

    csv_path = os.path.join(pwd, '{}'.format(filename + '.csv'))
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        x_mo.insert(0, 'MO x: Score')
        y_mo.insert(0, 'MO y: -1*Cost')
        x_so.insert(0, 'SO x: Weighted Sum')
        y_so.insert(0, 'SO y: -1*Cost')
        x_rs.insert(0, 'RS x: Score')
        y_rs.insert(0, 'RS y: -1*Cost')
        writer.writerows((x_mo, y_mo, x_so, y_so, x_rs, y_rs))


def mo_run(population_size):
    mo_problem = Problem(2, 2, 2)
    mo_problem.types[:] = Binary(len(requirements))
    mo_problem.directions[0] = Problem.MAXIMIZE
    mo_problem.directions[1] = Problem.MINIMIZE
    mo_problem.constraints[:] = "<={}".format(budget)
    mo_problem.function = multi_objective_nrp
    mo_nsga = NSGAII(mo_problem, population_size)
    mo_nsga.run(runs)
    x_mo = [solution.objectives[0] for solution in mo_nsga.result]
    y_mo = [solution.objectives[1] * (-1) for solution in mo_nsga.result]
    return x_mo, y_mo, mo_nsga


def so_run(population_size):
    so_problem = Problem(1, 1, 1)
    so_problem.types[:] = Binary(len(requirements))
    so_problem.directions[:] = Problem.MAXIMIZE
    so_problem.constraints[:] = "<={}".format(budget)
    so_problem.function = single_objective_nrp
    so_algorithm = GeneticAlgorithm(so_problem, population_size)
    so_algorithm.run(runs)
    x_so = [solution.objectives[0] for solution in so_algorithm.result]
    y_so = [solution.constraints[0] * (-1) for solution in so_algorithm.result]
    return x_so, y_so, so_algorithm


def rs_run(population_size):
    random_vars = run_random(runs, population_size)
    x_rs = [var[0] for var in random_vars]
    y_rs = [var[1] * (-1) for var in random_vars]
    return x_rs, y_rs


def single_objective_nrp(solution):
    score, cost = get_solution_vars(solution)
    fitness_function = weight * score + (1 - weight) * (cost)
    return fitness_function, cost


def multi_objective_nrp(solution):
    score, cost = get_solution_vars(solution)
    return [score, cost], [cost, cost]


def get_solution_vars(solution):
    solution = solution[0]
    total_cost = sum(req_costs)
    total_score = 0
    solution_score = 0
    solution_cost = 0
    for i, req in enumerate(solution):
        interested_customers = requirements[i]
        total_req_score = 0
        for cust in interested_customers:
            req_value = value(i + 1, cust)
            cust_weight = customers[cust - 1][0]
            single_req_score = req_value * cust_weight
            total_score += single_req_score
            total_req_score += single_req_score
        if req:
            req_cost = req_costs[i]
            solution_score += total_req_score
            solution_cost += req_cost
    if first_run: print_vars(total_score, total_cost)
    return solution_score/total_score, solution_cost/total_cost


def value(req_num, customer):
    cust_reqs = customers[customer - 1][1]
    reversed_reqs = cust_reqs[::-1]
    value = (reversed_reqs.index(req_num) + 1) / len(reversed_reqs)
    return value


def run_random(runs, population_size):
    random_vars = []
    for run in range(runs):
        score, cost = get_solution_vars(generate_random_solution())
        random_vars.append((score, cost))
    cut_random = random_vars[:population_size]
    return cut_random


def generate_random_solution():
    solution = [[random.randint(0, 1) for i in range(len(requirements))]]
    return solution


def print_vars(total_score, total_cost):
    print("Total Requirements Score:", total_score)
    print("Total Requirements Cost:", total_cost)
    global first_run
    first_run = False


if __name__ == "__main__":
    main()
