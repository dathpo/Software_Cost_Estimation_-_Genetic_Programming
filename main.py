__author__ = 'David T. Pocock'


import random
from pathlib import Path
from parser import Parser
import seaborn as sns; sns.set()
import os.path
import operator
import math
import numpy as np
from deap import algorithms, base, creator, tools, gp
from sklearn.model_selection import train_test_split


def main():
    global a

    pwd = os.path.abspath(os.path.dirname(__file__))
    data_dir_path = Path("data/")
    dataset = pwd / data_dir_path / "albrecht.arff"
    parser = Parser(dataset)
    df = parser.parse_data()
    x, y, pset = prep_albrecht(df)
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    x_train, x_test = split_data(df, 0.75)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset) # pset or not?

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) # genFull or HalfAndHalf?
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) # change to numpy array?
    toolbox.register("compile", gp.compile, pset=pset)
    fitness_function = get_fitness(toolbox)
    toolbox.register("evaluate", fitness_function, x_train)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17)) #maxval 5?
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    random.seed(1000)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 20, stats=mstats,
                                   halloffame=hof, verbose=True)

    print(str(hof.items[0]) + "     " + str(hof.items[0].fitness))
    print(fitness_function(x_test, hof.items[0]))

    return pop, log, hof


def split_data(data, split_percentage):
    x = int(len(data) * split_percentage)
    return data[0:x], data[x:]


def get_fitness(toolbox):
    def evaluate(data, individual):
        func = toolbox.compile(expr=individual)
        total_percentage_error = 0
        # print(data)
        for s in data:
            print(s[0], s[1])
            total_percentage_error += 100 * abs(func(s[1], s[2], s[3], s[4], s[5], s[6]) - s[-1]) / s[-1]
        return total_percentage_error / len(data),
    return evaluate


def prep_albrecht(df):
    y = df.Effort
    x = df.drop(columns=['Effort', 'AdjFP'])
    pset = gp.PrimitiveSet("main", 6)       # Num of inputs (cols)
    pset.addPrimitive(operator.add, 2)      # Num of arguments (a+b)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(np.square, 1)
    pset.addPrimitive(np.sqrt, 1)
    pset.addPrimitive(np.log, 1)
    pset.addPrimitive(np.exp, 1)
    pset.addPrimitive(operator.neg, 1)                                  # Negation
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1)) # No idea

    pset.renameArguments(ARG0='Input')
    pset.renameArguments(ARG1='Output')
    pset.renameArguments(ARG2='Inquiry')
    pset.renameArguments(ARG3='File')
    pset.renameArguments(ARG4='FPAdj')
    pset.renameArguments(ARG5='RawFPcounts')
    return x, y, pset


def prep_china():
    pset = gp.PrimitiveSet("main", 1)       # Num of inputs (cols)
    pset.addPrimitive(operator.add, 2)      # Num of arguments (a+b)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(np.square, 1)
    pset.addPrimitive(np.sqrt, 1)
    pset.addPrimitive(np.log, 1)
    pset.addPrimitive(np.exp, 1)
    pset.addPrimitive(operator.neg, 1)                                  # Negation
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1)) # No idea

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")


def prep_desharnais():
    pset = gp.PrimitiveSet("main", 1)       # Num of inputs (cols)
    pset.addPrimitive(operator.add, 2)      # Num of arguments (a+b)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(np.square, 1)
    pset.addPrimitive(np.sqrt, 1)
    pset.addPrimitive(np.log, 1)
    pset.addPrimitive(np.exp, 1)
    pset.addPrimitive(operator.neg, 1)                                  # Negation
    pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1)) # No idea

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


if __name__ == "__main__":
    main()
