__author__ = 'David T. Pocock'


import random
from pathlib import Path
from parser import Parser
import seaborn as sns; sns.set()
import os.path
import operator
import math
from sklearn.model_selection import KFold
import numpy as np
from deap import algorithms, base, creator, tools, gp

def main():
    pwd = os.path.abspath(os.path.dirname(__file__))
    data_dir_path = Path("data/")
    dataset = pwd / data_dir_path / "albrecht.arff"
    parser = Parser(dataset)
    dataframe = parser.parse_data()
    df, pset = prep_albrecht(dataframe)

    k_fold = KFold(10, True, 1)

    # for train_index, test_index in k_fold.split(df):
    #     train_data = df.iloc[train_index]
    #     test_data = df.iloc[test_index]
    #     print(train_data)
    #     print(test_data)
    #     print()

    k_fold_hofs = []
    for train_index, test_index in k_fold.split(df):
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin) # pset or not?

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) # genFull or HalfAndHalf?
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual) # change to numpy array?
        toolbox.register("compile", gp.compile, pset=pset)
        fitness_function = get_fitness(toolbox, 1)
        toolbox.register("evaluate", fitness_function, train_data)
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

        # aMuPlusLambda_data = np.zeros((no_generations, no_runs))
        # for j in range(no_runs):
        #     for i in range(no_generations):
        #         pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 100, 300, 0.25, 0.75, 1, stats=mstats, \
        #                                              halloffame=hof, verbose=True)
        #         eaSimple_data[i, j] = hof.items[0].fitness.values[0]
        #         aMuPlusLambda_data[i, j] = fit_func(test_data, hof.items[0])[0]
        #     print(j)

        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 100, 300, 0.5, 0.5, 100, stats=mstats,
                                       halloffame=hof, verbose=True)

        # pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 1, stats=mstats,
        #                                halloffame=hof, verbose=True)

        validation_fitness = fitness_function(test_data, hof.items[0])[0]

        print()
        print("Training Output:")
        print("Training Data Rows:", len(train_data))
        print("Fittest Individual (HoF):", hof.items[0])
        print("Fitness of Best Model (MMRE):", str(hof.items[0].fitness)[1:-2])
        print()
        print("Model Validation with Test Data:")
        print("Test Data Rows:", len(test_data))
        print("Test Data Fitness (MMRE):", validation_fitness)
        print()

        k_fold_hofs.append((hof.items[0], str(hof.items[0].fitness)[1:-2], validation_fitness))

        del creator.FitnessMin
        del creator.Individual

    validat_fits = []
    for i, hof in enumerate(k_fold_hofs):
        print()
        print("Fold #{}".format(i+1))
        print("Fittest Individual (HoF):", hof[0])
        print("HoF Fitness (MMRE):", hof[1])
        print("HoF Validation Fitness (MMRE):", hof[2])
        validat_fits.append(hof[2])

    print()
    print("Mean Validation Fitness among Folds (MMRE):", np.mean(validat_fits))
    print("Standard Deviation of Validation Fitness among Folds:", np.std(validat_fits))


def get_fitness(toolbox, dataset_num):
    def evaluate(df, individual):
        func = toolbox.compile(expr=individual)
        magnitude_of_relative_error = 0
        for row in df.itertuples():
            if dataset_num == 1:
                pred_effort = func(row[1], row[2], row[3], row[4], row[5], row[6])
                actual_effort = row[7]
            elif dataset_num == 2:
                pred_effort = func(row[1], row[2], row[3], row[4], row[5], row[6], row[7])
                actual_effort = row[8]
            elif dataset_num == 3:
                pred_effort = func(row[1], row[2], row[3], row[4], row[5], row[6], row[7])
                actual_effort = row[8]
            magnitude_of_relative_error += (abs(actual_effort - pred_effort) / actual_effort) * 100
        mean_mre = magnitude_of_relative_error / len(df)
        return mean_mre,
    return evaluate


def prep_albrecht(df):
    df = df.drop(columns='AdjFP')
    pset = gp.PrimitiveSet("main", 6)       # Num of inputs (cols)
    pset.addPrimitive(operator.add, 2)      # Num of arguments (a+b)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addEphemeralConstant("rand101", lambda: random.randint(0, 100))
    pset.renameArguments(ARG0='Input')
    pset.renameArguments(ARG1='Output')
    pset.renameArguments(ARG2='Inquiry')
    pset.renameArguments(ARG3='File')
    pset.renameArguments(ARG4='FPAdj')
    pset.renameArguments(ARG5='RawFPcounts')
    # pset.renameArguments(ARG6='AdjFP')

    return df, pset

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
