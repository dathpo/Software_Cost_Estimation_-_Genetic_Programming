__author__ = 'David T. Pocock'


import random
from pathlib import Path
from parser import Parser
import seaborn as sns; sns.set()
import os.path
import operator
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from deap import algorithms, base, creator, tools, gp
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None


def main():
    global pwd
    pwd = os.path.abspath(os.path.dirname(__file__))
    data_dir_path = Path("data/")
    dataset = pwd / data_dir_path / "desharnais.arff"
    parser = Parser(dataset)
    dataframe = parser.parse_data()
    df, pset = prep_desharnais(dataframe)

    exec_k_fold(df, pset, True)
    exec_k_fold(df, pset, False)
    # exec_train_test_split(df, pset, False)


def execute_ea(pset, train_data, test_data):
    global external_call
    pop_num = 300
    external_call = False
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    fitness_function = get_fitness(toolbox, 3)
    toolbox.register("evaluate", fitness_function, train_data)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    random.seed(1000)
    pop = toolbox.population(n=pop_num)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.5, 100, stats=mstats,
    #                                halloffame=hof, verbose=True)

    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 100, pop_num, 0.5, 0.5, 100, stats=mstats,
                                   halloffame=hof, verbose=True)

    # pop, log = algorithms.eaMuCommaLambda(pop, toolbox, 100, 300, 0.5, 0.5, 100, stats=mstats,
    #                                       halloffame=hof, verbose=True)

    validation_fitness = fitness_function(test_data, hof.items[0])[0]

    external_call = True
    pred_effort, actual_effort = fitness_function(test_data, hof.items[0])
    external_call = False

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

    return hof, validation_fitness, pred_effort, actual_effort


def exec_train_test_split(df, pset, is_ea):
    global j
    j = 100
    train_data, test_data = train_test_split(df, test_size=0.25, random_state=0)
    if is_ea:
        execute_ea(pset, train_data, test_data)
    else:
        train_info, test_info = execute_lr(train_data, test_data)
        pred_effort_train = np.expm1(train_info[0])
        actual_effort_train = np.expm1(train_info[1])
        pred_effort_test = np.expm1(test_info[0])
        actual_effort_test = np.expm1(test_info[1])
        plot_data(pred_effort_train, actual_effort_train, pred_effort_test, actual_effort_test)


def exec_k_fold(df, pset, is_ea):
    global j
    global gp_pred_effort
    global gp_actual_effort
    k_fold = KFold(10, True, 1)
    k_fold_hofs = []
    ea_pred_effort = []
    ea_actual_effort = []
    k_fold_lr_metrics = []
    lr_pred_effort_train_tbf = []
    lr_actual_effort_train_tbf = []
    lr_pred_effort_test_tbf = []
    lr_actual_effort_test_tbf = []
    test_x_list = []
    j = 0
    for train_index, test_index in k_fold.split(df):
        j += 1
        train_data = df.iloc[train_index]
        test_data = df.iloc[test_index]

        if is_ea:
            hof, validation_fitness, ea_pred_effort_sin, ea_actual_effort_sin = execute_ea(pset, train_data, test_data)
            k_fold_hofs.append((hof.items[0], str(hof.items[0].fitness)[1:-2], validation_fitness))
            ea_pred_effort.append(ea_pred_effort_sin)
            ea_actual_effort.append(ea_actual_effort_sin)
            del creator.FitnessMin
            del creator.Individual
        else:
            train_info, test_info = execute_lr(train_data, test_data)
            k_fold_lr_metrics.append((train_info[2], test_info[2]))
            lr_pred_effort_train_tbf.append(train_info[0])
            lr_actual_effort_train_tbf.append(train_info[1])
            lr_pred_effort_test_tbf.append(test_info[0])
            lr_actual_effort_test_tbf.append(test_info[1])
            test_x_list.append(test_info[3])

    if is_ea:
        ea_pred_effort = [item for sublist in ea_pred_effort for item in sublist]
        gp_pred_effort = ea_pred_effort
        ea_actual_effort = [item for sublist in ea_actual_effort for item in sublist]
        gp_actual_effort = ea_actual_effort
        validat_fits = []
        for i, hof in enumerate(k_fold_hofs):
            print()
            print("Genetic Programming Fold #{}".format(i+1))
            print("Fittest Individual (HoF):", hof[0])
            print("HoF Fitness (MMRE):", hof[1])
            print("HoF Validation Fitness (MMRE):", hof[2])
            validat_fits.append(hof[2])

        print("\n")
        print("Genetic Programming Metrics:")
        print("Mean Validation Fitness among Folds (MMRE):", np.mean(validat_fits))
        print("Standard Deviation of Validation Fitness among Folds:", np.std(validat_fits))
        print("PRED(25) of Validation Data:", pred(ea_pred_effort, ea_actual_effort, 25), "%")

    else:
        lr_pred_effort_train = []
        for l in lr_pred_effort_train_tbf:
            lr_pred_effort_train.append(list(l))
        lr_pred_effort_train = np.array([item for sublist in lr_pred_effort_train for item in sublist])
        lr_actual_effort_train = np.array([item for sublist in lr_actual_effort_train_tbf for item in sublist])
        lr_pred_effort_test = []
        for i in lr_pred_effort_test_tbf:
            lr_pred_effort_test.append(list(i))
        lr_pred_effort_test = np.array([item for sublist in lr_pred_effort_test for item in sublist])
        lr_actual_effort_test = np.array([item for sublist in lr_actual_effort_test_tbf for item in sublist])
        test_x = []
        for g in test_x_list:
            test_x.append(list(g))
        test_x = np.array([item for sublist in test_x for item in sublist])
        lr_validat_fits = []
        for i, el in enumerate(k_fold_lr_metrics):
            print()
            print("Linear Regression Fold #{}".format(i + 1))
            print("Model Fitness (MMRE):", el[0])
            print("Model Validation Fitness (MMRE):", el[1])
            lr_validat_fits.append(el[1])

        print("\n")
        print("Linear Regression Metrics:")
        print("Mean Validation Fitness among Folds (MMRE):", np.mean(lr_validat_fits))
        print("Standard Deviation of Validation Fitness among Folds:", np.std(lr_validat_fits))
        print("PRED(25) of Validation Data:", pred(lr_pred_effort_test, lr_actual_effort_test, 25), "%")

        # lr_pred_effort_train = np.expm1(lr_pred_effort_train)
        # lr_actual_effort_train = np.expm1(lr_actual_effort_train)
        # lr_pred_effort_test = np.expm1(lr_pred_effort_test)
        # lr_actual_effort_test = np.expm1(lr_actual_effort_test)

        plot_data(lr_pred_effort_train, test_x, lr_pred_effort_test, lr_actual_effort_test)


def pred(pred, actual, n):
    less_than_n = 0
    for i in range(len(pred)):
        distance_pc = (abs(actual[i] - pred[i]) / actual[i]) * 100
        if distance_pc <= n:
            less_than_n += 1
    value = (less_than_n / len(pred)) * 100
    return value


def execute_lr(train_data, test_data):
    concated = pd.concat([train_data, test_data])
    corr = concated.corr()
    corr = corr.sort_values([concated.columns[-1]], ascending=False)
    print()
    print("Linear Regression Correlation Coefficients:")
    print(corr[concated.columns[-1]])
    print()

    y_train = train_data[train_data.columns[-1]]
    x_train = train_data.drop(train_data.columns[len(train_data.columns)-1], axis=1)
    y_test = test_data[test_data.columns[-1]]
    x_test = test_data.drop(test_data.columns[len(test_data.columns)-1], axis=1)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    lr_model = LinearRegression()
    lr_model.fit(x_train, y_train)

    pred_effort_train = lr_model.predict(x_train)
    actual_effort_train = y_train.tolist()
    mmre_train = loop_thru_zip(pred_effort_train, actual_effort_train)

    pred_effort_test = lr_model.predict(x_test)
    actual_effort_test = y_test.tolist()
    mmre_test = loop_thru_zip(pred_effort_test, actual_effort_test)

    print()
    print("Training Output:")
    print("Training Data Rows:", len(train_data))
    print("Model Fitness (MMRE):", mmre_train)
    print()
    print("Model Validation with Test Data:")
    print("Test Data Rows:", len(test_data))
    print("Model Validation Fitness (MMRE):", mmre_test)
    print()

    train_info = pred_effort_train, actual_effort_train, mmre_train
    test_info = pred_effort_test, actual_effort_test, mmre_test, x_test
    return train_info, test_info


def plot_data(pred_effort_train, test_x, lr_pred_effort_test, lr_actual_effort_test):
    global j
    global pwd
    global gp_pred_effort
    global gp_actual_effort

    mean_log = np.mean(gp_actual_effort)
    mean_arr_log = np.zeros(len(gp_actual_effort))
    mean_arr_log.fill(mean_log)
    median_log = np.median(gp_actual_effort)
    median_arr_log = np.zeros(len(gp_actual_effort))
    median_arr_log.fill(median_log)

    # Plot predictions - Compare GP and LR - K folds
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(gp_pred_effort, gp_actual_effort, c="red", marker="D", edgecolors='black', label="Genetic Programming")
    plt.scatter(lr_pred_effort_test, lr_actual_effort_test, c="lightgreen", marker="D", edgecolors='black', label="Linear Regression")
    plt.scatter(mean_arr_log, gp_actual_effort, 4, marker='D', c="blue", label="Mean Effort")
    # plt.scatter(median_arr, gp_actual_effort, 4, marker='D', c="#ff00ff", label="Median Effort")
    plt.plot([0, 4], [0, 4], c="black", label="Ideal Model")
    plt.title("Effort Predictions - Validation Data - China")
    plt.xlabel("Predicted Effort (log(1 + x))")
    plt.ylabel("Actual Effort (log(1 + x))")
    # plt.xlim((0.8, 4))
    # plt.ylim((0, 4))
    plt.legend(loc="upper left")
    # plt.plot(test_x, lr_pred_effort_test, c="#ffa500", label="Ideal Model")
    # plt.plot([mean, mean],[0,100], c='blue', label="Mean Effort")
    # plt.plot([median, median], [0, 100], c='#ff00ff', label="Median Effort")
    graph_path = os.path.join(pwd, '{}'.format(str(j) + 'newpred_comparison.pdf'))
    fig.savefig(graph_path, bbox_inches="tight")

    gp_pred_effort = np.expm1(gp_pred_effort)
    gp_actual_effort = np.expm1(gp_actual_effort)
    lr_pred_effort_test = np.expm1(lr_pred_effort_test)
    lr_actual_effort_test = np.expm1(lr_actual_effort_test)

    mean = np.mean(gp_actual_effort)
    mean_arr = np.zeros(len(gp_actual_effort))
    mean_arr.fill(mean)
    median = np.median(gp_actual_effort)
    median_arr = np.zeros(len(gp_actual_effort))
    median_arr.fill(median)

    z_gp = []
    for gp_pred, gp_actual in zip(gp_pred_effort, gp_actual_effort):
        z_gp_sin = gp_pred / gp_actual
        z_gp.append(z_gp_sin)
    z_lr = []
    for lr_pred, lr_actual in zip(lr_pred_effort_test, lr_actual_effort_test):
        z_lr_sin = lr_pred / lr_actual
        z_lr.append(z_lr_sin)
    z_m = []
    for m_pred, m_actual in zip(mean_arr, gp_actual_effort):
        z_m_sin = m_pred / m_actual
        z_m.append(z_m_sin)
    z_md = []
    for md_pred, md_actual in zip(median_arr, gp_actual_effort):
        z_md_sin = md_pred / md_actual
        z_md.append(z_md_sin)

    # Plot Z Distribution - GP, LR, Mean, Median
    fig = plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    box_data = [z_gp, z_lr, z_m]
    df = pd.DataFrame(data=box_data)
    df = df.transpose()
    df.columns = ['Genetic Programming', 'Linear Regression', 'Mean Effort']
    bplot = sns.boxplot(data=df)
    bplot.axes.set_title("z Values Distribution - Comparison - China")
    bplot.set_xlabel("Search Technique")
    bplot.set_ylabel("z value")
    # plt.ylim((-0.5, 8))
    graph_path = os.path.join(pwd, '{}'.format(str(j) + 'boxplot.pdf'))
    fig.savefig(graph_path, bbox_inches="tight")

    # # Train Test Split Plot predictions - Training vs Validation Data
    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(pred_effort_train, actual_effort_train, c="blue", marker="s", edgecolors='black', label="Training Data")
    # plt.scatter(lr_pred_effort_test, lr_actual_effort_test, c="lightgreen", marker="D", edgecolors='black',
    #             label="Validation Data")
    # plt.title("Linear Regression - Predictions")
    # plt.xlabel("Predicted Effort")
    # plt.ylabel("Actual Effort")
    # plt.legend(loc="upper left")
    # plt.plot([0, 98], [0, 100], c="red")
    # graph_path = os.path.join(pwd, '{}'.format(str(j) + 'td_vd_predict.pdf'))
    # fig.savefig(graph_path, bbox_inches="tight")

    # # Plot residuals
    # fig = plt.figure(figsize=(8, 6))
    # plt.scatter(pred_effort_train, pred_effort_train - actual_effort_train, c="blue", marker="s", edgecolors='black', label="Training Data")
    # plt.scatter(pred_effort_test, pred_effort_test - actual_effort_test, c="lightgreen", marker="s", edgecolors='black', label="Validation Data")
    # plt.title("Linear Regression - Residuals")
    # plt.xlabel("Predicted Effort")
    # plt.ylabel("Residuals")
    # plt.legend(loc="upper left")
    # plt.hlines(y=0, xmin=10.5, xmax=13.5, color="red")
    # graph_path = os.path.join(pwd, '{}'.format(str(j) + 'resid.pdf'))
    # fig.savefig(graph_path, bbox_inches="tight")


def loop_thru_zip(x, y):
    mres = []
    for pred_effort, actual_effort in zip(x, y):
        mre = (abs(actual_effort - pred_effort) / actual_effort) * 100
        mres.append(mre)
    mmre = sum(mres) / len(x)
    return mmre


def get_fitness(toolbox, dataset_num):
    def evaluate(df, individual):
        func = toolbox.compile(expr=individual)
        magnitude_of_relative_error_list = []
        pred_effort_list = []
        actual_effort_list = []
        for row in df.itertuples():
            if dataset_num == 1:
                pred_effort = func(row[1], row[2], row[3], row[4], row[5], row[6])
                actual_effort = row[7]
            elif dataset_num == 2:
                pred_effort = func(row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15])
                actual_effort = row[16]
            elif dataset_num == 3:
                pred_effort = func(row[1], row[2], row[3], row[4], row[5], row[6], row[7])
                actual_effort = row[8]
            magnitude_of_relative_error = (abs(actual_effort - pred_effort) / actual_effort) * 100
            magnitude_of_relative_error_list.append(magnitude_of_relative_error)
            pred_effort_list.append(pred_effort)
            actual_effort_list.append(actual_effort)
        global external_call
        if external_call:
            return pred_effort_list, actual_effort_list
        mean_mre = sum(magnitude_of_relative_error_list) / len(df)
        return mean_mre,
    return evaluate


def prep_albrecht(df):
    df = df.drop(columns='AdjFP')
    df.Effort = np.log1p(df.Effort)
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
    return df, pset


def prep_china(df):
    df = df.drop(columns='ID')
    df = df.drop(columns='Dev.Type')
    df = df.drop(columns='N_effort')
    df.Effort = np.log1p(df.Effort)
    pset = gp.PrimitiveSet("main", 15)       # Num of inputs (cols)
    pset.addPrimitive(operator.add, 2)      # Num of arguments (a+b)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)                                  # Negation
    pset.addEphemeralConstant("rand101", lambda: random.randint(0, 100))
    pset.renameArguments(ARG0="AFP")
    pset.renameArguments(ARG1="Input")
    pset.renameArguments(ARG2="Output")
    pset.renameArguments(ARG3="Enquiry")
    pset.renameArguments(ARG4="File")
    pset.renameArguments(ARG5="Interface")
    pset.renameArguments(ARG6="Added")
    pset.renameArguments(ARG7="Changed")
    pset.renameArguments(ARG8="Deleted")
    pset.renameArguments(ARG9="PDR_AFP")
    pset.renameArguments(ARG10="PDR_UFP")
    pset.renameArguments(ARG11="NPDR_AFP")
    pset.renameArguments(ARG12="NPDU_UFP")
    pset.renameArguments(ARG13="Resource")
    pset.renameArguments(ARG14="Duration")
    return df, pset


def prep_desharnais(df):
    df = df.drop(columns='Project')
    df = df.drop(columns='Adjustment')
    df = df.drop(columns='PointsAjust')
    df = df.drop(columns='Language')
    effort_col = df['Effort']
    df = df.drop(labels=['Effort'], axis=1)
    df.insert(len(df.columns), 'Effort', effort_col)
    df.Effort = np.log1p(df.Effort)
    pset = gp.PrimitiveSet("main", 7)       # Num of inputs (cols)
    pset.addPrimitive(operator.add, 2)      # Num of arguments (a+b)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)                                  # Negation
    pset.addEphemeralConstant("rand101", lambda: random.randint(0, 100))
    pset.renameArguments(ARG0="TeamExp")
    pset.renameArguments(ARG1="ManagerExp")
    pset.renameArguments(ARG2="YearEnd")
    pset.renameArguments(ARG3="Length")
    pset.renameArguments(ARG4="Transactions")
    pset.renameArguments(ARG5="Entities")
    pset.renameArguments(ARG6="PointsNonAdjust")
    return df, pset


def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


if __name__ == "__main__":
    main()
