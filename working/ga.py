#Consider a sample problem of optimizing the gain and bandwidth of a
#simple analog amplifier circuit using a genetic algorithm.
#The circuit consists of a single-stage common-emitter amplifier with a resistive load.

import numpy as np
import pygad
import matplotlib.pyplot as plt
import threading

from pathlib import Path
from os import path
import os
import shutil

import PyLTSpice as lt
from PyLTSpice.LTSteps import LTSpiceLogReader

import logging
import requests

from tabulate import tabulate

# standard R values
std_R = [1,1.1,1.2,1.3,1.5,1.6,1.8,2,2.2,2.4,2.7,3,3.3,3.6,3.9,4.3,4.7,5.1,5.6,6.2,6.8,7.5,8.2,9.1,
              10,11,12,13,15,16,18,20,22,24,27,30,33,36,39,43,47,51,56,62,68,75,82,91,
              100,110,120,130,150,160,180,200,220,240,270,300,330,360,390,430,470,510,560,620,680,750,820,910,
              1e3,1.1e3,1.2e3,1.3e3,1.5e3,1.6e3,1.8e3,2e3,2.2e3,2.4e3,2.7e3,3e3,3.3e3,3.6e3,3.9e3,4.3e3,4.7e3,5.1e3,5.6e3,6.2e3,6.8e3,7.5e3,8.2e3,9.1e3,
              10e3,11e3,12e3,13e3,15e3,16e3,18e3,20e3,22e3,24e3,27e3,30e3,33e3,36e3,39e3,43e3,47e3,51e3,56e3,62e3,68e3,75e3,82e3,91e3,
              100e3,110e3,120e3,130e3,150e3,160e3,180e3,200e3,220e3,240e3,270e3,300e3,330e3,360e3,390e3,430e3,470e3,510e3,560e3,620e3,680e3,750e3,820e3,910e3,
              1e6,1.1e6,1.2e6,1.3e6,1.5e6,1.6e6,1.8e6,2e6,2.2e6,2.4e6,2.7e6,3e6,3.3e6,3.6e6,3.9e6,4.3e6,4.7e6,5.1e6,5.6e6,6.2e6,6.8e6,7.5e6,8.2e6,9.1e6,
              10e6,11e6,12e6,13e6,15e6,16e6,18e6,20e6,22e6]
# standard L values
std_L = [1e-9,1.1e-9,1.2e-9,1.3e-9,1.5e-9,1.6e-9,1.8e-9,2e-9,2.2e-9,2.4e-9,2.7e-9,3e-9,3.3e-9,3.6e-9,3.9e-9,4.3e-9,4.7e-9,5.1e-9,5.6e-9,6.2e-9,6.8e-9,7.5e-9,8.2e-9,8.7e-9,9.1e-9,
              10e-9,11e-9,12e-9,13e-9,15e-9,16e-9,18e-9,20e-9,22e-9,24e-9,27e-9,30e-9,33e-9,36e-9,39e-9,43e-9,47e-9,51e-9,56e-9,62e-9,68e-9,75e-9,82e-9,87e-9,91e-9,
              100e-9,110e-9,120e-9,130e-9,150e-9,160e-9,180e-9,200e-9,220e-9,240e-9,270e-9,300e-9,330e-9,360e-9,390e-9,430e-9,470e-9,510e-9,560e-9,620e-9,680e-9,750e-9,820e-9,870e-9,910e-9,
              1e-6,1.1e-6,1.2e-6,1.3e-6,1.5e-6,1.6e-6,1.8e-6,2e-6,2.2e-6,2.4e-6,2.7e-6,3e-6,3.3e-6,3.6e-6,3.9e-6,4.3e-6,4.7e-6,5.1e-6,5.6e-6,6.2e-6,6.8e-6,7.5e-6,8.2e-6,8.7e-6,9.1e-6,
              10e-6,11e-6,12e-6,13e-6,15e-6,16e-6,18e-6,20e-6,22e-6,24e-6,27e-6,30e-6,33e-6,36e-6,39e-6,43e-6,47e-6,51e-6,56e-6,62e-6,68e-6,75e-6,82e-6,87e-6,91e-6,
              100e-6,110e-6,120e-6,130e-6,150e-6,160e-6,180e-6,200e-6,220e-6,240e-6,270e-6,300e-6,330e-6,360e-6,390e-6,430e-6,470e-6,510e-6,560e-6,620e-6,680e-6,750e-6,820e-6,870e-6,910e-6,
              1e-3,1.1e-3,1.2e-3,1.3e-3,1.5e-3,1.6e-3,1.8e-3,2e-3,2.2e-3,2.4e-3,2.7e-3,3e-3,3.3e-3,3.6e-3,3.9e-3,4.3e-3,4.7e-3,5.1e-3,5.6e-3,6.2e-3,6.8e-3,7.5e-3,8.2e-3,8.7e-3,9.1e-3]
# standard C values
std_C = [2.2e-12,3.3e-12,4.7e-12,6.8e-12,
              10e-12,12e-12,15e-12,18e-12,22e-12,27e-12,33e-12,39e-12,47e-12,56e-12,68e-12,82e-12,
              100e-12,120e-12,150e-12,180e-12,220e-12,270e-12,330e-12,390e-12,470e-12,560e-12,680e-12,820e-12,
              1e-9,1.2e-9,1.5e-9,1.8e-9,2.2e-9,2.7e-9,3.3e-9,3.9e-9,4.7e-9,5.6e-9,6.8e-9,8.2e-9,
              10e-9,12e-9,15e-9,18e-9,22e-9,27e-9,33e-9,39e-9,47e-9,56e-9,68e-9,82e-9,
              100e-9,120e-9,150e-9,180e-9,220e-9,270e-9,330e-9,390e-9,470e-9,560e-9,680e-9,820e-9,
              1e-6,1.2e-6,1.5e-6,1.8e-6,2.2e-6,2.7e-6,3.3e-6,3.9e-6,4.7e-6,5.6e-6,6.8e-6,8.2e-6,
              10e-6,15e-6,22e-6,33e-6,47e-6,68e-6,
              100e-6,150e-6,220e-6,330e-6,470e-6,680e-6,
              1e-3,1.5e-3,2.2e-3,3.3e-3,4.7e-3,6.8e-3,
              10e-3,15e-3,22e-3,33e-3,47e-3,68e-3]

logging.disable(logging.CRITICAL)

logger = logging.getLogger("PyLTSpice.LTSteps")
logger.setLevel(logging.CRITICAL+1)

#ascPath=Path("D:/school/DataDrivenControl/mini-project/test_run.asc")
#componentList=["R1","R2"]
#outputName="rms_vout"
#desired_output = 7

# Define Circuit Simulation and Fitness Evaluation
def simulate_circuit(solution):
    filename=str(threading.get_ident())

    sim=lt.SimRunner(output_folder='./temp',verbose=False)
    shutil.copyfile(ascPath.stem + '.net', filename + '.net')
    net=lt.SpiceEditor(filename + '.net')

    solution = list(solution)
    for i,r in enumerate(resistorList):
        net.set_component_value(r, solution.pop(0))
    for i,c in enumerate(capacitorList):
        net.set_component_value(c, solution.pop(0))
    for i,l in enumerate(inductorList):
        net.set_component_value(l, solution.pop(0))

    sim.run_now(net,run_filename=filename)
    while not os.path.exists("temp/" + filename + ".log"):
        pass
    data=LTSpiceLogReader("temp/" + filename + ".log")

    #cleanup
    sim.file_cleanup()
    os.remove(filename + '.net')

    return data[outputName][0]


def fitness_function(ga_instance, solution, solution_idx):
    output = simulate_circuit(solution)
    fitness = 1.0 / np.abs(output - desired_output)
    return fitness

def gene_space(num_R, num_C, num_L):
    # returns the gene space given the component types
    global std_R, std_C, std_L
    return [std_R]*num_R + [std_C]*num_C + [std_L]*num_L

def print_values(solution):
    solution = list(solution)
    R = []
    C = []
    L = []
    # flag - change based on given component names
    for i,r in enumerate(resistorList):
        R.append(f"{r} = {solution.pop(0)}")
    for i,c in enumerate(capacitorList):
        C.append(f"{c} = {solution.pop(0)}")
    for i,l in enumerate(inductorList):
        L.append(f"{l} = {solution.pop(0)}")
    temp = {}
    if R:
        temp['Resistors'] = R
    if C:
        temp['Capacitors'] = C
    if L:
        temp['Inductors'] = L
    print(tabulate(temp, 
                headers='keys', 
                tablefmt='fancy_grid'))

def ga_sim(ascPathRaw,resistorListArg,capacitorListArg,inductorListArg,outputNameArg,desired_outputArg,genNum,solNum):
    global ascPath
    global resistorList
    global capacitorList
    global inductorList
    global outputName
    global desired_output
    resistorList=resistorListArg
    capacitorList = capacitorListArg
    inductorList = inductorListArg

    outputName=outputNameArg
    desired_output=float(desired_outputArg)

    ascPath=Path(ascPathRaw)
    workDir=ascPath.parent.absolute()
    os.chdir(workDir)

    LTC = lt.SimRunner(output_folder='./temp')
    LTC.create_netlist(ascPath.name)
    netlist = lt.SpiceEditor(ascPath.stem + '.net')

    num_generations = int(genNum)
    num_parents_mating = 4

    sol_per_pop = int(solNum) #increase for higher accuracy
    num_genes = len(resistorList) + len(capacitorList) + len(inductorList)

    #constraints
    init_range_low = 0
    init_range_high = 22e6

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           gene_space = gene_space(len(resistorList),len(capacitorList),len(inductorList)),
                           mutation_percent_genes=mutation_percent_genes,parallel_processing=["thread",min(num_parents_mating,8)])

    ga_instance.run()
    LTC.file_cleanup()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    prediction = simulate_circuit(solution)
    # print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
    print_values(solution)
