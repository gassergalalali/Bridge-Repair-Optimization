import csv
import logging
import math
import random

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

logging.getLogger().setLevel(logging.DEBUG)

random.seed(10)


class Model:
    def __init__(self):
        logging.info("Initiating the model.")
        # initiate the lists that will hold the data
        self.cond = []
        self.width = []
        self.length = []
        self.ADT = []
        # Temporary lists for reading the data
        cond_superstructure = []
        cond_substructure = []
        cond_deck = []
        logging.info("Reading the csv.")
        csv_filename = "MO17.csv"
        with open(csv_filename) as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header
            for row in csv_reader:
                self.ADT.append(int(row[1]))
                self.length.append(float(row[2]))
                self.width.append(float(row[3]))
                cond_deck.append(row[4])
                cond_superstructure.append(row[5])
                cond_substructure.append(row[6])
        self.numberOfBridges = len(self.ADT)
        # Make sure that all the lengths are equal
        if not all(self.numberOfBridges == len(x) for x in (
                self.ADT,
                self.length,
                self.width,
                cond_deck,
                cond_superstructure,
                cond_substructure
        )):
            raise Exception("The number of variables is not the same in each column")
        # Calculate the maximum ADT
        self.maxADT = max(self.ADT)
        # Calculate the conditions
        self.cond = []  # Reset the conditions
        for deck, sup, sub in zip(
                cond_deck,
                cond_superstructure,
                cond_substructure
        ):
            if any("N" in x for x in (deck, sup, sub)):
                self.cond.append("N")
            else:
                self.cond.append(int(round(int(deck) + int(sup) + int(sub)) / 3))
        logging.info("Number of bridges is {}".format(self.numberOfBridges))
        logging.info("Average cond is {}".format(
            np.average(
                [x for x in self.cond if x != "N"]
            )
        ))
        logging.info("Max ADT is {}".format(self.maxADT))

        # initialize parameters for the GA
        self.ga_population = []
        self.ga_scores = []
        self.ga_costs = []
        self.ga_population_size = 200
        self.ga_mutation_rate = 0.20
        self.ga_max_epochs = 1000

        # used for plotting later
        self.ga_plot_epochs = []
        self.ga_plot_scores = []

        # possible repair types for each condition
        self.repair_constraints = {
            "N": [0, ],
            0: [3, ],
            1: [3, ],
            2: [3, ],
            3: [2, 3],
            4: [2, 3],
            5: [1, 2],
            6: [1, 2],
            7: [1],
            8: [1],
            9: [0],
        }
        # cost per sqm of reach repair type
        self.repair_cost = {
            "N": 0,
            0: 0,
            1: 1,
            2: 2,
            3: 6,
        }
        # maximum repair cost of a plan
        # self.maximum_repair_cost = 20000000
        self.maximum_repair_cost = 13500000

    def evaluate_repair(self, repair: [int]):
        """
        Evaluate a repair plan represented as a list of ints
        :type repair: list[int]
        :return: int or false if repair is invalid
        """
        # Make sure that each repair is within the minimum and
        # maximum acceptable repair
        for r, c in zip(repair, self.cond):
            if r not in self.repair_constraints[c]:
                logging.debug("Repair does not meet repair type constraints")
                return False
        # Check the the repair meets the cost constraint
        total_cost = 0
        for r, l, w in zip(repair, self.length, self.width):
            total_cost += self.repair_cost[r] * l * w
        if total_cost > self.maximum_repair_cost:
            logging.debug("Repair does not meet budget constraint")
            return False
        # Calculate the condition and score for each bridge
        total_score = 0
        for i, r in enumerate(repair):
            if self.cond[i] == "N":
                new_condition = "N"
            elif r == 0:
                new_condition = self.cond[i]
            elif r == 1:
                new_condition = self.cond[i] + 1
            elif r == 2:
                new_condition = self.cond[i] + 2
            elif r == 3:
                new_condition = 9
            else:
                raise Exception("Unknown repair type {}".format(r))
            # check if the new condition is at least five
            if new_condition is not "N":
                if new_condition < 5:
                    logging.debug("Condition is less than 5")
                    return False
            # Calculate the new score
            if new_condition is not "N":
                score = (
                        new_condition
                        / (1 + math.exp(-5 * self.ADT[i] / self.maxADT))
                )
                total_score += score
        return total_score, total_cost

    def ga_run(self):
        logging.info("Running GA")
        # check that the minimum budget can ever be met
        logging.info("Initializing population...")
        while len(self.ga_population) < self.ga_population_size:
            self.ga_add_generated_repair()
            logging.debug("{}/{}".format(len(self.ga_population), self.ga_population_size))
        plt.ion()
        for epoch in range(self.ga_max_epochs + 1):
            logging.info("Epoch {}: max score = {} with budget {}".format(
                epoch,
                max(self.ga_scores),
                self.ga_costs[self.ga_scores.index(max(self.ga_scores))]
            ))
            logging.debug("killing the weakest...")
            while len(self.ga_population) > int(self.ga_population_size * 0.75):
                self.ga_kill_one_weak()
            if random.random() < self.ga_mutation_rate:
                logging.debug("Mutation!")
                while len(self.ga_population) < self.ga_population_size:
                    self.ga_add_generated_repair()
            else:
                logging.debug("Creating Children")
                while len(self.ga_population) < self.ga_population_size:
                    self.ga_add_one_cross_over()
            for s in self.ga_scores:
                self.ga_plot_epochs.append(epoch)
                self.ga_plot_scores.append(s)

            plt.scatter(self.ga_plot_epochs, self.ga_plot_scores, marker=".", c="black")
            plt.title("GA Optimization")
            plt.xlabel("Epoch #")
            plt.ylabel("Score")
            plt.pause(0.001)

    def ga_add_generated_repair(self):
        """Generate a new repair"""
        while True:
            new_repair = [random.choice(self.repair_constraints[c]) for c in self.cond]
            ev = self.evaluate_repair(new_repair)
            if ev is not False:
                self.ga_population.append(new_repair)
                self.ga_scores.append(ev[0])
                self.ga_costs.append(ev[1])
                return 0

    def ga_kill_one_weak(self):
        """kills one weak"""
        index = self.ga_scores.index(min(self.ga_scores))
        self.ga_scores.pop(index)
        self.ga_population.pop(index)

    def ga_add_one_cross_over(self):
        while True:
            parent_one_index = random.randint(0, len(self.ga_population) - 1)
            parent_two_index = random.randint(0, len(self.ga_population) - 1)
            if parent_one_index == parent_two_index:
                continue
            child_repair = []
            for i in range(self.numberOfBridges):
                child_repair.append(random.choice([
                    self.ga_population[parent_one_index][i],
                    self.ga_population[parent_two_index][i]
                ]))
            ev = self.evaluate_repair(child_repair)
            if ev is not False:
                self.ga_population.append(child_repair)
                self.ga_scores.append(ev[0])
                self.ga_costs.append(ev[1])
                return 0
            else:
                logging.debug("child is invalid")


if __name__ == "__main__":
    model = Model()
    model.ga_run()
    plt.savefig("plot.png")
    input("Done! Press anything to continue...")
