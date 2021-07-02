import Reporter
import numpy as np
import random
import math
import scipy.spatial.distance as dist

class GenAlg:
    def __init__(self, filename):
        #self.reporter = Reporter.Reporter(self.__class__.__name__)
        self.reporter = Reporter.Reporter(filename)

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here

        # Initialize Parameters
        self.params = Parameters(
            popSize=100, #population size
            offSize=100, #offspring size
            selectK=7, #maximum k in k-tournament selection
            convergenceDelta=1e-2, # for convergence criterion
            convBestHor=100, # for convergence criterion
            convMeanHor=100, # for convergence criterion
        )

        # Initialize Population
        population = self.initialization(len(distanceMatrix))
        # Compute fitnesses (for selection)
        fitnesses = np.array(list(map(lambda x: self.fitness(distanceMatrix, x), population)))

        # Define values for convergence criterion
        it_nb = 0
        prevBestFit = min(fitnesses)
        prevMeanFit = fitnesses.mean()
        sameMeanCounter = 0
        sameBestCounter = 0
        bestMeanConverge = 0

        # Begin loop
        while self.convergenceTest(bestMeanConverge):

            # Initialize offspring
            offspring = np.empty(self.params.offspringSize + (self.params.offspringSize % 2), dtype=Individual)

            # Penalized Fitness values for Fitness Sharing
            penalized_fitnesses = self.fitnessSharing(population, distanceMatrix)

            # Fitness Sharing Selection + Recombination (PMX_G)
            for i in range(0, self.params.offspringSize, 2):
                sel1 = self.fitnessSharing_selection(population, penalized_fitnesses)
                sel2 = self.fitnessSharing_selection(population, penalized_fitnesses)
                offspring[i], offspring[i+1] = self.PMX(sel1, sel2)

            # Local Search (Limited 2-opt)
            offspring = np.array(list(map(lambda x: self.two_opt(ind=x, cost_mat=distanceMatrix), offspring)))

            # Mutation of path, k, LSC, sigma, alpha
            offspring = list(map(lambda x: self.superMutation(x), offspring))

            # Elimination
            population = self.elimination(distanceMatrix, population, offspring)

            # Fitness measurements
            fitnesses = np.array(list(map(lambda x: self.fitness(distanceMatrix, x), population)))
            bestObjective = min(fitnesses)
            meanObjective = fitnesses.mean()
            bestSolution = np.insert(population[fitnesses.argmin()].path, 0, 0, axis=0)

            if (bestObjective != np.inf and meanObjective != np.inf) and \
                    (abs(bestObjective - meanObjective) < self.params.convDelta):
                bestMeanConverge += 1

            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break
            it_nb += 1

            print("[Iter %03d] Best: %.6f, Mean: %.6f" % (it_nb, bestObjective, meanObjective))
        print(bestSolution)

        return 0

    def convergenceTest(self, bestMeanConverge):
        """
        Checks whether the population has converged.
        The population has converged when:
            The objective value of the best individual did not improve for the past self.convBestHor iterations, or
            the mean fitness improvement of the past self.convMeanHor iterations is not larger than a certain delta.
        :return: True if the EA has not converged yet, false otherwise.
        """
        cond0 = bestMeanConverge < 25
        if not cond0:
            print('Best and Mean Fitness have Converged for the past 25 iterations. Stopping.')
        return cond0

    def fitness(self, dist, ind):
        """
        Returns the fitness value of the given Individual.
        :param dist: distance matrix
        :param ind: the individual to evaluate
        """
        # From city 0 to city path[0]
        fitness = dist[0][ind.path[0]]
        # From city path[i-1] to city path[i]
        for i in range(1, len(ind.path)):
            fitness += dist[ind.path[i-1]][ind.path[i]]
        # From city path[-1] to city 0
        fitness += dist[ind.path[-1]][0]
        return fitness

    def initialization(self, nrCities):
        """
        Creates a population ndarray with [params.popSize] Individuals.
        Each Individual has a path of size [nrCities-1], where we omit the first city which is always city 0.
        """
        pop = list(np.empty(self.params.popSize, dtype=Individual))
        #pop = np.empty(self.params.popSize, dtype=Individual)

        for i in range(self.params.popSize):
            path = np.arange(start=1, stop=nrCities, step=1)
            np.random.shuffle(path)

            mutateChance = (random.random() * 0.05) + 0.035

            localSearchChance = 1/nrCities # the largest tour I expect is 929, so lower bound is approx 0.001
            k = random.randint(1, self.params.selectK)
            sigma = random.sample([0.5, 0.6, 0.7, 0.8, 0.9], 1)
            #alpha = random.randint(25, 50)
            alpha = np.round(random.randint(100, 250)/np.log(nrCities), 0)
            # make fitness sharing less severe for large tours
            pop[i] = Individual(path, mutateChance, localSearchChance, k=k, sigma=sigma, alpha=alpha)

        return pop

    def normalized_hamming_dist(self, x1, x2):
        """
        x1, x2 are equal-length paths
        returns normalized hamming distance
        """
        # source: https://stackoverflow.com/questions/32730202/fast-hamming-distance-computation-between-binary-numpy-arrays
        hd = np.count_nonzero(x1 != x2) / len(x1)
        return hd

    def hamming_dist_mat(self, population):
        """
        :param population: array of PATHS from the population
        :return: normalized hamming distance matrix
        """
        D = np.zeros([len(population), len(population)])
        # diagonal of D is already 0 as it should be
        # fill the upper and lower triangle with hamming distances
        for i in range(D.shape[0]):
            for j in range(i):
                D[i, j] = self.normalized_hamming_dist(population[i], population[j])
                D[j, i] = D[i, j]
        return D

    def fitnessSharing(self, population, dist_mat):
        """
        returns penalized fitness of individual given a population
        """
        # extract the sigma and alphas
        sigmas = np.array(list(map(lambda x: x.sigma, population)), dtype=Individual)
        alphas = np.array(list(map(lambda x: x.alpha, population)), dtype=Individual)

        #print(np.mean(alphas))

        # convert population to np array
        popn = np.array([x.path for x in population], dtype=Individual)

        # compute normalized hamming distance matrix
        # using scipy
        D = dist.pdist(popn, 'hamming')
        D = dist.squareform(D)

        # using user-defined function
        #D = self.hamming_dist_mat(popn)

        # compute penalized fitness values
        f = lambda x: self.fitness(dist_mat, x)
        fitness_vals = np.array(list(map(f, population)))

        penalties = np.zeros(len(population)) + 1
        for i in range(self.params.popSize):
            #print(D[i,])
            y = np.where(D[i,] < sigmas[i])
            #print(y)
            penalties[i] = np.sum(1 - pow((D[i, y] / sigmas[i]), alphas[i]))

        return fitness_vals*penalties

    def fitnessSharing_selection(self, population, penalized_fitnesses):
        K = math.ceil(np.mean(np.array(list(map(lambda x: x.k, population)))))
        selected = random.sample(range(self.params.popSize), K)
        best = min(selected, key=lambda x: penalized_fitnesses[x])
        return population[best]

    def selection(self, population, dist):
        """
        Selects an Individual through k-tournament selection.
        :param population: The population
        :param dist: distance matrix
        """
        selected = random.sample(population, self.params.selectK)
        best = min(selected, key=lambda x: self.fitness(dist, x))
        return best

    def superMutation(self, ind):
        """
        Performs all mutations at once
        """
        if random.random() > ind.mutateChance: return ind

        # Mutate the path
        indexes = [0, 0]
        while indexes[1] - indexes[0] <= 0:
            indexes[0] = random.randint(0, len(ind.path) - 1)
            indexes[1] = random.randint(0, len(ind.path) - 1)
        a = ind.path[indexes[0]]
        ind.path[indexes[0]:indexes[1]] = ind.path[indexes[0] + 1:indexes[1] + 1]
        ind.path[indexes[1]] = a

        # Mutate the K
        ind.k = 1

        # Mutate the Sigma
        #ind.sigma = [0.1]
        ind.sigma = ind.sigma

        # Mutate the Alpha
        #ind.alpha = 100
        ind.alpha = ind.alpha

        # Mutate the LSC
        ind.localSearchChance = min(ind.localSearchChance*10, 0.05)

        return ind


    def mutation(self, ind):
        """
        Mutates a given Individual with a certain chance.
        also mutates the k in k tournament selection
        :param ind: The Individual to mutate
        :return: The mutated Individual
        """
        if random.random() > ind.mutateChance: return ind
        indexes = [0, 0]
        while indexes[1] - indexes[0] <= 0:
            indexes[0] = random.randint(0, len(ind.path) - 1)
            indexes[1] = random.randint(0, len(ind.path) - 1)
        a = ind.path[indexes[0]]
        ind.path[indexes[0]:indexes[1]] = ind.path[indexes[0] + 1:indexes[1] + 1]
        ind.path[indexes[1]] = a
        return ind


    def mutateK(self, ind):
        """
        randomly make a solution have k=1 in K tournament selection
        """
        if random.random() > ind.mutateChance: return ind
        ind.k = 1
        return ind

    def mutateSigma(self, ind):
        """
        randomly make a solution have k=1 in K tournament selection
        """
        if random.random() > ind.mutateChance: return ind
        #ind.sigma = 0.1
        ind.sigma = ind.sigma
        return ind


    def mutateLSC(self, ind):
        """
        randomly make a solution have k=1 in K tournament selection
        """
        if random.random() > ind.mutateChance: return ind
        ind.localSearchChance = min(ind.localSearchChance*10, 0.05)
        return ind


    def SIM(self, ind):
        """
        Simple Inversion Mutation (Holland 1975; Grefenstette 1987)
        selects randomly two cut points in the string,
        and it reverses the substring between these two cut points
        """
        if random.random() > ind.mutateChance: return ind
        path = ind.path
        pathlength = path.shape[0]
        cut1 = random.randint(1, pathlength - 2)
        cut2 = random.randint(cut1, pathlength - 2)
        if (cut2 - cut1) < 2:
            cut2 += 1
        subtour = path[cut1:cut2][::-1]
        #print(subtour)
        ind.path = np.concatenate([path[0:cut1], subtour, path[cut2::]])
        return ind


    def PMX_helper(self, a, b, start, stop):
        """
        Helper function for PMX crossover
        :param ind1: individual 1
        :param ind2: individual 2
        :return: one child parents ind1 and ind2
        """
        child = [None] * len(a)
        # Copy a slice from first parent:
        child[start:stop] = a[start:stop]
        # Map the same slice in parent b to child using indices from parent a:
        for ind, x in enumerate(b[start:stop]):
            ind += start
            if x not in child:
                while child[ind] is not None:
                    ind = b.index(a[ind])
                child[ind] = x

        # Copy over the rest from parent b
        for ind, x in enumerate(child):
            if x == None:
                child[ind] = b[ind]
        return child

    def PMX(self, ind1, ind2):
        """
        Partially Mapped Crossover
        """
        a, b = ind1.path.tolist(), ind2.path.tolist()
        half = len(a) // 2
        start = random.randint(0, len(a) - half)
        stop = start + half
        path1 = np.array(self.PMX_helper(a, b, start, stop))
        path2 = np.array(self.PMX_helper(b, a, start, stop))
        return Individual(path1, ind1.mutateChance, ind1.localSearchChance, ind1.k, sigma=ind1.sigma, alpha=ind1.alpha), \
               Individual(path2, ind2.mutateChance, ind1.localSearchChance, ind2.k, sigma=ind2.sigma, alpha=ind2.alpha)

    def PMX_G(self, ind1, ind2):
        """
        Variation of PMX by Grefenstette (1987b) on paper pg. 139
        :param ind1: parent 1, example: (12|345|678)
        :param ind2: parent 2, example: (15|372|468)
        :return: child that looks like parent 2 but with a random subtour from parent 1: (1|345|7268)
                 child that looks like parent 1 but with a random subtour from parent 2: (1|372|5468)
        """
        path1 = np.array(ind1.path)
        path2 = np.array(ind2.path)
        pathlength = ind1.path.shape[0]
        cut1 = random.randint(1, pathlength - 2)
        cut2 = random.randint(cut1, pathlength - 2)
        if (cut2 - cut1) < 2:
            cut2 += 1

        # Child1: looks like parent 2 with subtour from parent 1
        subtour1 = path1[cut1:cut2]
        part1 = path2[0:cut1]

        # first part of child: not efficient but for now let's do it this way
        c1 = np.array([], dtype=np.int32)
        for i in range(len(part1)):
            if part1[i] not in subtour1:
                c1 = np.append(c1, part1[i])

        c1 = np.concatenate((c1, subtour1))
        # now add the rest of parent2 to c1
        part2 = path2[cut1::]
        for i in range(len(part2)):
            if part2[i] not in c1:
                c1 = np.concatenate((c1, [part2[i]]))

        # Child2: child that looks like parent 1 but with a random subtour from parent 2: (15|372|468)
        subtour2 = path2[cut1:cut2]
        part1 = path1[0:cut1]

        # first part of child: not efficient but for now let's do it this way
        c2 = np.array([], dtype=np.int32)
        for i in range(len(part1)):
            if part1[i] not in subtour2:
                c2 = np.append(c2, part1[i])

        c2 = np.concatenate((c2, subtour2))
        # now add the rest of parent1 to c2
        part2 = path1[cut1::]
        for i in range(len(part2)):
            if part2[i] not in c2:
                c2 = np.concatenate((c2, [part2[i]]))

        return Individual(c1, ind1.mutateChance, ind1.localSearchChance, ind1.k, sigma=ind1.sigma, alpha=ind1.alpha), \
               Individual(c2, ind2.mutateChance, ind1.localSearchChance, ind2.k, sigma=ind2.sigma, alpha=ind2.alpha)

    def cost_change(self, cost_mat, n1, n2, n3, n4):
        """
        original: n1 -> n2, n3 -> n4
        new: n1 -> n3, n2 -> n4
        """
        return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

    def two_opt(self, ind, cost_mat):
        """
        :param ind: individual who undergoes two-opt
        :param cost_mat: distance matrix
        :return:
        """
        if random.random() > ind.localSearchChance: return ind
        route = ind.path
        best = route
        improved = True
        ii = 0
        max_ii = np.log(len(route)) * 2500
        while improved and ii < max_ii: # limit number of iterations in local search
            improved = False
            for i in range(1, len(route) - 2):
                ii += 1
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue
                    if self.cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                        # compute the cost change from switching two edges
                        # if it's < 0 then we switch the edges and return it
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            route = best
            ind.path = best
        return ind

    def elimination(self, dist, pop, offspring):
        """
        Performs (λ + μ)-elimination
        :param dist: distance matrix
        :param pop: old population
        :param offspring: created offspring
        """
        # other implementation options include using numpy.sort() when pop is an ndarray and numpy's argsort
        # → https://stackoverflow.com/a/40984689
        newPop = []
        newPop.extend(pop)
        newPop.extend(offspring)
        newPop.sort(key=lambda x: self.fitness(dist, x), reverse=False)
        newPop = newPop[0:self.params.popSize]
        K = math.ceil(np.mean(np.array(list(map(lambda x: x.k, newPop)))))
        #print(K)
        return newPop


class Parameters:
    def __init__(self, popSize, offSize, selectK, convergenceDelta, convBestHor=4, convMeanHor=3):
        self.popSize = popSize
        self.offspringSize = offSize
        self.selectK = selectK
        self.convDelta = convergenceDelta
        self.convBestHor = convBestHor
        self.convMeanHor = convMeanHor

class Individual:
    def __init__(self, path, mutateChance, localSearchChance, k, alpha, sigma):
        self.path = path
        self.mutateChance = mutateChance
        self.localSearchChance = localSearchChance
        self.k = k
        self.alpha = alpha
        self.sigma = sigma
