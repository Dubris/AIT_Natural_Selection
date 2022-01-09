# -*- coding: utf-8 -*-
"""
@Author: Alexandre Le Bris
@Date: 01/09/22
"""

# Import packages
import numpy as np
import pandas as pd
from itertools import combinations
from tqdm import tqdm
from joblib import Parallel, delayed
from pybdm import BDM


def create_mutations_generator(n_size, n_shift):
    """Create all the possible mutations of a n-size binary matrix
    (0: the value remains unchanged)
    (1: the value shifts)

    Args:
        n_size (int): size of the input matrix
        n_shift (int): number of mutations/shifts in the input matrix

    Returns:
        list of arrays: list of all the possible mutations 
    """
    mutations = []
    i = 0

    for shift in range(n_shift):
        shift += 1
        mutation_possibles = combinations(range(n_size ** 2), shift)
        for indexes in mutation_possibles:
            for index in indexes:
                mutations.append([index // n_size, index % n_size, i])
            i += 1

    mutations = tuple(zip(*mutations))
    mutation_generators = np.zeros((n_size, n_size, i))
    mutation_generators[mutations] = 1
    return mutation_generators


def compute_proba(mutations, method='bdm'):
    """Compute the probability distributions for a set of mutated matrices.

    Args:
        mutations (list of arrays): list of matrices
        method (str, optional): methods available: 'entropy' or 'bdm'.
        Defaults to 'bdm'

    Returns:
        array: probability distribution
    """
    n_mutations = mutations.shape[2]
    complexity_list = []
    if method == 'bdm':
        bdm = BDM(ndim=2).bdm
    elif method == 'entropy':
        bdm = BDM(ndim=2).ent

    for i in range(n_mutations):
        complexity_list.append(bdm(mutations[..., i]))

    complexity_list = np.array(complexity_list)
    proba = 2 ** (-complexity_list)
    return proba / proba.sum()


def compute_proba_uni(mutations):
    """Compute the uniform distributions for a set of matrices.

    Args:
        mutations (list of arrays): input matrices

    Returns:
        array: uniform distribution
    """
    n_mutations = mutations.shape[2]
    proba = np.ones(n_mutations)
    return proba / proba.sum()


class Evolution():
    def __init__(self, n_size: int, n_shift: int, alpha: int = 0,
                 t_ext: int = 2500, proba_method: str = 'bdm',
                 replace: bool = False):
        """Constructor of the object Evolution, with definition of the main
        parameters to use the related methods.

        Args:
            n_size (int): size of the matrix (ideally a mulitple of 3 or 4)
            n_shift (int): number of shifts 
            alpha (int, optional): convergence parameter. Defaults to 0
            t_ext (int, optional): extinction threshold. Defaults to 2500
            proba_method (str, optional): method for computing probabilities.
            Defaults to 'bdm'
            replace (bool, optional): whether a pick-uped mutated matrice is 
            replaced in the pool of mutants or not. Defaults to False.
        """
        self.n_size = n_size
        self.n_shift = n_shift
        self.alpha = alpha
        self.t_ext = t_ext
        self.replace = replace

        self.proba_method = proba_method
        if 'bdm' in proba_method:
            self.compute_proba = compute_proba
        elif 'entropy' in proba_method:
            self.compute_proba = lambda x: compute_proba(x, 'entropy')
        elif 'uniform' in proba_method:
            self.compute_proba = compute_proba_uni

        self.mutations_gen = create_mutations_generator(self.n_size,
                                                        self.n_shift)

    def evolve_ind(self, initial, target, compute_proba, verbose=True):
        """Make a single initial matrix evolve to the given target matrix.

        Args:
            initial (array): initial matrix
            target (array): target matrix
            compute_proba (function): function computing the probability
            distribution for a mutation to be selected
            verbose (bool, optional): further comments or not. Defaults to True.

        Returns:
            (int, list of arrays, list of int): extinction reached (1) or not (0),
            list of all the selected matrices that improves the distance
            to the target matrix (mat_evo), list of the related step numbers for
            these matrices (steps_evo)
        """
        # Parameters initialization
        n_mutations = self.mutations_gen.shape[-1]  # Number of mutations
        n_step = 0          # Total number of steps (time)
        n_step_wo = 0       # Number of steps without mutation
        mat_evo = []        # History of evolution of the selected matrix
        steps_evo = []      # History of steps of the selected matrix
        dist_ref = (target != initial).sum()

        if verbose:
            print(f"Initial distance : {dist_ref}")

        # Start mutations
        while True:
            # Generation of the mutants from the selected matrix
            initial = np.repeat(initial[..., np.newaxis], n_mutations, axis=-1)
            mutations = (initial + self.mutations_gen) % 2
            mutations = mutations.astype(int)

            # Computation of the probabilities
            proba = compute_proba(mutations)

            # Determination of the sequence of indexes for the pick-uped
            # mutated matrices (according to the computed probabilities)
            if self.replace:
                size_indexes = self.t_ext
            else:
                size_indexes = min(self.t_ext, n_mutations)

            indexes = np.random.choice(range(n_mutations), size=size_indexes,
                                       p=proba, replace=self.replace)

            # Comparison between the pick-uped mutants and the target
            for index in indexes:
                n_step += 1
                n_step_wo += 1
                initial = mutations[..., index]
                dist = (target != initial).sum()

                # A mutant decreases the distance to the target
                if dist < dist_ref:
                    mat_evo.append(initial)
                    steps_evo.append(n_step)
                    n_step_wo = 0
                    dist_ref = dist
                    if verbose:
                        print(
                            f"\t Mutation at step {n_step} (distance = {dist}) !")
                    break

            # Extinction threshold is reached
            if n_step_wo == self.t_ext:
                steps_evo.append(n_step)
                if verbose:
                    print(f"Extinction at step {n_step}.\n")
                return (1, mat_evo, steps_evo)

            # Target reached
            if dist_ref <= self.alpha:
                if verbose:
                    print(f"Target reached at step {n_step}.\n")
                return (0, mat_evo, steps_evo)

    def evolve(self, initials, targets, verbose=False, parallel=True, n_jobs=-1,
               force_uni=False):
        """Make several initial matrices evolve to their respective target. 

        Args:
            initials (list of arrays): initial matrices
            targets (list of arrays): target matrices (one per initial matrix)
            verbose (bool, optional): comments or not. Defaults to False.
            parallel (bool, optional): allows parallel processing. Defaults to True.
            n_jobs (int, optional): number of jobs for parallel processing.
            Defaults to -1 (all the processors are used).
            force_uni (bool, optional): force to uniform distribution.
            Defaults to False.

        Returns:
            list of (int, list of arrays, list of int):
            extinction reached (1) or not (0),
            list of all the selected matrices that improves the distance
            to the target matrix (mat_evo),
            list of the related step numbers for these matrices (steps_evo).
        """
        # Checking step
        assert initials.shape == targets.shape

        # Parameters initialization
        mat_evo_all = []        # History of evolution of the matrices
        steps_evo_all = []      # History of evolution regarding steps
        extinction_all = []    # Extinction reached or not
        n_matrices = len(initials)

        if force_uni:
            compute_fun = compute_proba_uni
        else:
            compute_fun = self.compute_proba

        # Iteration over the initial matrices (sequential version)
        if not parallel:
            for i, m in enumerate(self.initial_mat):
                if verbose:
                    print(f"Processing evolution for matrice {i+1}")
                results = self.evolve_ind(m, self.target, compute_fun,
                                          verbose)
                extinction, mat_evo, steps_evo = results
                extinction_all.append(extinction)
                mat_evo_all.append(mat_evo)
                steps_evo_all.append(steps_evo)

        # Iteration over the initial matrices (parallel version)
        else:
            results = Parallel(n_jobs)(delayed(self.evolve_ind)(initials[i],
                                                                targets[i],
                                                                compute_fun,
                                                                verbose)
                                       for i in tqdm(range(n_matrices)))
            results = list(zip(*results))
            extinction_all = np.array(results[0])
            mat_evo_all = list(results[1])
            steps_evo_all = list(results[2])

        return extinction_all, mat_evo_all, steps_evo_all

    def generate_df(self, initials, targets):
        """Generate a Dataframe with the main results of the evolution of initial
        matrices to their target :
        - 1st column: complexity of the target matrix,
        - 2nd column: method used to compute the probability distribution,
        - 3rd column: number of steps once reached the target matrix or the
        extinction threshold
        - 4th column: extinction threshold reached or not
        - 5th column: speed-up quotient 

        Args:
            initials (list of arrays): initial matrices
            targets (list of arrays): target matrices

        Returns:
            Dataframe: results
        """
        # Initialization
        n_matrices = len(initials)
        bdm = BDM(ndim=2)

        # Computing results
        print(f"Step 1 - Computing results for method '{self.proba_method}'")
        ext_ref_list, _, steps_ref_list = self.evolve(initials, targets)

        print("Step 2 - Computing results for method 'uniform'")
        ext_uni_list, _, steps_uni_list = self.evolve(initials, targets,
                                                      force_uni=True)

        # Generation of the Dataframe
        print("Step 3 - Generating DataFrame...")
        complexity = [bdm.bdm(targets[i]) for i in range(n_matrices)]
        data = [complexity + complexity]

        method = [self.proba_method for _ in range(n_matrices)]
        method += ['uniform' for _ in range(n_matrices)]
        data.append(method)

        steps_ref = [item[-1] for item in steps_ref_list]
        steps_uni = [item[-1] for item in steps_uni_list]
        data.append(steps_ref + steps_uni)
        data.append(list(ext_ref_list) + list(ext_uni_list))

        # Computing the speed-up quotient
        speedup = np.array(steps_uni) / np.array(steps_ref)
        mask = (ext_uni_list == 0) * (ext_ref_list == 0)
        speedup = speedup * mask
        data.append(list(speedup) + [1] * n_matrices)

        data = list(map(list, zip(*data)))

        print("Done !")
        return pd.DataFrame(data=data, columns=['complexity', 'method',
                                                'steps', 'extinction', 'speedup'])

    def compute_speedup(self, initials, targets, ext_list, steps):
        """Compute speed-up quotients

        Args:
            initials (list of arrays): initial matrices
            targets (list of arrays): related target matrices
            ext_list (list of int: extinction for each couple initial/target matrices
            steps (list of int): number of required steps for each couple
            initial/target matrices

        Returns:
            array: speed-up quotient
        """
        # Computation of results for uniform probabilities
        ext_list_uni, _, steps_list = self.evolve(
            initials, targets, force_uni=True)
        steps_uni = np.array([item[-1] for item in steps_list])

        # Computation of speed-up quotient (0 if extinction for either method)
        speedup = steps_uni / steps
        mask = (ext_list_uni == 0) * (ext_list == 0)
        return mask * speedup
