#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 15:20:28 2020

@author: berube

This code's documentation is available in
https://github.com/nicolasberube/Utree
"""
import sys
from time import time
import numpy as np
from scipy.stats import rankdata, mannwhitneyu


class node():
    """Node/leaf of a decision tree for the u-tree class

    Parameters
    ----------
    leaf_id: int
        Unique ID of the leaf

    Attributes
    ----------
    id: int
        Unique ID of the leaf
    is_leaf: bool
        Flag indicating if the node is a final node, a leaf.
    label: Key of the splitting database
        Key of split_data, where the best cutoff will be computed on
        split_data[split_label]
    cutoff: float
        Cutoff for the selected variable label.
        If the leaf is a final node, the it is the value of the ideal cutoff
        that was ultimately not made
    value: float
        efficiency value of the cutoff, where the best cutoff has the higher
        value. The value is based on the u-test, the effect strength
        and the number of data in the sub-groups.
        If the leaf is a final node, the it is the value of the ideal cutoff
        that was ultimately not made
    id_lower: int
        Unique ID of the leaf to go to for split_data[label] < cutoff
        None if is_leaf == True.
    id_higher: int
        Unique ID of the leaf to go to for split_data[label] > cutoff
        None if is_leaf == True.
    id_null: int
        Unique ID of the leaf to go to for split_data[label] is null/nan
        None if is_leaf == True.
    effect: float
        The difference in means of all data on that leaf between
        bin=True and bin=False.
        None if is_leaf == False.
    n_data_with: int
        Number of data point of bin=True
        None if is_leaf == False.
    n_data_without: int
        Number of data point of bin=True
        None if is_leaf == False.
    """

    def __init__(self,
                 leaf_id):
        self.id = leaf_id
        self.is_leaf = False
        self.label = None
        self.cutoff = None
        self.id_lower = None
        self.id_higher = None
        self.id_null = None
        self.value = None
        self.n_data_with = None
        self.n_data_without = None
        self.effect = None

    def __repr__(self):
        if self.is_leaf:
            return (f'id:{self.id} n_data:{self.n_data_with}/'
                    f'{self.n_data_without} effect:{self.effect}')
        else:
            return (f'id:{self.id} cutoff({self.label}): {self.cutoff}'
                    f' ids:{self.id_lower}/{self.id_higher}/{self.id_null}')


class quicksum_array():
    """Array that can quickly sum over its elements up to a certain index

    Optimized for scaling for big arrays. Small arrays should just use numpy.

    Parameters
    ----------
    dim: int
        1-D Dimension of the array
    """

    def __init__(self,
                 dim):
        self.array = np.zeros(dim)
        self.bloc_size = int(np.sqrt(dim))
        self.blocsum = np.zeros((dim-1)//self.bloc_size+1)

    def insert(self,
               position,
               add_value=1):
        """Add an value in the array at a specific index

        Parameters
        ----------
        position: int
            Index in the vector to add the element

        add_value: float, optional
            value to add at the specified position
            Default is 1
        """
        self.array[position] += add_value
        ibloc = int(position/self.bloc_size)
        self.blocsum[ibloc] += add_value

    def quicksum(self,
                 position):
        """Sums the array quickly from index 0 to a specified index

        Parameters
        ----------
        position: int
            Index until which to sum the array, exclusively
        """
        ibloc = int(position/self.bloc_size)
        return (self.blocsum[:ibloc].sum() +
                self.array[ibloc*self.bloc_size:position].sum())


class utree():
    """
    This class does segmentation based of the efficiency of a specific
    binary variable to affect another one.

    For example, if we want to test the effect of a certain intervention
    (bin_data = True or False depending on if the intervention was performed)
    on the reduction of injuries (y_data). This code will select subsection
    of the dataset based on other variables like time, size, location
    (split_data) to pinpoint in which subsection the intervention is the most
    (or least) efficient.

    This tree's algorithm picks a single partitioning based on split_data,
    and calculates the Mann-Whitney U-test between the y_data with
    bin_data=True and bin_data=False, to check is bin_data affect y_data in a
    significant way. This U-test value is calculated for *both* two new leaves
    created by a split based on split_data, for every possible split (using
    clever-ish brute force).

    For each possible cutoff the "value" of the cutoff is computed as the
    product of quantities to maximize the prediction of the effect of
    bin_data to split_data, mainly:
        - The difference in means
        - The U-test value between bin_data=True and bin_data=False
        - The amount of data points bin_data=True and bin_data=False

    For more info on how this value is calculated, see
    https://github.com/nicolasberube/Utree

    Attributes
    ----------
    y_data: 1D data vector
    Dependant variable on which you want to do segmentation.
    The tree leaves will contain statistically different distributions
    of y_data.

    split_data: 2D data vector
    Data used to do the segmentation on. The tree will be split according
    to values of x_datas.
    If the column labels of split_data (for pandas dataframe for example) are
    not integers, they need to be specified in the labels parameters.

    bin_data: 1D bool vector
    Independant variable affecting y_data. The tree will be split to create
    leaves where the effect of bin_data on y_data is statiscially different
    between leaves.

    max_tree_depth: int, optional
    Maximum depth of the decision tree. Value of 0 means
    infinite depth, which means value_threshold needs to be used as
    a stopping criterion.
    Default is 6.

    value_threshold: float, optional
    Threshold of the "value" calculated based on the U-test
    that a cutoff needs to be over to happen.
    If 0, the tree will branch until the max depth is reached.
    Default is 0.

    labels: list, optional
    List of the keys of the columns of split_data, in case they are not
    a range of integers.
    If None, indexing integers will be used. Default is None.

    Attributes
    ----------
    tree: dict of {int: node()}
    Decision tree of the data. key is the leaf index number, and the
    values are the leaves as node() objects.

    subidx: binary 1D array
    Index vector specifing the data points present on a certain leaf or node.
    It changes through various calculations and functions of the class.
    Dimension should match y_data (and split_data and bin_data).

    sub_y_data: 1D data vector
    Subset of y_data present on a certain leaf or node.
    It changes through various calculations and functions of the class.
    Its dimension should match sub_split_data and sub_bin_data.

    sub_split_data: 2D data vector
    Subset of split_data present on a certain leaf or node.
    It changes through various calculations and functions of the class.
    Its dimension should match sub_y_data and sub_bin_data.

    sub_bin_data: 1D bool vector
    Subset of bin_data present on a certain leaf or node.
    It changes through various calculations and functions of the class.
    Its dimension should match sub_split_data and sub_y_data.

    sub_y_ranks: 1D float vector
    Ranks of sub_y_data. In other words, sub_y_data[i] is the
    sub_y_ranks[i]_th data point in an ordered list of sub_y_data.
    Ranks are averaged in case of duplicates.
    It changes through various calculations and functions of the class.

    sub_split_args: 2D int vector
    Indexes that would sort each column of sub_split_data. In other words,
    sub_split_data[j][sub_split_args[j][i]] is the i_th element on an ordered
    list of the j_th column sub_split_data[j].
    Indexes are not averaged, duplicates are therefore not identified.
    It changes through various calculations and functions of the class.
    """

    def __init__(self,
                 y_data,
                 split_data,
                 bin_data,
                 labels=None,
                 max_tree_depth=6,
                 value_threshold=0):
        self.tree = {0: node(0)}
        self.y_data = y_data
        self.split_data = split_data
        self.bin_data = bin_data
        self.max_tree_depth = max(max_tree_depth, 0)
        self.value_threshold = max(value_threshold, 0)
        if labels is None:
            self.labels = range(len(split_data))
        else:
            self.labels = labels
        if self.max_tree_depth == 0 and self.value_threshold == 0:
            raise NameError('Needs nonzero max_tree_depth or value_threshold')

    def compute_path(self,
                     path):
        """Computes subidx, y_rank and leaf_id for a certain path.

        Computes the list of indexes of the data points corresponding on the
        node of a certain path in self.subidx

        Computes sub_y_data = y_data[subidx]
        sub_split_data = split_data[subidx]
        sub_bin_data = bin_data[subidx]
        sub_y_ranks = ranks of sub_y_data
        sub_split_args = indexes that sort sub_split_data, for each column

        Parameters
        ----------
        path: list in int
        list of integers in (-1, 0, 1) indicating the set of decisions to take
        through the tree (lower, null, higher).
        """
        self.subidx = ~np.isnan(self.y_data)
        leaf_id = 0
        for decision in path:
            node = self.tree[leaf_id]
            if decision == 0:
                leaf_id = node.id_null
                self.subidx = (self.subidx &
                               np.isnan(self.split_data[node.label]))
            else:
                non_null = ~np.isnan(self.split_data[node.label])
                new_subidx = np.zeros_like(self.split_data[node.label],
                                           dtype=bool)
                if decision < 0:
                    leaf_id = node.id_lower
                    new_subidx[non_null] = \
                        self.split_data[node.label][non_null] < node.cutoff
                elif decision > 0:
                    leaf_id = node.id_higher
                    new_subidx[non_null] = \
                        self.split_data[node.label][non_null] > node.cutoff
                self.subidx = self.subidx & new_subidx

        self.sub_y_data = self.y_data[self.subidx]
        self.sub_split_data = self.split_data[:, self.subidx]
        self.sub_bin_data = self.bin_data[self.subidx]
        self.sub_y_ranks = rankdata(self.sub_y_data)
        self.sub_split_args = self.sub_split_data.argsort(axis=1)

    def u_data(self,
               split_label,
               above=False,
               use_scipy=True):
        """Computes data to evalute all possible splits of a certain variable

        The variable to be considered is self.sub_split_data[split_label].
        The data is computed on the set of data below all possible cutoffs

        Parameters
        ----------
        split_label: key of split_data
        Key of split_data, where the best cutoff will be computed on
        split_data[split_label]

        above: bool, optional
        Since this function is only optimized for data below the cutoff,
        it needs to be ran twice, once forward, and once backwards through
        the split_args. Above is the flag that runs it backwards
        and computes for values above the cutoff.
        Default is False.

        use_scipy: bool, optional
        Use the native Mann-Whitney from the scipy.stats package.
        If False, will use explicit calculations of the u-test,
        efficiently coded to reuse datafrom one cut-off to another,
        which scales better for large size.
        Default is False.

        Returns
        -------
        (index_cutoffs, effects, rstats, n_data)
            index_cutoffs: list of index of split_args to be used as cutoff
            effects: list of the difference of means for the data
                     below the selected cutoff of the same index
            rstats: list of the r statistic of the U-test for the data
                    below the selected cutoff of the same index
            n_data: list of (int, int), the number of data the groups
                    bin_data=True and bin_data=False, respectively
        """
        idx_cut = []
        effects = []
        rstats = []
        n_data = []

        if use_scipy:
            # Value of y_data with bin_data = True
            ys_with = []
            # Value of y_data with bin_data = False
            ys_without = []
        else:
            # sum of the y_data with bin=True. To compute means.
            sum_with = 0
            # ranksum_with is used in the calculation of the U (statistic)
            ranksum_with = 0
            # number of data points with bin=True. To compute means and ranksum
            # Corresponds to n1 in the u-test formula
            len_with = 0
            # sum of the y_data with bin=False. To compute means.
            sum_without = 0
            # number of data points with bin=False. To compute means.
            # Corresponds to n2 in the u-test formula
            len_without = 0
            # Ranks of the y_data, where equal ranks are averaged and
            # null/nan values are at the end
            y_ranks = rankdata(self.sub_y_data)

            # Each element taken from the sub_y_data dataset to the new set
            # of data that is below (or above, depending on the flag value)
            # the threshold of the cut-off value, will be counted in the
            # "inserted" arrays.
            # The index of that array corresponds to the
            # original rank of the element in the sub_y_data dataset
            # inserted_with is the same, but only counts elements
            # where bin_data=True
            # Those vectors are used to quickly calculate the new rank in the
            # new partition, to compute the u-test
            inserted_all = quicksum_array(len(self.sub_y_data))
            inserted_with = quicksum_array(len(self.sub_y_data))

        # Iteration over ordered values of split_data
        if above:
            split_args = self.sub_split_args[split_label][::-1]
        else:
            split_args = self.sub_split_args[split_label]
        null_vector = (np.isnan(self.sub_split_data[split_label]) |
                       np.isnan(self.sub_y_data) |
                       np.isnan(self.sub_bin_data))
        # Final number of elements in ys_with and ys_without
        n_with = (self.sub_bin_data & ~null_vector).sum()
        n_without = (~self.sub_bin_data & ~null_vector).sum()
        # Flag to make sure the u-test is not going to crash by having
        # all data in ys_with and ys_without identical
        data_flag = False
        first_data = None
        for idx, arg in enumerate(split_args):
            # Ignores null data
            if not null_vector[arg]:
                if first_data is None:
                    first_data = self.sub_y_data[arg]
                elif (not data_flag and (self.sub_y_data[arg] != first_data)):
                    data_flag = True

                if use_scipy:
                    if self.sub_bin_data[arg]:
                        ys_with.append(self.sub_y_data[arg])
                    else:
                        ys_without.append(self.sub_y_data[arg])
                else:
                    yrank = int(y_ranks[arg]-0.5)
                    # Calculating the rank of the new element in the
                    # bin_data=True set of the dataset below the threshold.
                    # This is only to calculate the effect of all elements in
                    # bin_data=True above the newly added element
                    # on the sum of ranks, since they will all change by +1
                    new_rank_with = (inserted_with.quicksum(yrank) +
                                     inserted_with.array[yrank]/2 + 1)
                    ranksum_with += len_with + 1 - new_rank_with
                    if self.sub_bin_data[arg]:
                        sum_with += self.sub_y_data[arg]
                        len_with += 1
                        # Calculating the rank of the new element in the
                        # dataset below the threshold to add to the ranks sum
                        new_rank = (inserted_all.quicksum(yrank) +
                                    inserted_all.array[yrank]/2 + 1)
                        ranksum_with += new_rank
                        inserted_with.insert(yrank)
                    else:
                        sum_without += self.sub_y_data[arg]
                        len_without += 1
                    inserted_all.insert(yrank)

                # If a possible cutoff is reached
                if use_scipy:
                    len_with = len(ys_with)
                    len_without = len(ys_without)
                if (idx+1 != len(split_args) and
                    (self.sub_split_data[split_label][arg] !=
                     self.sub_split_data[split_label][split_args[idx+1]]) and
                    len_with not in {0, n_with} and
                    len_without not in {0, n_without} and
                        data_flag):
                    if use_scipy:
                        statistic, pvalue = mannwhitneyu(ys_with, ys_without)
                        sum_with = sum(ys_with)
                        sum_without = sum(ys_without)
                    else:
                        statistic = ranksum_with - len_with*(len_with+1)/2
                        statistic = min(statistic,
                                        len_with*len_without-statistic)
                        #P-value: https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test#Normal_approximation_and_tie_correction

                    # The r statistic represents the proportions of all
                    # pairs between samples A and B that shows an advantage
                    # to one of the samples (normalized between 0 and 1
                    # since no effect would mean half of the pairs would
                    # show an effect)
                    r = 1 - 2*statistic/len_with/len_without
                    if above:
                        idx_cut.append(len(split_args)-idx-1.5)
                    else:
                        idx_cut.append(idx+0.5)
                    rstats.append(r)
                    effects.append(sum_with/len_with -
                                   sum_without/len_without)
                    n_data.append([len_with, len_without])
        if above:
            idx_cut = idx_cut[::-1]
            effects = effects[::-1]
            rstats = rstats[::-1]
            n_data = n_data[::-1]
        return idx_cut, effects, rstats, n_data

    def best_cutoff(self,
                    split_label):
        """Computes the best cutoff for split_data[split_label]

        Parameters
        ----------
        split_label: key of split_data
        Key of split_data, where the best cutoff will be computed on
        split_data[split_label]

        Returns
        -------
        cutoff (float), value (float)
            cutoff: value of the best cutoff for the selected variable label
            value: value of the cutoff, where the best cutoff has the higher
                value. The value is based on the u-test, the effect strength
                and the number of data in the sub-groups.
        """
        split_args = self.sub_split_args[split_label]
        split_data = self.sub_split_data[split_label]
        # This criterion for the use_scipy flag is arbitrary and needs
        # further testing
        n_unique = len(np.unique(split_data[~np.isnan(split_data)]))
        use_scipy = True
        if n_unique > len(split_data)/1000:
            use_scipy = False
        idxcut_below, effects_below, rstats_below, ndata_below =\
            self.u_data(split_label, use_scipy=use_scipy)
        idxcut_above, effects_above, rstats_above, ndata_above =\
            self.u_data(split_label, above=True, use_scipy=use_scipy)

        # Default cutoff is min(split_data) - 1
        cutoff = split_data[split_args[0]] - 1
        value = 0
        # If no cutoff was possible
        if len(idxcut_below) == 0 or len(idxcut_above) == 0:
            return cutoff, value

        # All idx_cutoffs and values for cutoffs, for debugging
        for idx in range(len(idxcut_above)):
            idxcut = idxcut_above[idx]
            if idxcut != idxcut_below[idx]:
                raise NameError('Code error, invalid split')
            value_temp = (abs(effects_above[idx] -
                              effects_below[idx]) *
                          rstats_above[idx] *
                          rstats_below[idx] *
                          min(ndata_above[idx]) *
                          min(ndata_below[idx]))
            if value_temp > value:
                cutoff = (split_data[split_args[int(idxcut)]] +
                          split_data[split_args[int(idxcut)+1]])/2
                value = value_temp
        return cutoff, value

    def compute_tree(self,
                     verbose=True):
        """Computes the decision tree.

        Parameters
        ----------
        verbose: bool, optional
            If True, prints progress bar on screen. Default is True
        """

        # Tree structure in format {leaf_id: node()}
        self.tree = {}
        # A path is list of integers in (-1, 0, 1) indicating the set of
        # decisions to take through the tree (lower, null, higher)
        # based on the specified labels and cutoff of the nodes.
        paths = [[]]
        path_idx = 0
        start_time = time()

        # Each path will point to a leaf that is not yet in the tree.
        while path_idx < len(paths):
            if verbose:
                string = f'{path_idx}/{len(paths)} ({time()-start_time:.0f} s)'
                sys.stdout.write('\r'+string[:40]+' '*(40-len(string)))
                sys.stdout.flush()
            path = paths[path_idx]
            self.compute_path(path)
            leaf = node(path_idx)
            if self.sub_y_data.size == 0:
                raise NameError('No data on the leaf error')
            if len(path) < self.max_tree_depth or self.max_tree_depth <= 0:
                cutoffs = []
                for split_label in self.labels:
                    cutoff, value = self.best_cutoff(split_label)
                    cutoffs.append([split_label, cutoff, value])
                cutoffs = sorted(cutoffs, key=lambda x: -x[2])
                split_label, cutoff, value = cutoffs[0]
                leaf.value = value
                if value > self.value_threshold:
                    leaf.label = split_label
                    leaf.cutoff = cutoff
                    leaf.id_lower = len(paths)
                    paths.append(path+[-1])
                    leaf.id_higher = len(paths)
                    paths.append(path+[1])
                    if np.isnan(self.sub_split_data[split_label]).any():
                        leaf.id_null = len(paths)
                        paths.append(path+[0])
                else:
                    leaf.is_leaf = True
                    ys_with = self.sub_y_data[self.sub_bin_data]
                    ys_without = self.sub_y_data[self.sub_bin_data]
                    leaf.n_data_with = len(ys_with)
                    leaf.n_data_without = len(ys_without)
                    if ys_with.size == 0 or ys_without.size == 0:
                        leaf.effect = 0
                    else:
                        leaf.effect = ys_with.mean() - ys_without.mean()
                self.tree[leaf.id] = leaf
            path_idx += 1

        if verbose:
            string = f'{path_idx}/{len(paths)} ({time()-start_time:.0f} s)'
            sys.stdout.write('\r'+string[:40]+' '*(40-len(string)))
            sys.stdout.flush()
            print()

    def tree_to_leaf(self,
                     x_row):
        """
        Navigates through the tree based on the X labels (split_data)

        Parameters
        ----------
        x_row : 1D data vector
            X labels with the same structure as a row in split_data.

        Returns
        -------
        node()
            Node corresponding to the leaf where that data point would be.
        """
        node = self.tree[0]
        while True:
            if node.is_leaf:
                return node
            val = x_row[node.label]
            if np.isnan(val):
                node = self.tree[node.id_null]
            elif val <= node.cutoff:
                node = self.tree[node.id_lower]
            elif val >= node.cutoff:
                node = self.tree[node.id_higher]
            else:
                raise NameError


if __name__ == '__main__':
    n_data = 100
    n_splits = 1
    effect = np.random.rand(n_data)
    y_data = np.random.rand(n_data)
    y_data[1] = y_data[0]
    split_data = np.random.rand(n_splits, n_data)*effect
    split_data[0][-1] = np.nan
    split_data[0][-2] = -np.inf
    split_data[0][-3] = np.inf
    split_data[0][-4] = split_data[0][-5]
    bin_data = effect*y_data
    bin_data = bin_data > bin_data.mean()
    U = utree(y_data, split_data, bin_data)
    U.compute_tree()
