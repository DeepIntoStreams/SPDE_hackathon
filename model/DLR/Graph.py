# Adapted from https://github.com/sdogsq/DLR-Net
# Modified for current implementation by the authors of SPDEBench


from itertools import combinations_with_replacement as comb
from itertools import chain
import numpy as np


class Graph():

    def __init__(self, integration, rule, height, deg, derivative=False):
        # Integration function that creates integration({tree: values}) returns {I[tree]: I[values]}
        self.I = integration
        self.deg = deg  # maximum degree of the created trees
        self.H = height  # maximum height of the trees
        self.R = rule  # Rule involving several extra trees and widths
        self.models = None
        self.size = 0  # number of realizations of models
        self.derivative = derivative  # True if derivatives are present in the model. At the moment only differentiation order <= 1 are allowed

    # A helper function that returns degree of the tree with dictionary dic.
    def tree_deg(self, dic, done):
        return sum([done[w] * dic[w] for w in dic])

    # Helper function that multiplies trees. 
    # Given a dictionary {tree: power} outputs \prod_{tree} tree^power
    def trees_multiply(self, model, dic):

        trees = list(dic.keys())
        w1 = trees[0]

        if len(dic) == 1:  # If there is only one tree, it is faster to just return tree^n
            return model[w1] ** dic[w1]
        if len(dic) == 2:  # If only two unique trees is multiplied faster to return this
            w2 = trees[1]
            return (model[w1] ** dic[w1]) * (model[w2] ** dic[w2])

        tree_val = model[w1] ** dic[w1]
        for i in range(1, len(trees)):
            tree_val *= model[trees[i]] ** dic[trees[i]]
        return tree_val

    # Creates all possible combinations of the values of the trees that can multiply planted trees.

    def extra_trees(self, W):
        trees_vals = self.R.values.copy()
        if 'xi' in self.R.degrees:
            trees_vals['xi'] = W
        dic_values = {}
        for i in self.R.rule_extra:
            dic_values[i] = self.trees_multiply(trees_vals, self.R.rule_extra[i])

        return dic_values

    # Given a Rule, creates a Model where all trees conform to the rule and are of degree <= 'deg'.
    def create_model_graph(self, W, lollipop=None, extra_planted=None, extra_deg=None):

        # first let the model consist of the I[xi] only.
        if lollipop is None:  # if lollipop is not given Integrate noise W
            model = self.I({'xi': W}, derivative=self.derivative)
        else:  # otherwise simply add lollipop
            model = {'I[xi]': lollipop}
        # 'planted' is a set that keeps track of the planted trees I[\tau].
        # 'done' is the dictionary that keeps track of all trees that were created together with their degree.

        graph = dict()

        planted, done = {
            'I[xi]'}, self.R.degrees.copy()  # create set of planted trees and a dictinary of the trees degrees

        graph['xi'] = {}
        # Add planted trees that correspond to functions u^i for i in \mathcal{J}.
        if extra_planted is not None:
            model.update(extra_planted)
            planted = planted.union(sorted(extra_planted.keys()))
            # planted = planted.union(set(extra_planted.keys()))
            # for key in extra_deg.keys():
            #     if 'u_0' in key:
            #         graph.update({'u_0' : {}})
            # graph.update({key: {'u_0' : 1} if 'u_0' in key else {} for key in extra_deg.keys()})
            graph.update({key: {} for key in extra_deg.keys()})
            # graph.update(extra_deg)
            # print(planted)
            done.update(extra_deg)
            # print(done)
        # If necessary add spatial derivative of the I[xi] denoted by I'[xi]
        if self.derivative:
            planted.add("I'[xi]")
            done["I'[xi]"] = done["I[xi]"] - 1

        extra_trees_values = self.extra_trees(W)

        graph['I[xi]'] = {'xi': 1}
        # Add trees of greater height
        for j in range(1, self.H):

            # Compute multiplications of the planted trees. self.R.max is the maximum possible width
            for k in range(1, self.R.max + 1):  # k is the number of trees multiplied
                # check all possible combinations of product of k planted trees
                for words in comb(sorted(planted), k):
                    # for words in comb([w for w in planted], k):
                    tree, dic = self.R.words_to_tree(words)  # create one tree product out of the list of trees
                    temp_deg = self.tree_deg(dic, done)  # compute the degree of this tree
                    # check if the tree needs to be added. k <= self.R.free_num checks if the product of k trees can exist
                    if tree not in done and tree not in self.R.exceptions and k <= self.R.free_num and temp_deg + \
                            self.R.degrees['I'] <= self.deg:
                        model[tree] = self.trees_multiply(model, dic)  # add this tree to the model
                        graph[tree] = dic
                        # if necessary add the tree multiplied by extra trees.
                    done[tree] = temp_deg  # include the tree to the done dictionary together with its degree
                    # multiply by the extra trees if such are present in the rule
                    for i in extra_trees_values:  # add extra trees that correspond to multiplicative width
                        if k <= self.R.rule_power[i]:  # check if extra tree can multiply the k product of planted trees
                            extra_tree, extra_dic = self.R.words_to_tree(self.R.rule_to_words(i))
                            new_tree = extra_tree + '(' + tree + ')'  # shape of the new tree
                            deg = done[tree] + self.tree_deg(extra_dic, done)  # degree of a new tree
                            if new_tree not in done and new_tree not in self.R.exceptions and deg <= self.deg:
                                if tree in model:
                                    model[new_tree] = model[tree] * extra_trees_values[i]
                                    graph[new_tree] = dict(
                                        chain.from_iterable(d.items() for d in (extra_dic, {tree: 1})))
                                else:
                                    model[new_tree] = self.trees_multiply(model, dic) * extra_trees_values[i]
                                    graph[new_tree] = dict(chain.from_iterable(d.items() for d in (extra_dic, dic)))
                                done[new_tree] = done[tree] + self.tree_deg(extra_dic, done)
                                # integrate trees from the previous iteration.
            this_round = self.I(model, planted, self.R.exceptions, self.derivative)
            keys = [tree for tree in sorted(this_round.keys()) if tree not in self.R.degrees and tree not in planted]
            # keys = [tree for tree in this_round.keys() if tree not in self.R.degrees and tree not in planted]
            # include theese integrated trees to the model. Don't include trees that are not of the form I[\tau]
            for IZ in keys:
                if IZ[1] == "[":
                    Z = IZ[2:-1]  # IZ = I[Z]
                else:
                    Z = IZ[3:-1]  # IZ = I'[Z]
                model[IZ] = this_round.pop(IZ)
                graph.update({IZ: graph[Z] if Z[0] != 'I' else {Z: 1}})
                planted.add(IZ)  # add tree IZ to planted
                if Z not in planted and Z in model:
                    model.pop(Z)  # Delete Z tree from the model if it is not planted
                    graph.pop(Z)

                if IZ[1] == "[":
                    done[IZ] = done[Z] + self.R.degrees['I']  # add degree to IZ
                else:
                    done[IZ] = done[Z] + self.R.degrees['I'] - 1
        # delete all the trees of the form \partial_x I[\tau] and only keep
        # trees of the form I[\tau]
        if self.derivative:
            model = {IZ: model[IZ] for IZ in model if IZ[1] != "'"}

        return graph

    def create_ns_graph_2d():
        def Rule1(S):
            R = {}
            R.update(S)
            R.update({f"J1[{tau}]": {tau: 1} for tau in S if tau[-1] == ']' and f"J1[{tau}]" not in S.keys()})
            R.update({f"J2[{tau}]": {tau: 1} for tau in S if tau[-1] == ']' and f"J2[{tau}]" not in S.keys()})
            return R

        def Rule2(S):
            R = {}
            R.update(S)
            R.update({f"I({tau})": {tau: 1} for tau in S if tau[-1] == ']' and f"I[{tau}]" not in S.keys()})
            R.update({f"I1[{tau}]": {f"I({tau})": 1} for tau in S if tau[-1] == ']' and f"I1[{tau}]" not in S.keys()})
            R.update({f"I2[{tau}]": {f"I({tau})": 1} for tau in S if tau[-1] == ']' and f"I2[{tau}]" not in S.keys()})
            R.update({f"J1({tau})": {tau: 1} for tau in S if f"J1[{tau}]" not in S.keys()})
            R.update(
                {f"I(J1[{tau}]{tau})": {f"J1[{tau}]" if f"J1[{tau}]" in S.keys() else f"J1({tau})": 1, f"{tau}": 1} for
                 tau in S if tau[-1] == ']' and f"I[J1[{tau}]{tau}]" not in S.keys()})
            R.update({f"I1[J1[{tau}]{tau}]": {f"I(J1[{tau}]{tau})": 1} for tau in S if
                      tau[-1] == ']' and f"I1[J1[{tau}]{tau}]" not in S.keys()})
            R.update({f"J2({tau})": {tau: 1} for tau in S if f"J2[{tau}]" not in S.keys()})
            R.update(
                {f"I(J2[{tau}]{tau})": {f"J2[{tau}]" if f"J2[{tau}]" in S.keys() else f"J2({tau})": 1, f"{tau}": 1} for
                 tau in S if tau[-1] == ']' and f"I[J2[{tau}]{tau}]" not in S.keys()})
            R.update({f"I2[J2[{tau}]{tau}]": {f"I(J2[{tau}]{tau})": 1} for tau in S if
                      tau[-1] == ']' and f"I2[J2[{tau}]{tau}]" not in S.keys()})
            return R

        S = {'I_c[u_0]': {}, 'I[xi]': {'xi': 1}}
        Res = {'xi': {}}
        Res.update(Rule2(Rule1(S)))
        return Res

    def create_model_graph_2d(self, W, X, lollipop=None, extra_planted=None, extra_deg=None):
        dx = X[1, 0] - X[0, 0]

        # first let the model consist of the I[xi] only.
        if lollipop is None:  # if lollipop is not given Integrate noise W
            model = self.I({'xi': W}, derivative=self.derivative)
            if (self.derivative):
                model.update({'I1[xi]': self.discrete_diff_2d(model['I[xi]'], X.shape[0], axis=1, flatten=False,
                                                              higher=False) / dx, \
                              'I2[xi]': self.discrete_diff_2d(model['I[xi]'], X.shape[0], axis=2, flatten=False,
                                                              higher=False) / dx})
        else:  # otherwise simply add lollipop
            model = {'I[xi]': lollipop,
                     'I1[xi]': self.discrete_diff_2d(lollipop, X.shape[0], axis=1, flatten=False, higher=False) / dx, \
                     'I2[xi]': self.discrete_diff_2d(lollipop, X.shape[0], axis=2, flatten=False, higher=False) / dx}
        # 'planted' is a set that keeps track of the planted trees I[\tau].
        # 'done' is the dictionary that keeps track of all trees that were created together with their degree.

        graph = dict()

        planted, done = {
            'I[xi]'}, self.R.degrees.copy()  # create set of planted trees and a dictinary of the trees degrees

        graph['xi'] = {}

        # Add planted trees that correspond to functions u^i for i in \mathcal{J}.
        if extra_planted is not None:
            model.update(extra_planted)
            planted = planted.union(sorted(extra_planted.keys()))
            # planted = planted.union(set(extra_planted.keys()))
            graph.update({key: {} for key in extra_deg.keys()})
            done.update(extra_deg)
            if self.derivative:
                for arr in extra_planted.values():
                    model.update(
                        {'I1_c[u_0]': self.discrete_diff_2d(arr, X.shape[0], axis=1, flatten=False, higher=False) / dx, \
                         'I2_c[u_0]': self.discrete_diff_2d(arr, X.shape[0], axis=2, flatten=False, higher=False) / dx})
                    planted = planted.union({'I1_c[u_0]', 'I2_c[u_0]'})
                    graph.update({'I1_c[u_0]': {"I_c[u_0]": 1}, 'I2_c[u_0]': {"I_c[u_0]": 1}})
                    done.update({'I1_c[u_0]': 1, 'I2_c[u_0]': 1})
        # If necessary add spatial derivative of the I[xi] denoted by I'[xi]

        graph['I[xi]'] = {'xi': 1}

        if self.derivative:
            planted.add("I1[xi]")
            planted.add("I2[xi]")
            graph["I1[xi]"] = {'I[xi]': 1}
            graph["I2[xi]"] = {'I[xi]': 1}
            done["I1[xi]"] = done["I[xi]"] - 1
            done["I2[xi]"] = done["I[xi]"] - 1

        extra_trees_values = self.extra_trees(W)
        # Add trees of greater height
        for j in range(1, self.H):

            # Compute multiplications of the planted trees. self.R.max is the maximum possible width
            for k in range(1, self.R.max + 1):  # k is the number of trees multiplied
                # check all possible combinations of product of k planted trees
                for words in comb(sorted(planted), k):
                    # for words in comb([w for w in planted], k):
                    tree, dic = self.R.words_to_tree(words)  # create one tree product out of the list of trees
                    temp_deg = self.tree_deg(dic, done)  # compute the degree of this tree
                    # check if the tree needs to be added. k <= self.R.free_num checks if the product of k trees can exist
                    if tree not in done and tree not in self.R.exceptions and k <= self.R.free_num and temp_deg + \
                            self.R.degrees['I'] <= self.deg:
                        model[tree] = self.trees_multiply(model, dic)  # add this tree to the model
                        graph[tree] = dic
                        # if necessary add the tree multiplied by extra trees.
                    done[tree] = temp_deg  # include the tree to the done dictionary together with its degree
                    # multiply by the extra trees if such are present in the rule
                    for i in extra_trees_values:  # add extra trees that correspond to multiplicative width
                        if k <= self.R.rule_power[i]:  # check if extra tree can multiply the k product of planted trees
                            extra_tree, extra_dic = self.R.words_to_tree(self.R.rule_to_words(i))
                            new_tree = extra_tree + '(' + tree + ')'  # shape of the new tree
                            deg = done[tree] + self.tree_deg(extra_dic, done)  # degree of a new tree
                            if new_tree not in done and new_tree not in self.R.exceptions and deg <= self.deg:
                                if tree in model:
                                    model[new_tree] = model[tree] * extra_trees_values[i]
                                    graph[new_tree] = dict(
                                        chain.from_iterable(d.items() for d in (extra_dic, {tree: 1})))
                                else:
                                    model[new_tree] = self.trees_multiply(model, dic) * extra_trees_values[i]
                                    graph[new_tree] = dict(chain.from_iterable(d.items() for d in (extra_dic, dic)))
                                done[new_tree] = done[tree] + self.tree_deg(extra_dic, done)

                                # integrate trees from the previous iteration.
            this_round = self.I(model, planted, self.R.exceptions, self.derivative)
            keys = [tree for tree in sorted(this_round.keys()) if tree not in self.R.degrees and tree not in planted]
            # keys = [tree for tree in this_round.keys() if tree not in self.R.degrees and tree not in planted]

            # include theese integrated trees to the model. Don't include trees that are not of the form I[\tau]
            for IZ in keys:
                if IZ[1] == "[":
                    Z = IZ[2:-1]  # IZ = I[Z]
                else:
                    Z = IZ[3:-1]  # IZ = I'[Z]
                model[IZ] = this_round.pop(IZ)
                graph.update({IZ: graph[Z] if Z[0] != 'I' else {Z: 1}})
                planted.add(IZ)  # add tree IZ to planted
                if Z not in planted and Z in model:
                    model.pop(Z)  # Delete Z tree from the model if it is not planted
                    graph.pop(Z)

                if IZ[1] == "[":
                    done[IZ] = done[Z] + self.R.degrees['I']  # add degree to IZ
                else:
                    done[IZ] = done[Z] + self.R.degrees['I'] - 1

        # delete all the trees of the form \partial_x I[\tau] and only keep
        # trees of the form I[\tau]

        if self.derivative:
            model = {IZ: model[IZ] for IZ in model if IZ[1] != "1" and IZ[1] != "2"}
            # graph = {IZ: graph[IZ] for IZ in graph if IZ[1] != "1" and IZ[1] != "2"}

        return graph

    def discrete_diff_2d(self, vec, N, axis, f=None, flatten=True, higher=True):
        a = vec.copy()
        if len(a.shape) == 1:
            a = a.reshape(len(vec) // N, N)
        if axis == 1:
            if f is None:
                if higher:  # central approximation of a dervative
                    a[:, :-1, :] = (np.roll(a[:, :-1, :], -1, axis=1) - np.roll(a[:, :-1, :], 1, axis=1)) / 2
                else:
                    a[:, :-1, :] = a[:, :-1, :] - np.roll(a[:, :-1, :], 1, axis=1)
            else:
                # if a finction f given output d f(vec) / dx instead of d(vec)/dx
                if higher:
                    a[:, :-1, :] = (self.vectorized(f, np.roll(a[:, :-1, :], -1, axis=1)) - self.vectorized(f, np.roll(
                        a[:, :-1, :], 1, axis=1))) / 2
                else:
                    a[:, :-1, :] = self.vectorized(f, a[:, :-1, :]) - self.vectorized(f,
                                                                                      np.roll(a[:, :-1, :], 1, axis=1))
            a[:, -1, :] = a[:, 0, :]  # enforce periodic boundary condions
            if flatten:
                return a.flatten()
        if axis == 2:
            if f is None:
                if higher:  # central approximation of a dervative
                    a[:, :, :-1] = (np.roll(a[:, :, :-1], -1, axis=2) - np.roll(a[:, :, :-1], 1, axis=2)) / 2
                else:
                    a[:, :, :-1] = a[:, :, :-1] - np.roll(a[:, :, :-1], 1, axis=2)
            else:
                # if a finction f given output d f(vec) / dx instead of d(vec)/dx
                if higher:
                    a[:, :, :-1] = (self.vectorized(f, np.roll(a[:, :, :-1], -1, axis=2)) - self.vectorized(f, np.roll(
                        a[:, :, :-1], 1, axis=2))) / 2
                else:
                    a[:, :, :-1] = self.vectorized(f, a[:, :, :-1]) - self.vectorized(f,
                                                                                      np.roll(a[:, :, :-1], 1, axis=2))
            a[:, :, -1] = a[:, :, 0]  # enforce periodic boundary condions
            if flatten:
                return a.flatten()
        return a





