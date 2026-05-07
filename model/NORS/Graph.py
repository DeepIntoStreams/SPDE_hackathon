from __future__ import annotations

from itertools import chain, combinations_with_replacement as comb

import numpy as np


class Graph:
    """DLR-compatible graph builder for NORS model feature vectors."""

    def __init__(self, integration, rule, height, deg, derivative=False):
        self.I = integration
        self.deg = deg
        self.H = height
        self.R = rule
        self.derivative = derivative

    @staticmethod
    def tree_deg(dic, done):
        return sum(done[word] * power for word, power in dic.items())

    @staticmethod
    def trees_multiply(model, dic):
        trees = list(dic.keys())
        value = model[trees[0]] ** dic[trees[0]]
        for tree in trees[1:]:
            value = value * (model[tree] ** dic[tree])
        return value

    def extra_trees(self, W):
        values = self.R.values.copy()
        if "xi" in self.R.degrees:
            values["xi"] = W
        return {idx: self.trees_multiply(values, dic) for idx, dic in self.R.rule_extra.items()}

    def create_model_graph(self, W, lollipop=None, extra_planted=None, extra_deg=None):
        if lollipop is None:
            model = self.I({"xi": W}, derivative=self.derivative)
        else:
            model = {"I[xi]": lollipop}

        graph = {"xi": {}}
        planted = {"I[xi]"}
        done = self.R.degrees.copy()
        graph["I[xi]"] = {"xi": 1}

        if extra_planted is not None:
            model.update(extra_planted)
            planted = planted.union(sorted(extra_planted.keys()))
            graph.update({key: {} for key in extra_deg.keys()})
            done.update(extra_deg)

        self._expand_graph(model, graph, planted, done, W)
        return graph

    def create_model_graph_2d(self, W, X, lollipop=None, extra_planted=None, extra_deg=None):
        if lollipop is None:
            model = self.I({"xi": W}, derivative=self.derivative)
        else:
            model = {"I[xi]": lollipop}

        graph = {"xi": {}}
        planted = {"I[xi]"}
        done = self.R.degrees.copy()
        graph["I[xi]"] = {"xi": 1}

        if extra_planted is not None:
            model.update(extra_planted)
            planted = planted.union(sorted(extra_planted.keys()))
            graph.update({key: {} for key in extra_deg.keys()})
            done.update(extra_deg)

        self._expand_graph(model, graph, planted, done, W)
        return graph

    def _expand_graph(self, model, graph, planted, done, W):
        extra_values = self.extra_trees(W)
        for _ in range(1, self.H):
            for width in range(1, self.R.max + 1):
                for words in comb(sorted(planted), width):
                    tree, dic = self.R.words_to_tree(words)
                    temp_deg = self.tree_deg(dic, done)
                    if (
                        tree not in done
                        and tree not in self.R.exceptions
                        and width <= self.R.free_num
                        and temp_deg + self.R.degrees["I"] <= self.deg
                    ):
                        model[tree] = self.trees_multiply(model, dic)
                        graph[tree] = dic
                    done[tree] = temp_deg

                    for idx, extra_dic_value in extra_values.items():
                        if width > self.R.rule_power[idx]:
                            continue
                        extra_tree, extra_dic = self.R.words_to_tree(self.R.rule_to_words(idx))
                        new_tree = extra_tree + f"({tree})"
                        deg = done[tree] + self.tree_deg(extra_dic, done)
                        if new_tree in done or new_tree in self.R.exceptions or deg > self.deg:
                            continue
                        base = model[tree] if tree in model else self.trees_multiply(model, dic)
                        model[new_tree] = base * extra_dic_value
                        graph[new_tree] = dict(chain.from_iterable(d.items() for d in (extra_dic, {tree: 1})))
                        done[new_tree] = deg

            this_round = self.I(model, planted, self.R.exceptions, self.derivative)
            keys = [tree for tree in sorted(this_round.keys()) if tree not in self.R.degrees and tree not in planted]
            for integrated_tree in keys:
                inner = integrated_tree[2:-1] if integrated_tree[1] == "[" else integrated_tree[3:-1]
                model[integrated_tree] = this_round.pop(integrated_tree)
                graph[integrated_tree] = graph[inner] if inner and inner[0] != "I" else {inner: 1}
                planted.add(integrated_tree)
                if inner not in planted and inner in model:
                    model.pop(inner)
                    graph.pop(inner, None)
                done[integrated_tree] = done[inner] + self.R.degrees["I"]

    @staticmethod
    def discrete_diff_2d(vec, N, axis, flatten=True, higher=True):
        arr = vec.copy()
        if len(arr.shape) == 1:
            arr = arr.reshape(len(vec) // N, N)
        if axis == 1:
            if higher:
                arr[:, :-1, :] = (np.roll(arr[:, :-1, :], -1, axis=1) - np.roll(arr[:, :-1, :], 1, axis=1)) / 2
            else:
                arr[:, :-1, :] = arr[:, :-1, :] - np.roll(arr[:, :-1, :], 1, axis=1)
            arr[:, -1, :] = arr[:, 0, :]
        elif axis == 2:
            if higher:
                arr[:, :, :-1] = (np.roll(arr[:, :, :-1], -1, axis=2) - np.roll(arr[:, :, :-1], 1, axis=2)) / 2
            else:
                arr[:, :, :-1] = arr[:, :, :-1] - np.roll(arr[:, :, :-1], 1, axis=2)
            arr[:, :, -1] = arr[:, :, 0]
        else:
            raise ValueError(f"axis must be 1 or 2, got {axis}")
        return arr.flatten() if flatten else arr
