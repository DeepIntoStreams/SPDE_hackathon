from __future__ import annotations


class Rule:
    """DLR-compatible rule object for non-singular NORS MFV construction.

    The rule specifies which products of planted trees may be formed before
    parabolic integration. This is intentionally the same lightweight object
    used by DLR, but namespaced under NORS so this package is self-contained.
    """

    def __init__(self, kernel_deg=2, noise_deg=-1.5, free_num=0, extra_degrees=None, exceptions=None):
        self.rule_extra = {}
        self.rule_power = {}
        self.values = {}
        self.count = 0
        self.max = free_num
        self.free_num = free_num
        self.exceptions = set() if exceptions is None else set(exceptions)
        self.degrees = {"I": kernel_deg, "xi": noise_deg, "I[xi]": kernel_deg + noise_deg}
        if extra_degrees is not None:
            self.degrees.update(extra_degrees)

    def add_tree_deg(self, tree, deg):
        self.degrees[tree] = deg

    def add_exceptions(self, tree):
        self.exceptions.add(tree)

    def check_in_present(self, new):
        return all(tree in self.degrees for tree in new)

    def add_component(self, n, dic):
        if not self.check_in_present(dic):
            raise ValueError("Some of the trees are not in the degrees dictionary.")
        if n > self.max:
            self.max = n
        self.count += 1
        self.rule_extra[self.count] = dic
        self.rule_power[self.count] = n

    def assign_value(self, tree, data):
        if not self.check_in_present({tree}):
            raise ValueError(f"Tree {tree} is not present in the degrees dictionary.")
        self.values[tree] = data

    def rule_to_words(self, i):
        words = []
        for tree, power in self.rule_extra[i].items():
            words += [tree] * power
        return words

    @staticmethod
    def words_to_tree(words):
        words = list(words)
        if len(words) == 1:
            return words[0], {words[0]: 1}
        counts = {}
        for word in words:
            counts[word] = counts.get(word, 0) + 1
        name = ""
        for word, power in counts.items():
            name += f"({word})" if power == 1 else f"({word})^{power}"
        return name, counts
