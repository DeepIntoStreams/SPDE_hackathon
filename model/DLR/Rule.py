class Rule():
    
    # This is a class that creates a rule for the model. It consists of additive, multiplicative and degrees 
    # hyperparameters. Note that symbols that represent functions are called trees in this setting.
    # All components that do not correspond to the additive width are added separately.
    # In this class and everywhere else planted trees are the ones of the form I[\tau]. 
    
    def __init__(self, kernel_deg = 2, noise_deg = -1.5, free_num = 0, extra_degrees = None, exceptions = set()):

        self.rule_extra = {}
        self.rule_power = {}
        self.values = {} # values of extra trees that are present in components
        self.count = 0 # number of extra components in the rule
        self.max = free_num # maximum number of planted trees that can be multiplied
        self.free_num = free_num # number of planted trees that can be multiplied without extra trees. (additive width)
        self.exceptions = exceptions # set of trees that do not need to be presented in the rule
        # kernel_deg degree by which integration increases the tree
        # noise_deg degree of the noise or initial condition
        self.degrees = {'I' : kernel_deg, 'xi' : noise_deg, 'I[xi]' : kernel_deg + noise_deg}
        if extra_degrees is not None:
            self.degrees.update(extra_degrees)
            
    # add tree degree
    def add_tree_deg(self, tree, deg):
        self.degrees = {tree: deg}
    
    # add tree to the set of exceptions
    def add_exceptions(self, tree):
        self.exceptions.add(tree)
            
    def check_in_present(self, new):
        for a in new:
            if a not in self.degrees:
                return False
        return True
    
    # Add a component to the rule. E.g. 3, {'xi': 1, 'X':1} means that all the planted trees of width <= 3 
    # will be multiplied by xi * X. Note that component 3, {'xi': 1} and , 3, {'X': 1} needs to be aded 
    # separately at the moment. 
    def add_component(self, n, dic):
        # n is the max number of planted trees for this component. Corresponds to the multiplicative width of this component
        if self.check_in_present(dic):
            if n > self.max:
                self.max = n
            self.count += 1
            self.rule_extra[self.count] = dic # dictionary of extra trees that can multiply n planted trees
            self.rule_power[self.count] = n # keeps track of the maximum number of planted tree in this component
        else:
            print('Some of the trees are not in the degrees dictionary.')

    def assign_value(self, tree, data): # assigns values for some tree in the rule
        if self.check_in_present({tree}):
            self.values[tree] = data
        else:
            print('This tree is not present in the degrees dictionary.')

    def rule_to_words(self, i): # turns rule component {'xi': 2, 'X':1} into ['xi','xi', 'X'] for the function 
        tree = []
        for x in self.rule_extra[i]:
            tree += [x] * self.rule_extra[i][x]
        return tree
    
    # Turns words ['xi','xi','X'] into tree '(xi)^2(X)' and a rule component {'xi': 2, 'X':1}
    def words_to_tree(self, words):
        words = [w for w in words]
        if len(words) == 1:
            return words[0], {words[0]: 1}
        dic = {} # rule component
        for w in words:
            dic[w] = dic.get(w, 0) + 1

        tree = ''
        # "multiply"
        for w in dic:
            if dic[w] == 1:
                tree += '(' + w + ')'
            else:
                tree += '(' + w + ')^' + str(dic[w])
        return tree, dic