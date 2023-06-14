import numpy as np

class Node:
    '''
    Class for decision tree node objects. Each node contains it's value, which
    which attribute it is, the most common class under the current node and
    the nodes children.
    '''
    
    def __init__(self, value, attribute, most_common):
        self.value = value
        self.attribute = attribute
        self.most_common = most_common
        self.children = {}


class Decision_Tree:
    '''
    The decision tree class is used to build a decision tree using the ID3
    algorithm.
    '''
    
    def __init__(self, data, depth=None):
        '''
        Initialize all class variables and build the decision tree.
        
        data --> dataset used to build the decision tree
        '''
        
        self.tree = None
        self.default = self.find_default(data)
        
        attributes = np.arange(len(data[0, :-1]))
        self.tree = self.build_tree(data, attributes, self.default, depth)
        
    
    def find_default(self, data):
        '''
        Find the outcome that has the greatest prior at the current node to be
        used as the default value for the node.
        
        data --> remaining data at current node
        '''
        
        y_vals = data[:,-1]
        outcomes = np.unique(y_vals)
        
        priors = []
        # loop over all possible outcomes, finding the priors for each
        for outcome in outcomes:
            prior = (len(np.where(y_vals == outcome)[0]))/len(y_vals)
            priors.append(prior)
            
        # return outcome with largest prior
        return outcomes[np.argmax(priors)]
    
    
    def best_entropy(self, data, attributes):
        '''
        Of the remaining attributes to split on in the subtree, find the 
        attribute with the lowest entropy.
        
        data --> remaining data at current node
        attributes --> remaining attributes to split on
        '''
        
        entropies = []
        total_len = len(data)   # total examples at current node
        # get entropy for each attribute
        for attribute in attributes:
            a_vals = np.unique(data[:, attribute])   # attribute options
            outcomes = np.unique(data[:,-1])   # possible outcomes

            sub_entropy = 0   # total entropy for attribute

            # loop over attribute options
            for a in a_vals:
                # get the number of data points where the attribute equals current option
                attr_mask = data[:, attribute] == a
                attr_data = data[attr_mask, :]
                attr_len = len(attr_data)

                attr_entropy = 0   # total option entropy
                # loop over possible outcomes
                for outcome in outcomes:
                    outcome_mask = attr_data[:, -1] == outcome
                    outcome_data = attr_data[outcome_mask, :]
                    outcome_len = len(outcome_data)
                    if outcome_len == 0:
                        outcome_len = .000000001

                    temp_entropy = (-outcome_len/attr_len)*(np.log(outcome_len/attr_len) / np.log(len(outcomes)))
                    attr_entropy += temp_entropy

                weight_attr_entropy = (attr_len/total_len)*attr_entropy
                sub_entropy += weight_attr_entropy

            entropies.append(sub_entropy)

        return attributes[np.argmin(entropies)]
        
        
    def build_tree(self, data, attributes, default, depth=None):
        '''
        Recursively build the decision tree, each time spliting on the feature
        with the lowest entropy. If there are no reamin attributes to split on
        or all the data has been used, add a leaf node with the default value.
        
        data --> remaining data at current node
        attributes --> remaining attributes to split on
        default --> default outcome used when no data or attributes remain
        '''
        
        if len(data) == 0:
            node = Node(default, -1, default)
            return node
        elif len(np.unique(data[:,-1])) == 1:
            node = Node(np.unique(data[:,-1])[0], -1, np.unique(data[:,-1])[0])
            return node
        elif len(attributes) == 0:
            values, counts = np.unique(data[:,-1], return_counts=True)
            node = Node(values[np.argmax(counts)], -1, values[np.argmax(counts)])
            return node
        elif depth == 0:
            values, counts = np.unique(data[:,-1], return_counts=True)
            node = Node(values[np.argmax(counts)], -1, values[np.argmax(counts)])
            return node
        else:
            values, counts = np.unique(data[:,-1], return_counts=True)
            most_common = values[np.argmax(counts)]
            best_attr = self.best_entropy(data, attributes)
            node = Node(None, best_attr, most_common)
            options = np.unique(data[:, best_attr])
            
            i, = np.where(attributes == best_attr)
            sub_attr = np.delete(attributes, i)
            
            for option in options:
                option_mask = data[:, best_attr] == option
                sub_data = data[option_mask, :]
                sub_default = self.find_default(sub_data)
                if depth:
                    depth -= 1
                sub_node = self.build_tree(sub_data, sub_attr, sub_default, depth)
                node.children[option] = sub_node
            
            return node
            
        
    def predict(self, data, tree):
        '''
        Give a new data point and a decision tree, recursive search through
        the tree in order to predict the outcome for the new data point.
        
        data --> new data point to predict
        tree --> decision tree used to predict
        '''
        
        if not tree.value == None:
            return tree.value
        else:
            index = tree.attribute
            
            attr_option = data[index]
            if attr_option in tree.children:
                sub_tree = tree.children[attr_option]
                return self.predict(data, sub_tree)
            else:
                return tree.most_common
    