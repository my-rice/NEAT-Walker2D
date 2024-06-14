from neat.genome import DefaultGenomeConfig, DefaultGenome

from itertools import count

from neat.six_util import iteritems, iterkeys
from neat.genes import DefaultConnectionGene, DefaultNodeGene
class DefaultGenomeConfigWrapper(DefaultGenomeConfig):

    def __init__(self, key):
        super().__init__(key)
        self.max_key = None
    
    def get_new_node_key(self, node_dict):
        if self.max_key is None:
            self.max_key = max(node_dict.keys()) if node_dict else 0

        self.max_key += 1
        new_id = self.max_key

        if new_id in node_dict:
            new_id = max(iterkeys(node_dict)) + 1
            self.max_key = new_id

        assert new_id not in node_dict

        return new_id


    def reset_node_indexer(self):
        self.max_key = None


class DefaultGenomeWrapper(DefaultGenome):

    def __init__(self, key):
        """init the father"""
        super().__init__(key)

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfigWrapper(param_dict)