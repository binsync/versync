# This code is taken from the following:
# https://github.com/Cisco-Talos/binary_function_similarity/tree/f0da093ed5662de51182822991ebee68c1a93cc2/Models/GGSNN-GMN/NeuralNetwork

import collections

GraphData = collections.namedtuple('GraphData', [
    'from_idx',
    'to_idx',
    'node_features',
    'edge_features',
    'graph_idx',
    'n_graphs'])


class GraphFactoryBase(object):
    """Base class for all the graph similarity learning datasets.

    This class defines some common interfaces a graph similarity dataset can have,
    in particular the functions that creates iterators over pairs and triplets.
    """

    def triplets(self):
        """Create an iterator over triplets.

        Note:
          batch_size: int, number of triplets in a batch.

        Yields:
          graphs: a `GraphData` instance.  The batch of triplets put together.  Each
            triplet has 3 graphs (x, y, z).  Here the first graph is duplicated once
            so the graphs for each triplet are ordered as (x, y, x, z) in the batch.
            The batch contains `batch_size` number of triplets, hence `4*batch_size`
            many graphs.
        """
        pass

    def pairs(self):
        """Create an iterator over pairs.

        Note:
          batch_size: int, number of pairs in a batch.

        Yields:
          graphs: a `GraphData` instance.  The batch of pairs put together.  Each
            pair has 2 graphs (x, y).  The batch contains `batch_size` number of
            pairs, hence `2*batch_size` many graphs.
          labels: [batch_size] int labels for each pair, +1 for similar, -1 for not.
        """
        pass
