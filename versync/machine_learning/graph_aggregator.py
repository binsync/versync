# This code is taken from the following:
# https://github.com/Cisco-Talos/binary_function_similarity/tree/f0da093ed5662de51182822991ebee68c1a93cc2/Models/GGSNN-GMN/NeuralNetwork

import sonnet as snt
import tensorflow as tf

from .graph_prop_layer import GraphPropLayer


AGGREGATION_TYPE = {
    'sum': tf.math.unsorted_segment_sum,
    'mean': tf.math.unsorted_segment_mean,
    'sqrt_n': tf.math.unsorted_segment_sqrt_n,
    'max': tf.math.unsorted_segment_max,
}


class GraphAggregator(snt.AbstractModule):
    """This module computes graph representations by aggregating from parts."""

    def __init__(self,
                 node_hidden_sizes,
                 graph_transform_sizes=None,
                 gated=True,
                 aggregation_type='sum',
                 name='graph-aggregator'):
        """Constructor.

        Args:
          node_hidden_sizes: the hidden layer sizes of the node transformation nets.
            The last element is the size of the aggregated graph representation.
          graph_transform_sizes: sizes of the transformation layers on top of the
            graph representations.  The last element of this list is the final
            dimensionality of the output graph representations.
          gated: set to True to do gated aggregation, False not to.
          aggregation_type: one of {sum, max, mean, sqrt_n}.
          name: name of this module.
        """
        super(GraphAggregator, self).__init__(name=name)

        self._node_hidden_sizes = node_hidden_sizes
        self._graph_transform_sizes = graph_transform_sizes
        self._graph_state_dim = node_hidden_sizes[-1]
        self._gated = gated
        self._aggregation_type = aggregation_type
        self._aggregation_op = AGGREGATION_TYPE[aggregation_type]

    def _build(self, node_states, graph_idx, n_graphs):
        """Compute aggregated graph representations.

        Args:
          node_states: [n_nodes, node_state_dim] float tensor, node states of a
            batch of graphs concatenated together along the first dimension.
          graph_idx: [n_nodes] int tensor, graph ID for each node.
          n_graphs: integer, number of graphs in this batch.

        Returns:
          graph_states: [n_graphs, graph_state_dim] float tensor, graph
            representations, one row for each graph.
        """
        node_hidden_sizes = self._node_hidden_sizes
        if self._gated:
            node_hidden_sizes[-1] = self._graph_state_dim * 2

        node_states_g = snt.nets.MLP(
            node_hidden_sizes, name='node-state-g-mlp')(node_states)

        if self._gated:
            gates = tf.nn.sigmoid(node_states_g[:, :self._graph_state_dim])
            node_states_g = node_states_g[:, self._graph_state_dim:] * gates

        graph_states = self._aggregation_op(node_states_g, graph_idx, n_graphs)

        # unsorted_segment_max does not handle empty graphs in the way we want
        # it assigns the lowest possible float to empty segments, we want to reset
        # them to zero.
        if self._aggregation_type == 'max':
            # reset everything that's smaller than -1e5 to 0.
            graph_states *= tf.cast(graph_states > -1e5, tf.float32)

        # transform the reduced graph states further

        # pylint: disable=g-explicit-length-test
        if (self._graph_transform_sizes is not None and
                len(self._graph_transform_sizes) > 0):
            graph_states = snt.nets.MLP(
                self._graph_transform_sizes,
                name='graph-transform-mlp')(graph_states)

        return graph_states
