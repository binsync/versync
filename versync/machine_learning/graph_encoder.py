# This code is taken from the following:
# https://github.com/Cisco-Talos/binary_function_similarity/tree/f0da093ed5662de51182822991ebee68c1a93cc2/Models/GGSNN-GMN/NeuralNetwork

import sonnet as snt


class GraphEncoder(snt.AbstractModule):
    """Encoder module that projects node and edge features to some embeddings."""

    def __init__(self,
                 node_hidden_sizes=None,
                 edge_hidden_sizes=None,
                 name='graph-encoder'):
        """Constructor.

        Args:
          node_hidden_sizes: if provided should be a list of ints, hidden sizes of
            node encoder network, the last element is the size of the node outputs.
            If not provided, node features will pass through as is.
          edge_hidden_sizes: if provided should be a list of ints, hidden sizes of
            edge encoder network, the last element is the size of the edge outptus.
            If not provided, edge features will pass through as is.
          name: name of this module.
        """
        super(GraphEncoder, self).__init__(name=name)

        # this also handles the case of an empty list
        self._node_hidden_sizes = node_hidden_sizes if node_hidden_sizes else None
        self._edge_hidden_sizes = edge_hidden_sizes

    def _build(self, node_features, edge_features=None):
        """Encode node and edge features.

        Args:
          node_features: [n_nodes, node_feat_dim] float tensor.
          edge_features: if provided, should be [n_edges, edge_feat_dim] float
            tensor.

        Returns:
          node_outputs: [n_nodes, node_embedding_dim] float tensor, node embeddings.
          edge_outputs: if edge_features is not None and edge_hidden_sizes is not
            None, this is [n_edges, edge_embedding_dim] float tensor, edge
            embeddings; otherwise just the input edge_features.
        """
        if self._node_hidden_sizes is None:
            node_outputs = node_features
        else:
            node_outputs = snt.nets.MLP(
                self._node_hidden_sizes,
                name='node-feature-mlp')(node_features)

        if edge_features is None or self._edge_hidden_sizes is None:
            edge_outputs = edge_features
        else:
            edge_outputs = snt.nets.MLP(
                self._edge_hidden_sizes,
                name='edge-feature-mlp')(edge_features)

        return node_outputs, edge_outputs
