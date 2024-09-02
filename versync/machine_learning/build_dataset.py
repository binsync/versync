# This code is taken from the following:
# https://github.com/Cisco-Talos/binary_function_similarity/tree/f0da093ed5662de51182822991ebee68c1a93cc2/Models/GGSNN-GMN/NeuralNetwork

from .graph_factory_testing import GraphFactoryTesting
from .graph_factory_inference import GraphFactoryInference
from .graph_factory_training import GraphFactoryTraining
from .graph_factory_base import GraphData

import logging
log = logging.getLogger('gnn')


def build_train_validation_generators(config):
    """Utility function to build train and validation batch generators.

    Args
      config: global configuration
    """
    training_gen = GraphFactoryTraining(
        func_path=config['training']['df_train_path'],
        feat_path=config['training']['features_train_path'],
        batch_size=config['batch_size'],
        use_features=config['data']['use_features'],
        features_type=config['features_type'],
        bb_features_size=config['bb_features_size'],
    )

    validation_gen = GraphFactoryTesting(
        pos_path=config['validation']['positive_path'],
        neg_path=config['validation']['negative_path'],
        feat_path=config['validation']['features_validation_path'],
        batch_size=config['batch_size'],
        use_features=config['data']['use_features'],
        features_type=config['features_type'],
        bb_features_size=config['bb_features_size'])

    return training_gen, validation_gen


def build_testing_generator(config, csv_path):
    """Build a batch_generator from the CSV in input.

    Args
      config: global configuration
      csv_path: CSV input path
    """
    testing_gen = GraphFactoryInference(
        func_path=csv_path,
        feat_path=config['testing']['features_testing_path'],
        batch_size=config['batch_size'],
        use_features=config['data']['use_features'],
        features_type=config['features_type'],
        bb_features_size=config['bb_features_size'])

    return testing_gen


def fill_feed_dict(placeholders, batch):
    """Create a feed dict for the given batch of data.

    Args:
      placeholders: a dict of placeholders as defined in build_model.py
      batch: a batch of data, should be either a single `GraphData`
        instance for triplet training, or a tuple of (graphs, labels)
        for pairwise training.

    Returns:
      feed_dict: a dictionary that can be used in TF run.
    """
    if isinstance(batch, GraphData):
        graphs = batch
        labels = None
    else:
        graphs, labels = batch

    feed_dict = {
        placeholders['node_features']: graphs.node_features,
        placeholders['edge_features']: graphs.edge_features,
        placeholders['from_idx']: graphs.from_idx,
        placeholders['to_idx']: graphs.to_idx,
        placeholders['graph_idx']: graphs.graph_idx,
    }

    # Set the labels only if provided in the input batch data.
    if labels is not None:
        feed_dict[placeholders['labels']] = labels
    return feed_dict
