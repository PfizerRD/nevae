from utils import *
#create_dir, pickle_save, print_vars, load_data, get_shape, proxy
from config import SAVE_DIR, VAEGConfig
from datetime import datetime
from cell import VAEGCell
from model import VAEG

import tensorflow as tf
import numpy as np
import logging
import pickle
import os
import argparse

logging.basicConfig(
    format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FLAGS = None
placeholders = {
    'dropout': tf.placeholder_with_default(0., shape=()),
    'lr': tf.placeholder_with_default(0., shape=()),
    'decay': tf.placeholder_with_default(0., shape=())
}


def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_epochs", type=int,
                        default=32, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float,
                        default=0.00005, help="learning rate")
    parser.add_argument("--decay_rate", type=float,
                        default=1.0, help="decay rate")
    parser.add_argument("--dropout_rate", type=float,
                        default=0.00005, help="dropout rate")
    parser.add_argument("--log_every", type=int, default=5,
                        help="write the log in how many iterations")
    parser.add_argument("--sample_file", type=str, default=None,
                        help="directory to store the sample graphs")

    parser.add_argument("--random_walk", type=int,
                        default=5, help="random walk depth")
    parser.add_argument("--z_dim", type=int, default=5, help="z_dim")
    # parser.add_argument("--nodes", type=int, default=30, help="num_of_nodes")
    parser.add_argument("--num_batches", type=int, default=100, help="number of batches")
    parser.add_argument("--bin_dim", type=int, default=3, help="bin_dim")

    parser.add_argument("--graph_file", type=str, default=None,
                        help="The dictory where the training graph structure is saved")
    parser.add_argument("--z_dir", type=str, default=None,
                        help="The z values will be stored file to be stored")
    parser.add_argument("--sample", type=bool, default=False,
                        help="True if you want to sample")

    parser.add_argument("--mask_weight", type=bool,
                        default=False, help="True if you want to mask weight")

    parser.add_argument("--out_dir", type=str, default=None,
                        help="Store log/model files.")


def create_hparams(flags, data_dir):
    """Create training hparams."""
    return tf.contrib.training.HParams(
        # Data
        graph_file=data_dir,
        out_dir=flags.out_dir,
        z_dir=flags.z_dir,
        sample_file=flags.sample_file,
        z_dim=flags.z_dim,

        # training
        learning_rate=flags.learning_rate,
        decay_rate=flags.decay_rate,
        dropout_rate=flags.dropout_rate,
        num_epochs=flags.num_epochs,
        random_walk=flags.random_walk,
        log_every=flags.log_every,
        nodes=node_num,
        num_batches=flags.num_batches,
        bin_dim=flags.bin_dim,
        mask_weight=flags.mask_weight,
        # sample
        sample=flags.sample
    )


if __name__ == '__main__':
    nmt_parser = argparse.ArgumentParser()
    add_arguments(nmt_parser)
    FLAGS, unparsed = nmt_parser.parse_known_args()

    for batch in range(FLAGS.num_batches):
        for node_num in range(30, 56):
            data_dir = os.path.join(FLAGS.graph_file, 'n_{}'.format(node_num))
            hparams = create_hparams(FLAGS, data_dir)

            # loading the data from a file
            adj, weight, weight_bin, features, edges, hde = load_data(data_dir, node_num, hparams.bin_dim)
            num_nodes = adj[0].shape[0]
            num_features = features[0].shape[1]
            print("Num nodes: {}".format(num_nodes))
            print("Num features: {}".format(num_features))
            print("Size adj: {}".format(len(adj)))
            e = max([len(edge) for edge in edges])
                
            log_fact_k = log_fact(e)

            # Training
            model = VAEG(hparams, placeholders, num_nodes, num_features, edges, log_fact_k, hde)
            model.restore(hparams.out_dir)
            model.train(placeholders, hparams, adj, weight, weight_bin, features)
            tf.get_variable_scope().reuse_variables()