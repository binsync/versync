import argparse
import os
import logging

from versync.machine_learning import model_test, model_train, model_validate, get_config

_l = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        prog='gnn',
        description='GGSNN and GMN models',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-d', '--debug', action='store_true',
                        help='Log level debug')

    group0 = parser.add_mutually_exclusive_group(required=True)
    group0.add_argument('--train', action='store_true',
                        help='Train the model')
    group0.add_argument('--validate', action='store_true',
                        help='Run model validation')
    group0.add_argument('--test', action='store_true',
                        help='Run model testing')

    parser.add_argument("--featuresdir",
                        default="/preprocessing",
                        help="Path to the Preprocessing dir")

    parser.add_argument("--features_type", required=True,
                        choices=["nofeatures",
                                 "opc"],
                        help="Select the type of BB features")

    parser.add_argument("--model_type", required=True,
                        choices=["embedding", "matching"],
                        help="Select the type of network")

    parser.add_argument("--training_mode", required=True,
                        choices=["pair", "triplet"],
                        help="Select the type of network")

    parser.add_argument('--num_epochs', type=int,
                        required=False, default=2,
                        help='Number of training epochs')

    parser.add_argument('--restore',
                        action='store_true', default=False,
                        help='Continue the training from the last checkpoint')

    parser.add_argument('--dataset', required=True,
                        choices=['one', 'two', 'vuln'],
                        help='Choose the dataset to use for the train or test')

    parser.add_argument('-c', '--checkpointdir', required=True,
                        help='Input/output for model checkpoint')

    parser.add_argument('-o', '--outputdir', required=True,
                        help='Output dir')

    args = parser.parse_args()

    # Create the output directory
    if args.outputdir:
        if not os.path.isdir(args.outputdir):
            os.mkdir(args.outputdir)
            print("Created outputdir: {}".format(args.outputdir))

    if args.featuresdir:
        if not os.path.isdir(args.featuresdir):
            print("[!] Non existing directory: {}".format(args.featuresdir))
            return

    if args.checkpointdir:
        if not os.path.isdir(args.checkpointdir):
            os.mkdir(args.checkpointdir)
            print("Created checkpointdir: {}".format(args.checkpointdir))

    # Load the model configuration and save to file
    # TODO: change how this config is loaded
    config = get_config(args)

    if args.train:
        _l.info("Running model training")
        model_train(config, restore=args.restore)

    if args.validate:
        _l.info("Running model validation")
        model_validate(config)

    if args.test:
        _l.info("Running model testing")
        model_test(config)


if __name__ == '__main__':
    main()
