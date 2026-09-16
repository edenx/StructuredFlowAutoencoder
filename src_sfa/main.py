import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os

# Add project root to path so shared modules (utils, models, etc.) are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import copy

from utils import dict2namespace


RUNNERS = {
    "pinwheel": ("runner_continuous", "PinWheelRunner"),
    "mnist": ("runner_continuous", "MNISTRunner"),
    "hvg": ("runner_continuous", "HVGRunner"),
    "pinwheel_mixture": ("runner_mixture", "PinWheelMixtureRunner"),
    "mnist_mixture": ("runner_mixture", "MNISTMixtureRunner"),
    "pendulum": ("runner_pendulum", "PENDULUMRunner"),
}


def _get_runner_cls(doc):
    """Lazy import: only load the module needed for the requested runner."""
    import importlib
    entry = RUNNERS.get(doc)
    if entry is None:
        raise ValueError(f"Unknown runner: {doc}. Available: {list(RUNNERS.keys())}")
    module_name, class_name = entry
    mod = importlib.import_module(module_name)
    return getattr(mod, class_name)


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--runner', type=str, required=True,
                        choices=list(RUNNERS.keys()),
                        help='Which runner to use: ' + ', '.join(RUNNERS.keys()))
    parser.add_argument('--doc', type=str, default=None,
                        help='Log folder name (defaults to --runner value)')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('--inference', action='store_true', help='Whether to conduct inference in latent space')
    parser.add_argument('--ood', action='store_true', help='Whether to conduct OOD inference in latent space')
    parser.add_argument('--predict', action='store_true', help='Whether to predict latent')
    parser.add_argument('--figure', action='store_true', help='Whether to generate figures')
    parser.add_argument('--fast_fid', action='store_true', help='Whether to do fast fid test')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='data', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")

    args = parser.parse_args()
    if args.doc is None:
        args.doc = args.runner
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    new_config = dict2namespace(config)

    # Auto-derive model.cnn (image data flag) from dataset name
    IMAGE_DATASETS = {'MNIST', 'EMNIST', 'CIFAR10', 'CIFAR100'}
    if not hasattr(new_config.model, 'cnn'):
        new_config.model.cnn = new_config.data.dataset in IMAGE_DATASETS

    eval_mode = args.test or args.sample or args.fast_fid or args.inference or args.ood or args.figure or args.predict

    if not eval_mode:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.log_path)
                    os.makedirs(args.log_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample or args.figure:
            os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Image folder already exists. Overwrite? (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)

        elif args.fast_fid:
            os.makedirs(os.path.join(args.exp, 'fid_samples'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'fid_samples', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = False
                else:
                    response = input("Image folder already exists. \n "
                                     "Type Y to delete and start from an empty folder?\n"
                                     "Type N to overwrite existing folders (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    try:
        runner_cls = _get_runner_cls(args.runner)
        runner = runner_cls(args, config)

        if args.test:
            raise NotImplementedError("Implement Test.")
        elif args.sample:
            runner.sample()
        elif args.inference:
            runner.inference()
        elif args.ood:
            runner.ood()
        elif args.predict:
            runner.predict()
        elif args.figure:
            runner.figure()
        elif args.fast_fid:
            raise NotImplementedError("Implement FID.")
        else:
            runner.train()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
