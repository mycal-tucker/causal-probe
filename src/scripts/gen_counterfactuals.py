import os
from argparse import ArgumentParser

import h5py
import numpy as np
import torch
import yaml

from utils.training_utils import choose_task_classes, choose_dataset_class, choose_model_class, choose_probe_class


"""
Generates counterfactual embeddings using probes for each layer.
"""


# Given some config, a probe, d, and a loss function, updates the elements of the dataset to minimize the loss of the
# probe. This is the core counterfactual generation technique.
def gen_counterfactuals(args, probe, dataset, loss):
    loss_tolerance = 0.05
    probe_params_path = os.path.join(args['reporting']['root'], args['probe']['params_path'])
    probe.load_state_dict(torch.load(probe_params_path, map_location=torch.device('cuda:0')))
    probe.eval()
    original_embeddings = None
    word_embeddings = []
    updated_embeddings = []
    test_dataloader = dataset.get_test_dataloader()
    iteration_idx = -1
    for batch_idx, batch in enumerate(test_dataloader):
        iteration_idx += batch[0].shape[0]
        observation_batch, label_batch, length_batch, _ = batch
        start_embeddings = observation_batch
        true_labels = label_batch
        if args['dataset']['embeddings']['break_on_qmark']:
            q_mark_idxs = []
            for intra_batch_idx in range(batch[0].shape[0]):
                observation = test_dataloader.dataset.observations[iteration_idx]
                sentence = observation.sentence
                q_mark_idx = sentence.index('?')
                q_mark_idxs.append(q_mark_idx)
            for i, q_mark_idx in enumerate(q_mark_idxs):
                true_labels[i, :q_mark_idx + 1] = -1
                true_labels[i, :, :q_mark_idx + 1] = -1
        curr_embeddings = start_embeddings
        word_embeddings.extend([curr_embedding.clone() for curr_embedding in curr_embeddings])
        curr_embeddings = curr_embeddings.cuda()
        curr_embeddings.requires_grad = True
        my_optimizer = torch.optim.Adam([curr_embeddings], lr=0.00001)
        prediction_loss = 100  # Initialize the prediction loss as really high - it gets overwritten during updates.
        increment_idx = 0
        # We implement patience manually.
        smallest_loss = prediction_loss
        steps_since_best = 0
        patience = 5000
        while prediction_loss > loss_tolerance:
            if increment_idx >= 500000:
                print("Breaking because of increment index")
                break
            predictions = probe(curr_embeddings)
            prediction_loss, count = loss(predictions, torch.reshape(true_labels, predictions.shape), length_batch)
            prediction_loss.backward()
            my_optimizer.step()
            if prediction_loss < smallest_loss:
                steps_since_best = 0
                smallest_loss = prediction_loss
            else:
                steps_since_best += 1
                if steps_since_best == patience:
                    print("Breaking because of patience with loss", prediction_loss)
                    break
            increment_idx += 1
        print("Exited grad update loop after", increment_idx, "steps!")
        updated_embeddings.extend([curr_embedding for curr_embedding in curr_embeddings])
    return original_embeddings, word_embeddings, updated_embeddings


def execute_experiment(args):
    dataset_class = choose_dataset_class(args)
    task_class, reporter_class, loss_class = choose_task_classes(args)
    probe_class = choose_probe_class(args)
    task = task_class()
    expt_dataset = dataset_class(args, task)
    expt_probe = probe_class(args)
    expt_loss = loss_class(args)
    results_dir = yaml_args['reporting']['root']

    # Generate the counterfactuals by delegating to the right method.
    _, word_only_embeddings, updated_embeddings = gen_counterfactuals(args, expt_probe, expt_dataset, expt_loss)

    # Save the updated and original words to files.
    with torch.no_grad():
        np_word = [embedding.cpu().numpy() for embedding in word_only_embeddings]
        np_updated = [embedding.cpu().numpy() for embedding in updated_embeddings]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    hf = h5py.File(results_dir + '/original_words.hdf5', 'w')
    for i, embedding in enumerate(np_word):
        hf.create_dataset(str(i), data=embedding)
    hf = h5py.File(results_dir + '/updated_words.hdf5', 'w')
    for i, embedding in enumerate(np_updated):
        hf.create_dataset(str(i), data=embedding)


if __name__ == '__main__':
    argp = ArgumentParser()
    argp.add_argument('experiment_config')
    argp.add_argument('--results-dir', default='',
                      help='Set to reuse an old results dir; '
                           'if left empty, new directory is created')
    argp.add_argument('--report-results', default=1, type=int,
                      help='Set to report results; '
                           '(optionally after training a new probe)')
    argp.add_argument('--seed', default=0, type=int,
                      help='sets all random seeds for (within-machine) reproducibility')
    cli_args = argp.parse_args()
    if cli_args.seed:
        np.random.seed(cli_args.seed)
        torch.manual_seed(cli_args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    yaml_args = yaml.load(open(cli_args.experiment_config))

    true_reporting_root = yaml_args['reporting']['root']
    for layer_idx in range(1, 25):
        # Somewhat gross, but we override part of the config file to do a full "experiment" for each layer.
        yaml_args['model']['model_layer'] = layer_idx
        yaml_args['reporting']['root'] = true_reporting_root + str(layer_idx)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        yaml_args['device'] = device
        execute_experiment(yaml_args)
