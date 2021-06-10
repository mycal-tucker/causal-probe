import ast

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from matplotlib.ticker import FormatStrFormatter
import matplotlib

from tabulate import tabulate


text_dir = 'data/qa_example/'
counterfactual_dir = 'counterfactuals/qa_example/model_dist_1layer/'
probe_type = 'model_dist'
test_layers = [i for i in range(1, 25)]
layer_offset = test_layers[0]


# For a single sentence, plot the distribution over start probabilities for the original and updated embeddings
# as well as just the deltas.
def plot_sentence_probs(sentence_idx):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
    fig.suptitle("Absolute and Relative Start Token Probabilities")
    x_axis = [i + 1 for i in range(len(original_start_probs[sentence_idx]))]
    # Plot the absolute probability values
    ax1.set_title("Start probabilities for layer " + str(layer) + " and sentence " + str(sentence_idx))
    ax1.set_xlabel('Token idx')

    ax1.errorbar(x_axis, original_start_probs[sentence_idx], linestyle='--', color='green', marker='s', label='Original')
    ax1.errorbar(x_axis, nn1_parse_updated_start_probs[sentence_idx], color='red', marker='s', label='Conj. Parse')
    ax1.errorbar(x_axis, nn2_parse_updated_start_probs[sentence_idx], color='blue', marker='s', label='NN2 Parse')
    ax1.legend(loc="upper left")

    ax2.set_title("Changes in start probabilities for layer " + str(layer) + " and sentence " + str(sentence_idx))
    ax2.set_xlabel('Token idx')
    nn1_delta = [nn1_parse_updated_start_probs[sentence_idx][i] - original_start_probs[sentence_idx][i] for i in range(len(original_start_probs[sentence_idx]))]
    nn2_delta = [nn2_parse_updated_start_probs[sentence_idx][i] - original_start_probs[sentence_idx][i] for i in range(len(original_start_probs[sentence_idx]))]
    ax2.errorbar(x_axis, nn1_delta, color='red', marker='s', label='Conj. Parse')
    ax2.errorbar(x_axis, nn2_delta, color='blue', marker='s', label='NN2 Parse')
    ax2.legend(loc='upper left')
    plt.show()


# Read in the other question info as well
corpus_types = []
answer_lengths = []
start_likelihoods = []
contexts = []
questions = []
answers = []
with open(text_dir + 'setup.txt', 'r') as setup_file:
    for line_idx, line in enumerate(setup_file):
        split_line = line.split('\t')
        corpus_types.append(split_line[0])
        answer_lengths.append(int(split_line[1]))
        start_likelihoods.append(float(split_line[2]))
        contexts.append(split_line[3])
        questions.append(split_line[4])
        answers.append(split_line[5])


# Read in the token id data. We care about probability changes at specific locations, which were stored way back
# when the corpus was generated in token_idxs.txt.
# We care about 4 locations (determiner and noun) x (location 1 and location 2)
det1_token_idxs = []
nn1_token_idxs = []
det2_token_idxs = []
nn2_token_idxs = []
all_token_idxs = [det1_token_idxs, nn1_token_idxs, det2_token_idxs, nn2_token_idxs]
with open(text_dir + 'token_idxs.txt', 'r') as token_file:
    for line_idx, line in enumerate(token_file):
        if line_idx % 2 == 0:
            continue  # Have twice as many token lines as needed because the sentence were duplicated.
        split_line = line.split('\t')
        det1_token_idxs.append(int(split_line[0]))
        nn1_token_idxs.append(int(split_line[1]))
        det2_token_idxs.append(int(split_line[2]))
        nn2_token_idxs.append(int(split_line[3]))

all_layers_original_starts = []
all_layers_nn1_parse_starts = []
all_layers_nn2_parse_starts = []
for layer in test_layers:
    # Read in how the probabilities got updated.
    original_start_probs = []
    nn1_parse_updated_start_probs = []
    nn2_parse_updated_start_probs = []
    with open(counterfactual_dir + probe_type + str(layer) + '/updated_probs.txt', 'r') as results_file:
        for line_idx, line in enumerate(results_file):
            split_line = line.split('\t')
            if line_idx == 0:  # The first line has some cruft based on how files are generated.
                continue
            if line_idx % 2 == 1:
                original_start_probs.append([ast.literal_eval(data)[0] for data in split_line])
                nn1_parse_updated_start_probs.append([ast.literal_eval(data)[2] for data in split_line])
            else:
                nn2_parse_updated_start_probs.append([ast.literal_eval(data)[2] for data in split_line])
    # Now we have the data, so if you want to plot probabilities for a single sentence, you can.
    # Plot stuff for just a single sentence.
    # for i in range(1):
    #     plot_sentence_probs(i)
    # Dump the layer-specific data into an aggregator.
    all_layers_original_starts.append(original_start_probs)
    all_layers_nn1_parse_starts.append(nn1_parse_updated_start_probs)
    all_layers_nn2_parse_starts.append(nn2_parse_updated_start_probs)


def get_token_idx_start_update(token_idxs):
    nn1_updates = []
    nn1_updates_std = []
    nn2_updates = []
    nn2_updates_std = []
    nn1_all = []
    nn2_all = []
    for layer in test_layers:
        layer_specific_nn1_updates = []
        layer_specific_nn2_updates = []
        for sentence_idx, token_idx in enumerate(token_idxs):
            if token_idx == -1:
                print("Invalid token, skipping")
                layer_specific_nn1_updates.append(0)
                layer_specific_nn2_updates.append(0)
                continue
            original_prob = all_layers_original_starts[layer - layer_offset][sentence_idx][token_idx]
            nn1_parse_prob = all_layers_nn1_parse_starts[layer - layer_offset][sentence_idx][token_idx]
            nn2_parse_prob = all_layers_nn2_parse_starts[layer - layer_offset][sentence_idx][token_idx]
            layer_specific_nn1_updates.append(nn1_parse_prob - original_prob)
            layer_specific_nn2_updates.append(nn2_parse_prob - original_prob)
        nn1_updates.append(np.mean(layer_specific_nn1_updates))
        nn1_updates_std.append(np.std(layer_specific_nn1_updates))
        nn2_updates.append(np.mean(layer_specific_nn2_updates))
        nn2_updates_std.append(np.std(layer_specific_nn2_updates))
        nn1_all.append(layer_specific_nn1_updates)
        nn2_all.append(layer_specific_nn2_updates)
    return nn1_updates, nn1_updates_std, nn2_updates, nn2_updates_std, nn1_all, nn2_all


def plot_start_updates():
    x_axis = [i for i in test_layers]
    fig, axes = plt.subplots(nrows=4, figsize=(10, 20))
    for i in range(4):
        tokens = all_token_idxs[i]
        _, _, _, _, nn1_all, nn2_all =\
            get_token_idx_start_update(tokens)
        # Now do the plotting
        ax = axes[i]
        ax.set_title("Start prob deltas for token " + str(i))
        ax.set_xlabel('Layer idx')
        ax.errorbar(x_axis, np.mean(nn1_all, axis=1), color='red', marker='s', label='NP1 Parse')
        ax.errorbar(x_axis, np.mean(nn2_all, axis=1), color='blue', marker='s', label='NP2 Parse')
        ax.axhline()
        ax.legend(loc='upper left')
    plt.savefig(counterfactual_dir + probe_type + '_token_updates.png')
    plt.show()


# Plot aggregate data.
plot_start_updates()

_, _, _, _, p1_tok0, p2_tok0 = get_token_idx_start_update(all_token_idxs[0])
_, _, _, _, p1_tok1, p2_tok1 = get_token_idx_start_update(all_token_idxs[1])
_, _, _, _, p1_tok2, p2_tok2 = get_token_idx_start_update(all_token_idxs[2])
_, _, _, _, p1_tok3, p2_tok3 = get_token_idx_start_update(all_token_idxs[3])


def calculate_stats(p1_tokens, p2_tokens, string_label):
    p1 = np.asarray(p1_tokens[0])
    for p1_idx, p1_tokens_entry in enumerate(p1_tokens):
        if p1_idx == 0:
            continue
        p1 = p1 + np.asarray(p1_tokens_entry)
    p2 = np.asarray(p2_tokens[0])
    for p2_idx, p2_tokens_entry in enumerate(p2_tokens):
        if p2_idx == 0:
            continue
        p2 = p2 + np.asarray(p2_tokens_entry)
    for layer in range(p1.shape[0]):
        stat, p = wilcoxon(p1[layer], p2[layer], alternative='greater')
        _, less_p = wilcoxon(p1[layer], p2[layer], alternative='less')
        if p < 0.01:
            print("Sig. greater:\t", string_label, "for layer", layer + layer_offset)
            continue
        if less_p < 0.01:
            print("Sig. less:\t", string_label, "for layer", layer + layer_offset)
            continue
        print("Not significant for layer", layer + layer_offset)
    print()


calculate_stats((p1_tok0, p1_tok1), (p2_tok0, p2_tok1), "NP1")
calculate_stats((p1_tok2, p1_tok3), (p2_tok2, p2_tok3), "NP2")
parse1_np1_delta = np.asarray(p1_tok0) + np.asarray(p1_tok1)
parse1_np2_delta = (np.asarray(p1_tok2) + np.asarray(p1_tok3))
parse2_np1_delta = np.asarray(p2_tok0) + np.asarray(p2_tok1)
parse2_np2_delta = (np.asarray(p2_tok2) + np.asarray(p2_tok3))
calculate_stats((parse1_np1_delta, -1 * parse1_np2_delta),
                (parse2_np1_delta, -1 * parse2_np2_delta),
                "Overall shift")
calculate_stats((p1_tok0, p1_tok1), (np.zeros_like(p2_tok2), np.zeros_like(p2_tok3)), "Parse1 NP1 vs. 0")
calculate_stats((p1_tok2, p1_tok3), (np.zeros_like(p2_tok2), np.zeros_like(p2_tok3)), "Parse1 NP2 vs. 0")
calculate_stats((p2_tok2, p2_tok3), (np.zeros_like(p2_tok2), np.zeros_like(p2_tok3)), "Parse2 NP2 vs. 0")
calculate_stats((p2_tok0, p2_tok1), (np.zeros_like(p2_tok2), np.zeros_like(p2_tok3)), "Parse2 NP1 vs. 0")


# Plot the net shift in probability mass between tokens from the different noun phrases, plotting one line per
# parse.
def plot_aggregate_delta():
    p1_net = parse1_np1_delta - parse1_np2_delta
    p2_net = parse2_np1_delta - parse2_np2_delta
    x_axis = [i for i in test_layers]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(x_axis, np.mean(p1_net, axis=1), color='red', marker='s', label='Parse 1')
    ax.errorbar(x_axis, np.mean(p2_net, axis=1), color='blue', marker='s', label='Parse 2')
    fig.tight_layout()
    ax.legend(loc='upper left')
    ax.axhline()
    ax.set_xlabel('Layer idx')
    plt.show()


plot_aggregate_delta()
p1_all = np.asarray(all_layers_nn1_parse_starts)
p2_all = np.asarray(all_layers_nn2_parse_starts)
original_all = np.asarray(all_layers_original_starts)


# Similar logic to plot_aggregate_delta, but plotted on an absolute scale instead of just the deltas. And we plot
# only the probabilities words in np1
def net_probabilities():
    matplotlib.rcParams.update({'font.size': 9})
    x_axis = test_layers
    tokens0 = np.asarray(all_token_idxs[0])
    tokens1 = np.asarray(all_token_idxs[1])

    p1_np1_mean = np.mean(p1_all[:, :, tokens0] + p1_all[:, :, tokens1], axis=(1, 2))
    p2_np1_mean = np.mean(p2_all[:, :, tokens0] + p2_all[:, :, tokens1], axis=(1, 2))
    original_mean = np.mean(original_all[:, :, tokens0] + original_all[:, :, tokens1], axis=(1, 2))

    fig, ax1 = plt.subplots(nrows=1, figsize=(10, 2.1))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.errorbar(x_axis, p1_np1_mean, color='red', linestyle='--', label='Parse 1')
    ax1.errorbar(x_axis, original_mean, color='green', label='Original')
    ax1.errorbar(x_axis, p2_np1_mean, color='blue', label='Parse 2')
    ax1.legend(loc="upper left")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Prob. NP1 Start")
    fig.suptitle("Likelihood of NP1 Start by Layer")
    plt.xlim(1, 24)
    fig.tight_layout()
    plt.show()


net_probabilities()
