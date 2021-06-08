import ast
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from matplotlib.ticker import FormatStrFormatter

cloze_task = 'cloze'

text_dir = 'data/example/'
counterfactual_dir = 'counterfactuals/example/model_dist_1layer/'
probe_type = 'model_dist'

POS1 = 'Plural'
POS2 = 'Singular'
parse1_label = 'Plural'
parse2_label = 'Singular'
# You need to define which words belong to which part of speech
parts_of_speech1 = ['were', 'are', 'as']
parts_of_speech2 = ['was', 'is']

test_layers = [i for i in range(1, 25)]


word_to_type = {}
for word1 in parts_of_speech1:
    word_to_type[word1] = POS1
for word2 in parts_of_speech2:
    word_to_type[word2] = POS2

layer_offset = test_layers[0]  # Have to handle the zero vs. one-indexing of layers.
candidates = None


# For a single sentence and layer id, plot the distribution over possible values for the masked words.
def plot_sentence_probs(sentence_id):
    print("sentence id", sentence_id)
    fig, ax1 = plt.subplots(nrows=1, figsize=(10, 2.5))
    # Create a single bar chart that shows how updates changed dist over word options.
    # Each x axis tick is a new word
    x_axis = [i for i in range(len(original_probs[sentence_id]))]
    # Easier to read if the candidates are separated.
    sorted_candidates = sorted(parts_of_speech1) + sorted(parts_of_speech2)
    permutation = []
    for sort_candidate in sorted_candidates:
        unsorted_idx = candidates.index(sort_candidate)
        permutation.append(unsorted_idx)
    # Look up the sentence within the corpus if you want to read the raw text.
    ax1.set_title('Masked word prediction for sentence #' + str(sentence_id))
    ax1.set_xlabel('Candidate Word')
    ax1.set_xticks(x_axis)
    ax1.set_ylabel("Prob. Candidate")
    ax1.set_xticklabels(sorted_candidates)
    ax1.bar(x_axis, np.asarray(original_probs[sentence_id])[permutation], width=0.25, color='green', label='Original')
    ax1.bar(np.asarray(x_axis) - 0.25, np.asarray(p1_updated_probs[sentence_id])[permutation], hatch='//', width=0.25,
            color='red', label=parse1_label + ' Parse')
    ax1.bar(np.asarray(x_axis) + 0.25, np.asarray(p2_updated_probs[sentence_id])[permutation], hatch='\\\\',
            width=0.25, color='blue', label=parse2_label + ' Parse')
    for tick in ax1.get_xticklabels():
        tick.set_rotation(25)
    ax1.legend(loc="upper left")
    fig.tight_layout()
    plt.show()


# Read in the data about original and updated_probabilities
all_layers_originals = []
all_layers_parse1 = []
all_layers_parse2 = []
for layer in test_layers:
    # Read in how the probabilities got updated.
    original_probs = []
    p1_updated_probs = []
    p2_updated_probs = []
    with open(counterfactual_dir + probe_type + str(layer) + '/updated_probs.txt', 'r') as results_file:
        for line_idx, line in enumerate(results_file):
            split_line = line.split('\t')
            if line_idx == 0 and split_line[0] == 'Candidates':
                new_candidates = split_line[1:-1]
                candidates = [cand.strip() for cand in new_candidates]
                continue
            # First half is the original probability, second half is updated
            updated = split_line[int(len(split_line) / 2):]
            if line_idx % 2 == 1:  # Off by 1 because of candidates thing!
                original = split_line[:int(len(split_line) / 2)]
                original_probs.append([ast.literal_eval(data) for data in original])
                p1_updated_probs.append([ast.literal_eval(data) for data in updated])
            else:
                p2_updated_probs.append([ast.literal_eval(data) for data in updated])
    # Now we have the data, so if you want to plot probabilities for a single sentence, you can by uncommenting below.
    # for i in range(2):
    #     plot_sentence_probs(i)
    # Dump the layer-specific data into an aggregator.
    all_layers_originals.append(original_probs)
    all_layers_parse1.append(p1_updated_probs)
    all_layers_parse2.append(p2_updated_probs)

# Now that candidates is initialized, group by part of speech.
pos1_idxs = []
pos2_idxs = []
for candidate_idx, candidate in enumerate(candidates):
    tag = word_to_type.get(candidate)
    if tag is None:
        assert False, "Need to tag " + candidate + " in word_to_type"
    if tag == POS1:
        pos1_idxs.append(candidate_idx)
    elif tag == POS2:
        pos2_idxs.append(candidate_idx)
    else:
        assert False, "Bad tag for candidate " + candidate + " in word_to_type"


# For the specified layer, aggregate the shirts of probability mass across all sentences for each word. This
# allows us to see if some layers update specific words more or less.
def plot_layer_updates(specific_layer, normalize=False):
    x_axis = np.asarray([i for i in range(len(candidates))])
    fig, ax = plt.subplots(nrows=1, figsize=(10, 5))
    original_mean = np.mean(all_layers_originals[specific_layer - layer_offset], axis=0)
    parse1_mean = np.mean(all_layers_parse1[specific_layer - layer_offset], axis=0)
    parse2_mean = np.mean(all_layers_parse2[specific_layer - layer_offset], axis=0)
    p1_diff = parse1_mean - original_mean
    p2_diff = parse2_mean - original_mean
    if normalize:
        p1_diff = p1_diff / original_mean
        p2_diff = p2_diff / original_mean
    ax.bar(x_axis - 0.1, p1_diff, color='red', width=0.2, label=parse1_label + ' Parse')
    ax.bar(x_axis + 0.1, p2_diff, color='blue', width=0.2, label=parse2_label + ' Parse')
    ax.set_xticks(x_axis)
    ax.set_xticklabels(candidates)
    ax.legend(loc='upper left')
    ax.set_title("Mean word prob updates for layer " + str(specific_layer))
    ax.axhline()
    plt.show()


# Plot aggregated sentence data, still for just a single layer.
for layer_id in test_layers:
    plot_layer_updates(layer_id, normalize=False)


# Now we aggregate words by part of speech, and plot how probabilities change for all layers. We no longer have
# insight into specific words, but we can see if groups of words experience a net shift.
def plot_full_aggregate(normalize=False):
    x_axis = test_layers
    # Create a new plot for each word.
    # Within each plot, for each layer, calculate mean shifts.
    # Also store results from all words for total aggregation.
    p1_all = []
    p2_all = []
    num_words = len(candidates)
    fig, axes = plt.subplots(nrows=num_words, figsize=(10, 5 * num_words))
    for word_idx, word in enumerate(candidates):
        cross_layer_p1 = []
        cross_layer_p2 = []
        cross_layer_p1_std = []
        cross_layer_p2_std = []
        for layer_id in test_layers:
            original_mean = np.mean(all_layers_originals[layer_id - layer_offset], axis=0)[word_idx]
            parse1_mean = np.mean(all_layers_parse1[layer_id - layer_offset], axis=0)[word_idx]
            parse2_mean = np.mean(all_layers_parse2[layer_id - layer_offset], axis=0)[word_idx]
            cross_layer_p1.append(parse1_mean - original_mean)
            cross_layer_p2.append(parse2_mean - original_mean)
            if normalize:
                cross_layer_p1[-1] = cross_layer_p1[-1] / original_mean
                cross_layer_p2[-1] = cross_layer_p2[-1] / original_mean

            p1_diff = np.asarray(all_layers_parse1[layer_id - layer_offset]) - np.asarray(all_layers_originals[layer_id - layer_offset])
            p1_std = np.std(p1_diff, axis=0)[word_idx]
            p2_diff = np.asarray(all_layers_parse1[layer_id - layer_offset]) - np.asarray(all_layers_originals[layer_id - layer_offset])
            p2_std = np.std(p2_diff, axis=0)[word_idx]
            cross_layer_p1_std.append(p1_std)
            cross_layer_p2_std.append(p2_std)
        ax = axes[word_idx]
        ax.errorbar(x_axis, cross_layer_p1, yerr=cross_layer_p1_std, color='red', label=parse1_label + ' Parse')
        ax.errorbar(x_axis, cross_layer_p2, yerr=cross_layer_p2_std, color='blue', label=parse2_label + ' Parse')
        ax.set_title('Candidate prob updates for word "' + word + '" across layers.')
        ax.legend(loc="upper left")
        p1_all.append(cross_layer_p1)
        p2_all.append(cross_layer_p2)
    plt.show()

    # Now create a single plot that aggregates the changes by part of speech for all layers.
    numpy_p1 = np.asarray(p1_all)
    pos1_p1 = numpy_p1[pos1_idxs]
    pos1_p1_median = np.median(pos1_p1, axis=0)
    numpy_parse2 = np.asarray(p2_all)
    pos1_p2 = numpy_parse2[pos1_idxs]
    pos1_p2_median = np.median(pos1_p2, axis=0)
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 10))
    ax1.set_title(POS1 + " updates")
    ax1.errorbar(x_axis, pos1_p1_median, color='red', label=parse1_label + ' Parse')
    ax1.errorbar(x_axis, pos1_p2_median, color='blue', label=parse2_label + ' Parse')
    ax1.legend(loc="upper left")

    pos2_p1 = numpy_p1[pos2_idxs]
    pos2_p1_median = np.median(pos2_p1, axis=0)
    pos2_p2 = numpy_parse2[pos2_idxs]
    pos2_p2_median = np.median(pos2_p2, axis=0)
    ax2.set_title(POS2 + " updates")
    ax2.errorbar(x_axis, pos2_p1_median, color='red', label=parse1_label + ' Parse')
    ax2.errorbar(x_axis, pos2_p2_median, color='blue', label=parse2_label + ' Parse')
    ax2.legend(loc="upper left")
    plt.show()

    return pos1_p1, pos1_p2, pos2_p1, pos2_p2


plot_full_aggregate()


# There's a bit of ugly reshuffling/grouping of the data to get it sliced up by different parts of speeches and by
# different parses.
original_pos1s = np.asarray(all_layers_originals)[:, :, pos1_idxs]
original_pos2s = np.asarray(all_layers_originals)[:, :, pos2_idxs]
p1_pos1s = np.asarray(all_layers_parse1)[:, :, pos1_idxs]
p2_pos1s = np.asarray(all_layers_parse2)[:, :, pos1_idxs]
p1_pos2s = np.asarray(all_layers_parse1)[:, :, pos2_idxs]
p2_pos2s = np.asarray(all_layers_parse2)[:, :, pos2_idxs]
p1_pos1_delta = p1_pos1s - original_pos1s
p2_pos1_delta = p2_pos1s - original_pos1s
p1_pos2_delta = p1_pos2s - original_pos2s
p2_pos2_delta = p2_pos2s - original_pos2s

p1_pos1_change_by_sentence = np.sum(p1_pos1_delta, axis=2)
p2_pos1_change_by_sentence = np.sum(p2_pos1_delta, axis=2)
p1_pos2_change_by_sentence = np.sum(p1_pos2_delta, axis=2)
p2_pos2_change_by_sentence = np.sum(p2_pos2_delta, axis=2)


# We can plot the net shift between the two types of parts of speech, for all of the parses, and plot by layer.
def net_shift():
    matplotlib.rcParams.update({'font.size': 12})  # 10 for big images, 12 for small
    x_axis = test_layers
    p1_pos1_median_change = np.median(p1_pos1_change_by_sentence, axis=1)
    p2_pos1_median_change = np.median(p2_pos1_change_by_sentence, axis=1)

    fig, ax1 = plt.subplots(nrows=1, figsize=(10, 2.5))
    ax1.errorbar(x_axis, p1_pos1_median_change, color='red', linestyle='--', label=parse1_label + ' parse')
    ax1.errorbar(x_axis, p2_pos1_median_change, color='blue', label=parse2_label + ' parse')
    ax1.legend(loc="upper left")
    ax1.set_xlabel("Layer idx")
    ax1.axhline()
    fig.suptitle("Mean change in " + parse1_label + " likelihood by layer")
    fig.tight_layout()
    plt.show()


net_shift()


# Plot the original and updated agregated probabilities by layer for each parse and part of speech. These are the
# types of plots included in the main paper.
def net_probabilities():
    matplotlib.rcParams.update({'font.size': 9})
    x_axis = test_layers
    p1_pos1_mean = np.mean(np.sum(p1_pos1s, axis=2), axis=1)
    p2_pos1_mean = np.mean(np.sum(p2_pos1s, axis=2), axis=1)
    original_pos1s_mean = np.mean(np.sum(original_pos1s, axis=2), axis=1)

    fig, ax1 = plt.subplots(nrows=1, figsize=(10, 2.1))
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.errorbar(x_axis, p1_pos1_mean, color='red', linestyle='--', label=parse1_label + ' parse')
    ax1.errorbar(x_axis, original_pos1s_mean, color='green', label='Original')
    ax1.errorbar(x_axis, p2_pos1_mean, color='blue', label=parse2_label + ' parse')
    ax1.legend(loc="upper left")
    ax1.set_xlabel("Layer index")
    ax1.set_ylabel("Prob. " + POS1)
    fig.suptitle("Likelihood of " + POS1 + " Candidates by Layer")
    plt.xlim(1, 24)
    fig.tight_layout()
    plt.show()


net_probabilities()


# Now th the data are all aggregated for plotting, you can also do some quick statistical tests on them.
def calculate_stats(p1_results, p2_results, string_label, alpha=0.01):
    for layer in range(p1_results.shape[0]):
        stat, p = wilcoxon(p1_results[layer], p2_results[layer], alternative='greater')
        _, less_p = wilcoxon(p1_results[layer], p2_results[layer], alternative='less')
        if p < alpha:
            print("Significantly greater:\t", string_label, " for layer", layer + layer_offset)
            continue
        if less_p < alpha:
            print("Significantly less:\t", string_label, " for layer", layer + layer_offset)
            continue
        print("Not significant for layer", layer + layer_offset)
    print()


calculate_stats(p1_pos1_change_by_sentence, p2_pos1_change_by_sentence, parse1_label + " parse " + POS1 + " than " + parse2_label + " Parse")
calculate_stats(p1_pos2_change_by_sentence, p2_pos2_change_by_sentence, parse1_label + " parse " + POS2 + " than " + parse2_label + " Parse")
calculate_stats(p1_pos1_change_by_sentence, np.zeros_like(p1_pos1_change_by_sentence), parse1_label + " parse " + POS1 + " update than 0")
calculate_stats(p1_pos2_change_by_sentence, np.zeros_like(p1_pos2_change_by_sentence), parse1_label + " parse " + POS2 + " update than 0")
calculate_stats(p2_pos2_change_by_sentence, np.zeros_like(p2_pos2_change_by_sentence), parse2_label + " parse " + POS2 + " update than 0")
calculate_stats(p2_pos1_change_by_sentence, np.zeros_like(p2_pos1_change_by_sentence), parse2_label + " parse " + POS1 + " update than 0")


# Count for each type of embedding the percentage of time certain parts of speech were the answer. This allows us to
# see at a very high level if using the counterfactuals changes the actual answers that the model uses.
total_count = 0
original_pos1_count = 0
original_pos2_count = 0
p1_pos1_count = 0
p1_pos2_count = 0
p2_pos1_count = 0
p2_pos2_count = 0
for layer in test_layers:
    for sentence_idx in range(len(original_pos1s)):
        total_count += 1
        original_sentence_pos1s = original_pos1s[sentence_idx, layer]
        original_sentence_pos2s = original_pos2s[sentence_idx, layer]
        if np.max(original_sentence_pos1s) > np.max(original_sentence_pos2s):
            original_pos1_count += 1
        else:
            original_pos2_count += 1
        p1_sentence_pos1s = p1_pos1s[sentence_idx, layer]
        p1_sentence_pos2s = p1_pos2s[sentence_idx, layer]
        if np.max(p1_sentence_pos1s) > np.max(p1_sentence_pos2s):
            p1_pos1_count += 1
        else:
            p1_pos2_count += 1
        p2_sentence_pos1s = p2_pos1s[sentence_idx, layer]
        p2_sentence_pos2s = p2_pos2s[sentence_idx, layer]
        if np.max(p2_sentence_pos1s) > np.max(p2_sentence_pos2s):
            p2_pos1_count += 1
        else:
            p2_pos2_count += 1

print("Percentage original pos1_count", original_pos1_count / total_count)
print("Percentage original pos2_count", original_pos2_count / total_count)
print("Percentage parse1 pos1_count", p1_pos1_count / total_count)
print("Percentage parse1 pos2_count", p1_pos2_count / total_count)
print("Percentage parse2 pos1_count", p2_pos1_count / total_count)
print("Percentage parse2 pos2_count", p2_pos2_count / total_count)

