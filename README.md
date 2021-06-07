# causal-probe

## Open issues to clean up before publicizing fully:
1) Make gen_training_data.py take parameters.
2) Plotting probe performance needs to be a lot easier.
3) How do we copy trained probes to directories for generating counterfactuals?


## Example workflow
This README is mostly designed to walk you through a full working example from training probes to generating counterfactuals to evaluating results.
We'll work through a specific example because modifying bits of functionality should be relatively simple within the framework.

All scripts must be run from causal-probe/

### Training probes
To generate counterfactuals, we need trained probes. This isn't the core research here, so if you have trouble, just email mycal@mit.edu. But the steps below should get you training probes easily enough.
1) Run scripts/gen_embeddings.py with ``source_dir = 'data/ptb/'`` and ``filename='ptb_test``(and dev and train),
   As a heads up, the resulting .hdf5 files are quite big, so make sure you have about 80 GB of disk space.
   Oh, and set break_on_qmark to False, because we don't want to break things up by question mark when training the probe (but we will later for QA counterfactual embeddings).
2) Run ``src/scripts/train_probes.py`` with a parameter pointing to the config file ``config/example/parse_dist_ptb.yaml``. This trains the probes and takes a few hours to run.
Trained probes, with performance metrics, are saved to the reporting destination, specified in config.
 3) If you want to plot the probe performance metrics instead of just reading the files, look at the ``plotting/plot_probe_perf.py`` script.
   It pulls out some of the basic metrics and plots them. You might have to do some editing of the file if you want the depth probe vs. distance probe, for example.
   
### Generating syntactically interesting setup.
This is how to generate the data that we'll use for counterfactuals. It's actually a lot like the steps used for training probes.


1) Run scripts/gen_cloze_suite.py. This generates some example sentences and trees in data/example.
2) Generate the conllx file by going into the stanford nlp folder somewhere and running the following command (obviously updated for your specific paths)

``sudo java -mx3g -cp "*" edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile ~/src/causal-probe/data/example/text.trees -checkConnected -basic -keepPunct -conllx > ~/src/causal-probe/data/example/text.conllx``

3) Run scripts/gen_embeddings.py. This generates embeddings for each layer for each of the sentences in the specified text file and saves them to an hdf5 file.

At this point, we have the embeddings for the interesting sentences created, so in the next step, we will create counterfactuals.

### Generating the counterfactuals.
Now that we have the embeddings for the interesting sentences and the interesting parses, let's generate the counterfactual embeddings.

We'll end up saving the embeddings as files, and it makes sense to group them with the trained probes, but maybe not in the saved_models directory.

1) To copy over the trained models, you can do it by hand, or you can use the ``src/scripts/migrate_trained_probes.py`` script. It just copies over the model parameters.
2) Run ``src/scripts/gen_counterfactuals.py`` with the argument of ``config/example/counterfactual_dist_cloze.yaml``.
   This generates two .hdf5 files in the directories with the probe parameters (all under counterfactuals).
   ``updated_words.hdf5`` had the counterfactual embeddings; ``original_words.hdf5`` has the original embeddings, but only for the words that got updated.
   For QA models, for example, this means only the words in the sentence, rather that words in the question as well.

### Evaluating counterfactual behaviors.
We've saved the counterfactual embeddings to hdf5 files in the previous step. Now we want to see if it has changed the model outputs.


