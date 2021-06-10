# causal-probe
Implementation of counterfactual probing technique discussed in "What if This Modified That? Syntactic Interventions via Counterfactual Embeddings" by Tucker et al.
In brief, we generate counterfactual embeddings according to syntactic probes and see if these new embeddings modify the model outputs.

If you find this code useful, please cite the paper.
If you have questions, feel free to email mycal@mit.edu.

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
Evaluation is done in two steps: first we pass the counterfactuals through the model and record outputs, and then we plot these outputs.
We break the plotting up like this just because the first evaluation step can take a long time, so it's nice to have the saved files when you're playing with new plotting utils.

1) Measure the effect of counterfactual embeddings by running ``src/scripts/eval_counterfactuals.py``.
The variables near the top fo the script define directories to look up text data and the embeddings.
The script produces .txt files that save relevant metrics about the model outputs using original and updated embeddings.
2) Plot these saved outputs by running ``src/plotting/plot_counterfactual_results.py``.
You set relevant directories by directly modifying the variables for different directories.
There are lots of different types of plots to generate - the different plotting methods have comments at the top of them to say what each one does.
Based on the findings from our work, you should find a consistent effect from using the distance-based probes to generate counterfactuals for mask word prediction.
   
### What about QA Models?
Although the prior steps all focused on the masked word prediction, we support models for extractive question answering.
You'll have to redo everything for these models, though: generate new embeddings, train new probes, generate a test suite, generate counterfactuals, and plot.
Most of the time, this just means modifying config or a few variables at the top of scripts (in particular, make sure you use appropriate tokenizers and models!!!).
Because of the custom nature of plotting, we have a separate plotting script, ``plotting/plot_qa_counterfactual_results.py``, for visualizing the effect of QA counterfactuals.

## Congratulations!
Congratulations! You made it to the end. At this point, you should have done everything from training probes to plotting the effect of counterfactual embeddings.

But don't stop here! There's so much more to do, as detailed by some of the ideas below:

1) Use a different language model.
To do this, you'll need to do this whole process from scratch: create new embeddings, train probes, etc.
2) Evaluate on a different test suite.
If all you want is a different evaluation suite, you can reuse your trained probes,t you'll need to start at the "generating syntactically interesting setup" to create new sentences, trees, and counterfactuals.
3) Use a different type of probe.
Very cool idea. To do this, you'll have to start editing the code itself instead of just tweaking parameters. You'll likely want to define a new probe in ``probe.py``. Make sure to update config to point to the new class, as well as updating the delegation logic in ``training_utils.py`` to then point to that class.
4) Calculate other metrics for counterfactuals.
If you want to look into aspects of the generated counterfactuals, the good news is that you don't need to train new probes or even generate new counterfactuals. Just update ``eval_counterfactuals.py`` to calculate a new metric and save it to a file, and then write new plotting logic to plot those metrics.
We have an example of a fun metric of calculating the distance of counterfactuals in ``eval_counterfactuals.py`` that we do actively use.
   
If you have questions or more ideas, please reach out to mycal@mit.edu.
If you find this work useful, please cite the paper with the BibTex below:


