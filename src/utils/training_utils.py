import models.model as model
import models.probe as probe
import utils.dataset as dataset
import utils.loss as loss
import utils.reporter as reporter
import utils.task as task


# Helper class that looks up things like relevant dataset, model, or probe classes from config.

def choose_task_classes(args):
    """Chooses which task class to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a task specification.
  """
    if args['probe']['task_name'] == 'parse-distance':
        task_class = task.ParseDistanceTask
        reporter_class = reporter.WordPairReporter
        if args['probe_training']['loss'] == 'L1':
            loss_class = loss.L1DistanceLoss
        else:
            raise ValueError("Unknown loss type for given probe type: {}".format(
                args['probe_training']['loss']))
    elif args['probe']['task_name'] == 'parse-depth':
        task_class = task.ParseDepthTask
        reporter_class = reporter.WordReporter
        if args['probe_training']['loss'] == 'L1':
            loss_class = loss.L1DepthLoss
        else:
            raise ValueError("Unknown loss type for given probe type: {}".format(
                args['probe_training']['loss']))
    else:
        raise ValueError("Unknown probing task type: {}".format(
            args['probe']['task_name']))
    return task_class, reporter_class, loss_class


def choose_dataset_class(args):
    """

    Legacy layer of abstraction for representing dataset class. We only have one type, so just pipes through right
    now.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a dataset.
  """
    if args['model']['model_type'] == 'BERT-disk':
        dataset_class = dataset.BERTDataset
    else:
        raise ValueError("Unknown model type for datasets: {}".format(
            args['model']['model_type']))

    return dataset_class


def choose_probe_class(args):
    """Chooses which probe and reporter classes to use based on config.

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A probe_class to be instantiated.
  """
    if args['probe']['task_signature'] == 'word':
        return probe.OneWordPSDProbe
    elif args['probe']['task_signature'] == 'word_pair':
        return probe.TwoWordPSDProbe
    else:
        raise ValueError("Unknown probe type (probe function signature): {}".format(
            args['probe']['task_signature']))


def choose_model_class(args):
    """Chooses which representation learner class to use based on config.

    Like dataset, this is legacy from prior code supporting different model types. We only support BERT-disk

  Args:
    args: the global config dictionary built by yaml.
  Returns:
    A class to be instantiated as a model to supply word representations.
  """
    if args['model']['model_type'] == 'BERT-disk':
        return model.DiskModel
    else:
        raise ValueError("Unknown model type: {}".format(
            args['model']['model_type']))

