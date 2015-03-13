# encoding: utf-8

from termcolor import colored


class Printer(object):
    def __init__(self):
        pass

    def nprint(self, msg):  # Normal print
        print msg

    def info(self, msg, bold=False):
        attrs = []
        if bold: attrs.append("bold")
        print colored(msg, "magenta", attrs=attrs)

    def info_success(self, msg):
        print colored(msg, "green", attrs=["bold"])

    def info_warning(self, msg):
        print colored(msg, "red", attrs=["bold"])

    def show_instruction(self, msg):
        print "*"*70
        print msg


def feature_bootstrap(activepipe, get_class, get_labeled_features,
                      max_iterations=None):
    """Presents a class and possible features until the prediction is stop.

    Args:
        get_class: A function that receives a list of classes and returns
        one of them. Can return None in case of error.
        get_labeled_features: A function that receives a class and a list
        of features. It must return a list of features associated with the
        class. Can return None in case of error.
        max_iterations: Optional. An integer. The cycle will execute at
        most max_iterations times if the user does not enter stop before.

    Returns:
        The number of features the user has labeled.
    """
    printer = Printer()

    result = 0
    while not max_iterations or result < max_iterations:
        class_name = get_class(activepipe.get_class_options())
        if not class_name:
            continue
        if class_name == 'stop':
            break
        if class_name == 'train':
            activepipe._train()
            activepipe._expectation_maximization()
            continue
        class_number = activepipe.classes.index(class_name)
        feature_numbers = activepipe.get_next_features(class_number)
        e_prediction = []
        prediction = []
        if activepipe.emulate:
            e_prediction = [f for f in feature_numbers
                            if activepipe.feature_corpus[class_number][f] == 1]
            feature_numbers = [f for f in feature_numbers
                               if f not in e_prediction]
            printer.info( "Adding {0} features from corpus for class {1}".format(
                len(e_prediction), class_name
            ))
        if feature_numbers:
            feature_names = [activepipe.training_corpus.get_feature_name(pos)
                             for pos in feature_numbers]
            prediction = get_labeled_features(class_name, feature_names)
            if not prediction and not e_prediction:
                continue
            if prediction == 'stop':
                break
            if prediction == 'train':
                activepipe._train()
                activepipe._expectation_maximization()
                continue
            prediction = [feature_numbers[feature_names.index(f)]
                          for f in prediction]
        activepipe.handle_feature_prediction(class_number,
                                       feature_numbers + e_prediction,
                                       prediction + e_prediction)
        result += len(prediction + e_prediction)
    return result


def instance_bootstrap(activepipe, get_labeled_instance, max_iterations=None):
    """Presents a new question to the user until the answer is 'stop'.

    Args:
        get_labeled_instance: A function that takes the representation of
        an instance and a list of possible classes. Returns the correct
        class for the instance.
        max_iterations: Optional. An integer. The cycle will execute at
        most max_iterations times if the user does not enter stop before.

    Returns:
        The number of instances the user has labeled.
    """
    printer = Printer()

    it = 0
    result = 0
    while ((not max_iterations or it < max_iterations) and
          len(activepipe.unlabeled_corpus)):
        it += 1
        new_index = activepipe.get_next_instance()
        try:
            new_instance = activepipe.unlabeled_corpus.instances[new_index]
        except IndexError:
            import ipdb; ipdb.set_trace()
        representation = activepipe.unlabeled_corpus.representations[new_index]
        if (activepipe.emulate and
            activepipe.unlabeled_corpus.primary_targets[new_index]):
            prediction = activepipe.unlabeled_corpus.primary_targets[new_index]
            message = "Emulation: Adding instance {}, {}".format(
                representation, prediction
            )
            printer.info(message)
        if (not activepipe.emulate or
            not activepipe.unlabeled_corpus.primary_targets[new_index]):
            classes = activepipe._most_probable_classes(new_instance)
            prediction = get_labeled_instance(representation, classes)
        if prediction == 'stop':
            break
        if prediction == 'train':
            activepipe._train()
            activepipe._expectation_maximization()
            continue

        activepipe.new_instances += 1
        result += 1
        instance, targets, r = activepipe.unlabeled_corpus.pop_instance(new_index)
        activepipe.user_corpus.add_instance(
            instance, [prediction] + targets, r
        )

    return result
