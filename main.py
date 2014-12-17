"""
Interfaz between Quepy and ACTIVEpipe.

Usage:
    -- For classification
    -- For training
"""

import argparse
from termcolor import colored

from activepipe import ActivePipeline
from visualization import Printer, feature_bootstrap, instance_bootstrap

printer = Printer()

def get_class_for_instance(instance, classes):
    printer.show_instruction("What is the correct template? Write the number or STOP")
    printer.info(instance, bold=True)
    message = "{} - {}"
    for (counter, class_name) in enumerate(classes):
        print message.format(counter, class_name)
    line = raw_input(">>> ")
    try:
        line = int(line)
        prediction = classes[line]
    except (ValueError, IndexError):
        return line
    printer.info_success("Adding result")
    return prediction


def get_class(classes_list):
    printer.show_instruction( "Choose a class number to label its features")
    for index, class_ in enumerate(classes_list):
        print index, class_
    line = raw_input(">>> ")
    try:
        line = int(line)
    except ValueError:
        printer.info_warning('Choose a number')
        return line if (line == 'stop' or line == 'train') else None
    if line < 0 or line >= len(classes_list):
        printer.info_warning('Choose a number between 0 and ' + str(len(classes_list)))
        return False
    return classes_list[line]


def get_labeled_features(class_name, features):
    printer.show_instruction( "Insert the asociated feature numbers separated by a blank space.")
    printer.info(class_name, bold=True)
    for number, feature in enumerate(features):
        print number, feature
    line = raw_input(">>> ")
    if line.lower() == 'stop':
        return 'stop'
    indexes = line.split()
    result = []
    for index in indexes:
        try:
            index = int(index)
            result.append(features[index])
        except ValueError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--output_file', required=False,
                        type=str)
    parser.add_argument('-e', '--emulate', action='store_true')
    parser.add_argument('-l', '--label_corpus', action='store_true')
    args = parser.parse_args()

    pipe = ActivePipeline(session_filename=args.output_file,
                          emulate=args.emulate)
    try:
        #pipe.instance_bootstrap(get_class_for_instance)
        res = feature_bootstrap(pipe, get_class, get_labeled_features)
    except:
        pipe.save_session('sessions/error')
        import ipdb; ipdb.set_trace()

    print pipe.get_report()

    filename = raw_input("Insert the filename to save or press enter\n")
    if filename:
        pipe.save_session("sessions/{}".format(filename))

    if args.label_corpus:
        printer.info_success("Adding {} classes.".format(pipe.label_corpus()))

    printer.info_success('Thanks for labeling ' + str(res) + ' instances')


if __name__ == '__main__':
    main()
