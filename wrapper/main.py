from data_processing import *
from lstm_model import LSTMModel
from dict_bin_search import search
import pandas as pd
import os.path
import menu


def get_dataset(sel_intersection, intersections):
    intersection_file = search(intersections, 'name', sel_intersection)
    intersection_file = intersections[intersection_file]['dataset']
    df = pd.read_csv(intersection_file)
    return df


def train(model, x_train, y_train, model_file):
    model.train(x_train, y_train, model_file, save=True)


def main():
    """Program execution starts here."""
    intersections = read_intersections("intersections.data")
    #sel_intersection = intersections[0]['name']
    sel_intersection = "4589"
    df = get_dataset(sel_intersection, intersections)

    x_train, x_test, y_train, y_test = preprocessing(df)
    model = LSTMModel(x_train.shape[1], y_train.shape[1])
    model_file = search(intersections, 'name', sel_intersection)
    if model_file is not None:
        model_file = intersections[model_file]['model']
        model.load_network(model_file)

    main_menu = [
        "CAPSTONE TRAFFIC PREDICTION",

        "Select Intersection",
        "List of Intersections",
        "Train Intersection",
        "Test Intersection (Accuracy)",
        "Route Check (Not Implemented Yet)"
    ]
    while True:
        choice = menu.do_menu(main_menu)
        if choice is None:
            return  # Exit main() (and program).
        if choice == 1:
            # Select Intersection
            print("Currently Selected Intersection:", sel_intersection)
            temp_menu = ["Please Select a New Intersection"]

            for line in intersections:
                option = line["name"] + ": " + line["street"]
                temp_menu.append(option)

            choice = menu.do_menu(temp_menu)
            if choice is not None:
                sel_intersection = intersections[choice - 1]['name']
                print(sel_intersection, "set as current intersection.")

                df = get_dataset(sel_intersection, intersections)
                x_train, x_test, y_train, y_test = preprocessing(df)

                model = LSTMModel(x_train.shape[1], y_train.shape[1])
                model_file = search(intersections, 'name', sel_intersection)
                if model_file is not None:
                    model_file = intersections[model_file]['model']
                    model.load_network(model_file)

        elif choice == 2:
            # List Intersections
            print_intersections(intersections)
        elif choice == 3:
            # Train Intersections
            if model_file is not None:
                model.load_network(model_file)
            else:
                intersection_idx = search(intersections, 'name', sel_intersection)
                model_file = "model/" + sel_intersection + ".hdf"
                model.init_network(hidden_size=50)

            model.train(x_train, y_train, model_file)

            intersections[intersection_idx]['model'] = model_file

            save_intersections(intersections, "intersections.data")

        elif choice == 4:
            # Test Intersections
            test_output = model.predict(x_test, y_test)
            N, S, E, W = confusion_matrix(test_output, y_test, True)

        elif choice == 5:
            # Route Check
            print("  Not implemented yet, please check back later")


main()
