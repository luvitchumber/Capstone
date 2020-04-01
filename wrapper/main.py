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


def main():
    """Program execution starts here."""
    intersections = read_intersections("intersections.data")
    #sel_intersection = intersections[0]['name']
    sel_intersection = "4589"

    df = get_dataset(sel_intersection, intersections)
    x_train, x_test, y_train, y_test = preprocessing(df)

    model = LSTMModel(x_train.shape[1], y_train.shape[1])
    model_file = search(intersections, 'name', sel_intersection)

    main_menu = [
        "CAPSTONE TRAFFIC PREDICTION",

        "Select Intersection",
        "List of Intersections",
        "Train Intersection",
        "Test Intersection (Accuracy)",
        "Route Check (Not Implemented Yet)"
    ]
    train_menu = [
        "Train Mode:",

        "Train from scratch w/ events",
        "Train from file w/ events",
        "Train from scratch",
        "Train from file",
    ]

    while True:
        print("Currently Selected Intersection:", sel_intersection)
        choice = menu.do_menu(main_menu)
        if choice is None:
            return  # Exit main() (and program).
        if choice == 1:
            # Select Intersection
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
            # TODO test each option
            choice = menu.do_menu(train_menu)
            if choice == 1:
                x_train, x_test, y_train, y_test = preprocessing(df, events=True)
                model = LSTMModel(x_train.shape[1], y_train.shape[1])

                intersection_idx = search(intersections, 'name', sel_intersection)
                model_file = "model/" + sel_intersection + ".hdf"
                if os.path.exists(model_file):
                    os.remove(model_file)
                model.init_network(hidden_size=50)
            elif choice == 2:
                x_train, x_test, y_train, y_test = preprocessing(df, events=True)
                model = LSTMModel(x_train.shape[1], y_train.shape[1])

                intersection_idx = search(intersections, 'name', sel_intersection)
                model_file = "model/" + sel_intersection + ".hdf"
                if os.path.exists(model_file):
                    model.load_network(model_file)
                else:
                    print("Model does not exist, starting from scratch")
                    model.init_network(hidden_size=50)
            elif choice == 3:
                x_train, x_test, y_train, y_test = preprocessing(df, events=False)
                model = LSTMModel(x_train.shape[1], y_train.shape[1])

                intersection_idx = search(intersections, 'name', sel_intersection)
                model_file = "model/" + sel_intersection + ".hdf"
                if os.path.exists(model_file):
                    os.remove(model_file)
                model.init_network(hidden_size=50)
            elif choice == 4:
                x_train, x_test, y_train, y_test = preprocessing(df, events=False)
                model = LSTMModel(x_train.shape[1], y_train.shape[1])

                intersection_idx = search(intersections, 'name', sel_intersection)
                model_file = "model/" + sel_intersection + ".hdf"
                if os.path.exists(model_file):
                    model.load_network(model_file)
                else:
                    print("Model does not exist, starting from scratch")
                    model.init_network(hidden_size=50)
            try:
                e = int(input("Enter Epochs to train for (Default=50): "))
                model.epochs = e
            except ValueError:
                print("Invalid number entered, using default value")
            model.train(x_train, y_train, model_file)

            intersections[intersection_idx]['model'] = model_file
            save_intersections(intersections, "intersections.data")

        elif choice == 4:
            # Test Intersections
            model_file = "model/" + sel_intersection + ".hdf"
            if os.path.exists(model_file):
                model.load_network(model_file)
                test_output = model.predict(x_test, y_test)
                N, S, E, W = confusion_matrix(test_output, y_test, True)
            else:
                print("Please train intersection first")

        elif choice == 5:
            # Route Check
            print("  Not implemented yet, please check back later")


main()
