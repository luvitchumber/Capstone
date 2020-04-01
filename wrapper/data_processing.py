from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

sc = MinMaxScaler(feature_range=(0, 1))
features_idx_events =   [6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, -2, -1]
features_idx =          [6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20]
labels_idx = [21, 22, 23, 24]


def preprocessing(df, events=True, verbose=False):
    features = df.iloc[:, features_idx]
    labels = df.iloc[:, labels_idx]
    if events:
        features = df.iloc[:, features_idx_events]

    if verbose:
        print(features)
        print(labels)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True,
                                                        random_state=1997)

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, x_test, y_train, y_test


def confusion_matrix(predict, actual, Verbose=False):
    '''           p Low | p Med | p High
    a Low       |
    a Med       |
    a High      |
    '''

    W = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    S = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    N = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    E = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    for i in range(len(predict)):
        pred = predict[i]
        pred = [int(round(p, 0)) for p in pred]
        actu = actual[i]
        actu = [int(round(a, 0)) for a in actu]
        # print(actu)
        W[actu[0]][pred[0]] += 1
        S[actu[1]][pred[1]] += 1
        N[actu[2]][pred[2]] += 1
        E[actu[3]][pred[3]] += 1

    N[0][0] = ''
    N[0][1] = 'p Low'
    N[0][2] = 'p Med'
    N[0][3] = 'p High'
    N[1][0] = 'a Low'
    N[2][0] = 'a Med'
    N[3][0] = 'a High'

    E[0][0] = ''
    E[0][1] = 'p Low'
    E[0][2] = 'p Med'
    E[0][3] = 'p High'
    E[1][0] = 'a Low'
    E[2][0] = 'a Med'
    E[3][0] = 'a High'

    S[0][0] = ''
    S[0][1] = 'p Low'
    S[0][2] = 'p Med'
    S[0][3] = 'p High'
    S[1][0] = 'a Low'
    S[2][0] = 'a Med'
    S[3][0] = 'a High'

    W[0][0] = ''
    W[0][1] = 'p Low'
    W[0][2] = 'p Med'
    W[0][3] = 'p High'
    W[1][0] = 'a Low'
    W[2][0] = 'a Med'
    W[3][0] = 'a High'

    if Verbose == True:
        print()
        print("North Confusion Matrix")
        print(N[0])
        print(N[1])
        print(N[2])
        print(N[3])
        print()
        print("East Confusion Matrix")
        print(E[0])
        print(E[1])
        print(E[2])
        print(E[3])
        print()
        print("South Confusion Matrix")
        print(S[0])
        print(S[1])
        print(S[2])
        print(S[3])
        print()
        print("West Confusion Matrix")
        print(W[0])
        print(W[1])
        print(W[2])
        print(W[3])

    return N, S, E, W


def read_intersections(file_loc, verbose=False):
    intersections = list()
    f = open(file_loc, "r")

    keys = f.readline().rstrip('\n').split(',')
    if verbose:
        print("Keys:", keys)
        print("Intersections Data:")

    for line in f:
        line = line.rstrip('\n')
        line = line.split(',')
        intersection_dict = dict()
        for i, x in enumerate(line):
            intersection_dict[keys[i]] = x
        if verbose:
            print(intersection_dict)
        intersections.append(intersection_dict)

    f.close()
    return intersections


def save_intersections(intersections, file_loc):
    print("Saving updated intersections file")
    f = open(file_loc, "w")

    keys = intersections[0].keys()
    keys = ",".join(keys) + "\n"
    f.write(keys)

    for line in intersections:
        line = ",".join(line.values()) + "\n"
        f.write(line)

    f.close()


def print_intersections(intersections):
    print("List of Trained Intersections:")
    keys = intersections[0].keys()
    for line in intersections:
        for k, v in line.items():
            if k == "name":
                print(f"{k}: {v}")
            else:
                print(f"\t{k}: {v}")
        print()