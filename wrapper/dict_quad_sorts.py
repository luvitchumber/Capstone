"""This module implements 4 different quadratic sorts which works on a list of
dictionaries and sorts a given key.

Author: Luvit Chumber
Date: 2020-03-23
"""


def insertion_sort(data, sort_on_key):
    """Sorts dictionary on key using insertion sort algorithm."""
    for i in range(1, len(data)):
        first_dict = data[i - 1]
        second_dict = data[i]
        while i > 0 and first_dict[sort_on_key] > second_dict[sort_on_key]:
            data[i - 1], data[i] = data[i], data[i - 1]
            i = i - 1
            first_dict = data[i - 1]
            second_dict = data[i]


def bubble_sort(data, sort_on_key):
    """Sorts dictionary on key using bubble sort algorithm."""
    swap_flag = True
    n = len(data)
    while swap_flag:
        swap_flag = False
        for i in range(1, n):
            first_dict = data[i - 1]
            second_dict = data[i]

            if first_dict[sort_on_key] > second_dict[sort_on_key]:
                data[i - 1], data[i] = data[i], data[i - 1]
                swap_flag = True


def bubble_sort_opt(data, sort_on_key):
    """sorts dictionary on key using optimized bubble sort algorithm."""
    swap_flag = True
    n = len(data)
    while swap_flag:
        swap_flag = False
        for i in range(1, n):
            first_dict = data[i - 1]
            second_dict = data[i]

            if first_dict[sort_on_key] > second_dict[sort_on_key]:
                data[i - 1], data[i] = data[i], data[i - 1]
                swap_flag = True
        n -= 1


def selection_sort(data, sort_on_key):
    """Sorts dictionary on key using selection sort algorithm."""
    n = len(data)
    for i in range(n - 1):
        min_idx = i  # assume first index is min to start
        min_dict = data[min_idx]
        for j in range(i + 1, n):
            check_dict = data[j]
            if check_dict[sort_on_key] < min_dict[sort_on_key]:
                min_idx = j
                min_dict = data[min_idx]
        if min_idx != i:
            data[i], data[min_idx] = data[min_idx], data[i]
