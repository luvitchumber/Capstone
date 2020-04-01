"""This module implements a binary search through the method search(). Given a
key, the method will search through a previously sorted list of dictionaries
and return the index of the first matching value.

Author: Luvit Chumber
Date: 2020-03-23
"""
from dict_quad_sorts import insertion_sort


def search(data, key, search_val):
    """Searches through a particular key from a list of dictionaries."""
    insertion_sort(data, key)

    low = 0
    high = len(data) - 1

    while low <= high:
        mid = (high + low) // 2
        mid_dict = data[mid]

        if mid_dict[key] == search_val:
            return mid
        elif search_val < mid_dict[key]:
            high = mid - 1
        else:
            low = mid + 1
    return None
