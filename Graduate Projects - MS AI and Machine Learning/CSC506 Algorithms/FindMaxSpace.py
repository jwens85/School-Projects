# FindMax Space Complexity: O(1)

def find_max_space_complexity(lst, lst_size):
    if lst_size >= 1:
        maximum = lst[0]
        i = 1
        while i < lst_size:
            if lst[i] > maximum:
                maximum = lst[i]
            i += 1
        return maximum
    else:
        return None  # In case the list size is invalid (empty list)

# Example usage:
lst = [3, 5, 2, 9, 6]
lst_size = len(lst)

result = find_max_space_complexity(lst, lst_size)
if result is not None:
    print("The maximum value is:", result)
else:
    print("The list is empty or invalid")
