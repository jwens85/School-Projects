import os

def compute_lps(pattern):
    lps = [0] * len(pattern)
    length = 0
    i = 1

    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text, pattern):
    lps = compute_lps(pattern)
    i = 0
    j = 0
    found = False

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == len(pattern):
            print(f"Pattern found at index {i - j}")
            print(f"Pattern: {pattern}")
            found = True
            j = lps[j - 1]

        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    if not found:
        print("No Results Match")

def search_pattern_in_file(file_path, pattern):
    try:
        if os.path.exists(file_path):
            print(f"File found: {file_path}")
        else:
            print(f"File not found: {file_path}")
            return

        abs_path = os.path.abspath(file_path)
        print(f"Absolute path of the file: {abs_path}")

        with open(file_path, 'r') as file:
            text = file.read()
            print("File content loaded successfully.")

            kmp_search(text, pattern)

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except IOError:
        print(f"An error occurred while reading the file {file_path}.")

file_path = input("Enter the file path: ")
pattern = input("Enter the pattern to search for: ")

search_pattern_in_file(file_path, pattern)
