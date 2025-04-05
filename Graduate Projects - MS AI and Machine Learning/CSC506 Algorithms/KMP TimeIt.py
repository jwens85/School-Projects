import os
import timeit

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
    match_count = 0
    pattern_len = len(pattern)

    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1

        if j == pattern_len:
            print(f"Pattern found at index {i - j}")
            found = True
            match_count += 1
            j = lps[j - 1]

        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    if not found:
        print("No Results Match")
    else:
        print(f"Total matches found: {match_count}")

def run_kmp(text, pattern):
    start_time = timeit.default_timer()
    kmp_search(text, pattern)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print(f"Execution time: {elapsed_time:.6f} seconds")
    return elapsed_time

def search_pattern():
    choice = input("Would you like to search in a text or a file? Enter 'T' for text and 'F' for file: ").strip().upper()

    if choice == 'T':
        text = input("Enter the text to search in: ")
        pattern = input("Enter the pattern to search for: ")
        run_kmp(text, pattern)
    elif choice == 'F':
        file_path = input("Enter the file path: ").strip()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                    print("File content loaded successfully.")
                    pattern = input("Enter the pattern to search for: ")
                    run_kmp(text, pattern)
            except IOError:
                print(f"An error occurred while reading the file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Invalid choice. Please enter 'T' for text or 'F' for file.")

search_pattern()
