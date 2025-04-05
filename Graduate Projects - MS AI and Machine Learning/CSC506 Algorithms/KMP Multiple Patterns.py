import os
import timeit

def compute_lps_array(pattern):
    lps_array = [0] * len(pattern)
    prefix_length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[prefix_length]:
            prefix_length += 1
            lps_array[i] = prefix_length
            i += 1
        else:
            if prefix_length != 0:
                prefix_length = lps_array[prefix_length - 1]
            else:
                lps_array[i] = 0
                i += 1
    return lps_array

def kmp_search_multiple_patterns(text, patterns):
    match_positions = {pattern: [] for pattern in patterns}
    for pattern in patterns:
        lps_array = compute_lps_array(pattern)
        text_index = 0
        pattern_index = 0
        pattern_length = len(pattern)

        while text_index < len(text):
            if pattern[pattern_index] == text[text_index]:
                text_index += 1
                pattern_index += 1

            if pattern_index == pattern_length:
                match_positions[pattern].append(text_index - pattern_index)
                pattern_index = lps_array[pattern_index - 1]
            elif text_index < len(text) and pattern[pattern_index] != text[text_index]:
                if pattern_index != 0:
                    pattern_index = lps_array[pattern_index - 1]
                else:
                    text_index += 1
    return match_positions

def run_kmp_multiple_patterns(text, patterns):
    start_time = timeit.default_timer()
    match_positions = kmp_search_multiple_patterns(text, patterns)
    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time

    total_matches = 0
    for pattern, positions in match_positions.items():
        print(f"\nMatches for pattern '{pattern}':")
        if positions:
            for count, position in enumerate(positions, start=1):
                print(f"{count}. Match found at index {position}")
        else:
            print("No matches found")
        total_matches += len(positions)
        print(f"\nKMP found {len(positions)} matches for the pattern '{pattern}'")

    print(f"\nTotal matches found across all patterns: {total_matches}")
    print(f"Execution time: {elapsed_time:.3f} seconds")
    return match_positions, elapsed_time

def search_multiple_patterns():
    choice = input("Would you like to search in a text or a file? Enter 'T' for text and 'F' for file: ").strip().upper()

    if choice == 'T':
        text = input("Enter the text to search in: ")
        patterns = input("Enter the patterns to search for, separated by commas: ").split(',')
        patterns = [pattern.strip() for pattern in patterns]
        run_kmp_multiple_patterns(text, patterns)
    elif choice == 'F':
        file_path = input("Enter the file path: ").strip()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                    print("File content loaded successfully.")
                    patterns = input("Enter the patterns to search for, separated by commas: ").split(',')
                    patterns = [pattern.strip() for pattern in patterns]
                    run_kmp_multiple_patterns(text, patterns)
            except IOError:
                print(f"An error occurred while reading the file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Invalid choice. Please enter 'T' for text or 'F' for file.")

search_multiple_patterns()
