import os
import timeit

def rabin_karp_multiple_patterns_fixed(text, patterns, prime=101):
    pattern_lengths = [len(pattern) for pattern in patterns]
    text_length = len(text)
    base = 256
    rolling_multipliers = [1] * len(patterns)
    pattern_hashes = [0] * len(patterns)
    current_text_hashes = [0] * len(patterns)
    match_positions = {pattern: [] for pattern in patterns}

    for index, pattern in enumerate(patterns):
        for i in range(pattern_lengths[index] - 1):
            rolling_multipliers[index] = (rolling_multipliers[index] * base) % prime

        for i in range(pattern_lengths[index]):
            pattern_hashes[index] = (base * pattern_hashes[index] + ord(pattern[i])) % prime
            current_text_hashes[index] = (base * current_text_hashes[index] + ord(text[i])) % prime

    for i in range(text_length - min(pattern_lengths) + 1):
        for index, pattern in enumerate(patterns):
            pattern_length = pattern_lengths[index]
            if i + pattern_length <= text_length:
                if pattern_hashes[index] == current_text_hashes[index]:
                    if text[i:i + pattern_length] == pattern:
                        match_positions[pattern].append(i)

                if i < text_length - pattern_length:
                    current_text_hashes[index] = (base * (current_text_hashes[index] - ord(text[i]) * rolling_multipliers[index]) + ord(text[i + pattern_length])) % prime
                    if current_text_hashes[index] < 0:
                        current_text_hashes[index] += prime

    return match_positions

def run_rabin_karp_multiple_patterns(text, patterns):
    start_time = timeit.default_timer()
    match_positions = rabin_karp_multiple_patterns_fixed(text, patterns)
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
        print(f"\nRK found {len(positions)} matches for the pattern '{pattern}'")

    print(f"\nTotal matches found across all patterns: {total_matches}")
    print(f"Execution time: {elapsed_time:.3f} seconds")
    return match_positions, elapsed_time

def search_multiple_patterns():
    choice = input("Would you like to search in a text or a file? Enter 'T' for text and 'F' for file: ").strip().upper()

    if choice == 'T':
        text = input("Enter the text to search in: ")
        patterns = input("Enter the patterns to search for, separated by commas: ").split(',')
        patterns = [pattern.strip() for pattern in patterns]
        run_rabin_karp_multiple_patterns(text, patterns)
    elif choice == 'F':
        file_path = input("Enter the file path: ").strip()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                    print("File content loaded successfully.")
                    patterns = input("Enter the patterns to search for, separated by commas: ").split(',')
                    patterns = [pattern.strip() for pattern in patterns]
                    run_rabin_karp_multiple_patterns(text, patterns)
            except IOError:
                print(f"An error occurred while reading the file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Invalid choice. Please enter 'T' for text or 'F' for file.")

search_multiple_patterns()
