import os

def rabin_karp(text, pattern, prime=101):
    m = len(pattern)
    n = len(text)
    pattern_hash = 0
    text_hash = 0
    h = 1
    d = 256
    found = False

    print("Rabin-Karp Algorithm:")

    for i in range(m - 1):
        h = (h * d) % prime

    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i + m] == pattern:
                print(f"Pattern found at index {i}")
                print(f"Pattern: {pattern}")
                found = True

        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime

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

            rabin_karp(text, pattern)

    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except IOError:
        print(f"An error occurred while reading the file {file_path}.")

file_path = input("Enter the file path: ")
pattern = input("Enter the pattern to search for: ")

search_pattern_in_file(file_path, pattern)
