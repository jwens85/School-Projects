import os

def rabin_karp(text, pattern, prime=101):
    m = len(pattern)
    n = len(text)
    pattern_hash = 0
    text_hash = 0
    h = 1
    d = 256
    found = False

    print("\nRabin-Karp Algorithm:")

    for i in range(m - 1):
        h = (h * d) % prime

    for i in range(m):
        pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
        text_hash = (d * text_hash + ord(text[i])) % prime

    for i in range(n - m + 1):
        if pattern_hash == text_hash:
            if text[i:i + m] == pattern:
                print(f"Pattern found at index {i}")
                found = True

        if i < n - m:
            text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if text_hash < 0:
                text_hash += prime
    if not found:
        print("No Results Match")

def search_pattern():
    choice = input("Would you like to search in a text or a file? Enter 'T' for text and 'F' for file: ").strip().upper()

    if choice == 'T':
        text = input("Enter the text to search in: ")
        pattern = input("Enter the pattern to search for: ")
        rabin_karp(text, pattern)
    elif choice == 'F':
        file_path = input("Enter the file path: ").strip()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                    pattern = input("Enter the pattern to search for: ")
                    rabin_karp(text, pattern)
            except IOError:
                print(f"An error occurred while reading the file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Invalid choice. Please enter 'T' for text or 'F' for file.")

search_pattern()
