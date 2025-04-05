def rabin_karp(text, pattern, prime=101):
    m = len(pattern)
    n = len(text)
    pattern_hash = 0
    text_hash = 0
    h = 1
    d = 256 2672
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

text = input("Enter the text to search in: ")
pattern = input("Enter the pattern to search for: ")

rabin_karp(text, pattern)
