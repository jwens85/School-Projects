import os
import timeit
from collections import deque, defaultdict

class AhoCorasick:
    def __init__(self, patterns):
        self.trie = {}
        self.output = defaultdict(list)
        self.fail = {}
        self.build_trie(patterns)
        self.build_failure_links()

    def build_trie(self, patterns):
        for pattern in patterns:
            node = self.trie
            for char in pattern:
                if char not in node:
                    node[char] = {}
                node = node[char]
            self.output[id(node)].append(pattern)

    def build_failure_links(self):
        queue = deque()
        for key, node in self.trie.items():
            self.fail[id(node)] = self.trie
            queue.append(node)

        while queue:
            current_node = queue.popleft()
            for key, next_node in current_node.items():
                fail_state = self.fail[id(current_node)]
                while fail_state and key not in fail_state:
                    fail_state = self.fail.get(id(fail_state))
                self.fail[id(next_node)] = fail_state[key] if fail_state and key in fail_state else self.trie
                if id(self.fail[id(next_node)]) in self.output:
                    self.output[id(next_node)].extend(self.output[id(self.fail[id(next_node)])])
                queue.append(next_node)

    def search(self, text):
        node = self.trie
        matches = defaultdict(list)
        for i, char in enumerate(text):
            while node and char not in node:
                node = self.fail.get(id(node))
            if not node:
                node = self.trie
                continue
            node = node[char]
            if id(node) in self.output:
                for pattern in self.output[id(node)]:
                    matches[pattern].append(i - len(pattern) + 1)
        return matches

def run_aho_corasick(text, patterns):
    start_time = timeit.default_timer()
    aho_corasick = AhoCorasick(patterns)
    match_positions = aho_corasick.search(text)
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
        print(f"\nAho-Corasick found {len(positions)} matches for the pattern '{pattern}'")

    print(f"\nTotal matches found across all patterns: {total_matches}")
    print(f"Execution time: {elapsed_time:.3f} seconds")
    return match_positions, elapsed_time

def search_multiple_patterns():
    choice = input("Would you like to search in a text or a file? Enter 'T' for text and 'F' for file: ").strip().upper()

    if choice == 'T':
        text = input("Enter the text to search in: ")
        patterns = input("Enter the patterns to search for, separated by commas: ").split(',')
        patterns = [pattern.strip() for pattern in patterns]
        run_aho_corasick(text, patterns)
    elif choice == 'F':
        file_path = input("Enter the file path: ").strip()
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as file:
                    text = file.read()
                    print("File content loaded successfully.")
                    patterns = input("Enter the patterns to search for, separated by commas: ").split(',')
                    patterns = [pattern.strip() for pattern in patterns]
                    run_aho_corasick(text, patterns)
            except IOError:
                print(f"An error occurred while reading the file: {file_path}")
        else:
            print(f"File not found: {file_path}")
    else:
        print("Invalid choice. Please enter 'T' for text or 'F' for file.")

search_multiple_patterns()
