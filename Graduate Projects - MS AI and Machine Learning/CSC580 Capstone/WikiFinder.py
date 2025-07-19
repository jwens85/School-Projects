#!/usr/bin/env python3
"""
WikiFinder.py - Simple Wikipedia URL Collector for Solvay Conference Scientists
Just loops through each scientist and asks for the URL. No validation, no suggestions.
"""

import json

# Complete list of 29 scientists from the 1927 Solvay Conference
scientists = [
    # Front Row (9 scientists)
    "Irving Langmuir",
    "Max Planck", 
    "Marie Curie",
    "Hendrik Lorentz",
    "Albert Einstein",
    "Paul Langevin",
    "Charles Eugene Guye",
    "Charles Wilson",  # Also known as CTR Wilson
    "Owen Richardson",
    
    # Middle Row (9 scientists)
    "Peter Debye",
    "Martin Knudsen",
    "William Lawrence Bragg",
    "Hendrik Anthony Kramers",
    "Paul Dirac",
    "Arthur Compton",
    "Louis de Broglie",
    "Max Born",
    "Niels Bohr",
    
    # Back Row (11 scientists)
    "Auguste Piccard",
    "Emile Henriot",
    "Paul Ehrenfest",
    "Edouard Herzen",
    "Theophile de Donder",
    "Erwin Schrodinger",
    "Jules Emile Verschaffelt",  # Also known as JE Verschaffelt
    "Wolfgang Pauli",
    "Werner Heisenberg",
    "Ralph Fowler",
    "Leon Brillouin"
]

def main():
    """Simple loop through each scientist and collect URLs"""
    url_mapping = {}
    
    print("WikiFinder.py - Simple URL Collector")
    print("="*50)
    print(f"We'll go through all {len(scientists)} scientists.")
    print("Just enter the Wikipedia URL for each one.")
    print("="*50)
    
    for i, scientist in enumerate(scientists, 1):
        print(f"\n[{i}/{len(scientists)}] {scientist}")
        url = input("Enter Wikipedia URL: ").strip()
        
        if url:
            url_mapping[scientist] = url
            print(f"✓ Added: {scientist}")
        else:
            print(f"⚠ Skipped: {scientist}")
    
    # Save results
    output_file = "data/W1CT_Faces/wikipedia_url_mapping.json"
    with open(output_file, 'w') as f:
        json.dump(url_mapping, f, indent=2, sort_keys=True)
    
    print(f"\n{'='*50}")
    print("COMPLETE!")
    print(f"Collected {len(url_mapping)} URLs out of {len(scientists)} scientists")
    print(f"Saved to: {output_file}")
    print("="*50)

if __name__ == "__main__":
    main()