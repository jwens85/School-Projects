import sys
import os
sys.path.append(os.getcwd())

from W1CT_Face_Detector_Advanced import WikipediaLLMSummarizer

def test_wikipedia_fetch():
    """Test Wikipedia fetching with HuggingFace summarization"""
    print("Testing Wikipedia API integration with HuggingFace summarization...")
    
    # Initialize with HuggingFace transformers
    summarizer = WikipediaLLMSummarizer()
    
    # Test with a known scientist
    test_scientists = ["Albert Einstein", "Max Planck"]  # Reduced for faster testing
    
    for scientist in test_scientists:
        print(f"\n{'='*50}")
        print(f"Testing: {scientist}")
        print('='*50)
        
        result = summarizer.get_scientist_summary(scientist)
        
        print(f"Status: {result['status']}")
        print(f"Wikipedia available: {result['wikipedia_available']}")
        if result['wikipedia_available']:
            print(f"Content length: {result['wikipedia_length']} characters")
            print(f"Cached: {result['cached']}")
            print(f"\nSUMMARY:")
            print(f"{result['summary']}")
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    test_wikipedia_fetch()