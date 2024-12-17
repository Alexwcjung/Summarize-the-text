import streamlit as st
import pandas as pd
import random
import transformers 
import pipeline
import transformers.utils.logging

# Disable progress bars globally
transformers.utils.logging.set_verbosity_error()

def summarize_story(story, max_length=100, min_length=30):
    """
    Summarizes the given story using a pre-trained model.

    Parameters:
    - story (str): The story text to summarize.
    - max_length (int): Maximum length of the summary.
    - min_length (int): Minimum length of the summary.

    Returns:
    - str: The summarized story.
    """
    try:
        # Load a pre-trained summarization pipeline with progress bars disabled
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

        # Generate the summary
        summary = summarizer(
            story,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )

        return summary[0]["summary_text"]
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
if __name__ == "__main__":
    print("Type your story below:")
    user_story = input()

    print("\nSummarizing your story...\n")
    summary = summarize_story(user_story)
    print("Summary:\n", summary)
