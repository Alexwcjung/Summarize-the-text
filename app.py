import streamlit as st
from transformers import pipeline
import transformers.utils.logging

# Disable progress bars globally
transformers.utils.logging.set_verbosity_error()

# Function to summarize the story
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
        # Load a pre-trained summarization pipeline
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

# Streamlit UI
st.title("Text Summarization App")
st.write("Enter your text below and get a summarized version using Hugging Face Transformers!")

# User input for text
user_story = st.text_area("Enter your story here:", placeholder="Type your story...")

# User input for parameters
max_length = st.slider("Maximum summary length", min_value=50, max_value=300, value=100, step=10)
min_length = st.slider("Minimum summary length", min_value=10, max_value=50, value=30, step=5)

# Button to generate the summary
if st.button("Summarize"):
    if user_story:  # Check if text is provided
        with st.spinner("Summarizing..."):
            summary = summarize_story(user_story, max_length=max_length, min_length=min_length)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please enter some text to summarize.")

# Footer
st.caption("Built with ❤️ using Hugging Face Transformers and Streamlit")
