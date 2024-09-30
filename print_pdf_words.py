import PyPDF2
import matplotlib.pyplot as plt
from collections import Counter
import re

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file_path):
    text = ""
    with open(pdf_file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to clean and get words from text
def get_words(text):
    # Use regular expressions to extract only words (alphanumeric sequences)
    words = re.findall(r'\b\w+\b', text.lower())  # Convert to lowercase for consistency
    return words

# Function to get word length frequencies
def get_word_length_frequencies(words):
    word_lengths = [len(word) for word in words]  # Calculate word lengths
    length_count = Counter(word_lengths)  # Count occurrences of each word length
    return length_count

# Function to plot word length frequencies
def plot_word_length_frequencies(word_length_frequencies):
    lengths = list(word_length_frequencies.keys())
    counts = list(word_length_frequencies.values())

    plt.figure(figsize=(10, 6))
    plt.scatter(lengths, counts, marker='o', linestyle='-', color='b')
    plt.title('Word Length Frequency')
    plt.xlabel('Word Length')
    plt.ylabel('Number of Words')
    plt.grid(True)
    plt.show()

# Main function
def main(pdf_file_path):
    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_file_path)

    # Step 2: Get words from the extracted text
    words = get_words(text)

    # Step 3: Get word length frequencies
    word_length_frequencies = get_word_length_frequencies(words)

    # Step 4: Plot the frequencies
    plot_word_length_frequencies(word_length_frequencies)

# Example usage
pdf_file_path = R'C:\Users\tay\Desktop\magisterka\180357_Praca magisterska_newest.pdf'
main(pdf_file_path)
