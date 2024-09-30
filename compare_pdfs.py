import difflib
import re
import PyPDF2


# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
        return text


def split_into_sentences(text):
    # Split text into sentences based on punctuation followed by a space
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text.strip())
    return sentences


def split_into_words(sentence):
    # Split sentence into words using spaces and punctuations as delimiters
    return re.findall(r'\w+', sentence)


def compare_sentences(text1, text2):
    sentences1 = split_into_sentences(text1)
    sentences2 = split_into_sentences(text2)

    d = difflib.Differ()
    diff = list(d.compare(sentences1, sentences2))

    return diff


def compare_words(text1, text2):
    words1 = split_into_words(text1)
    words2 = split_into_words(text2)

    d = difflib.Differ()
    diff = list(d.compare(words1, words2))

    return diff


def compare_chars(text1, text2):
    d = difflib.Differ()
    diff = list(d.compare(list(text1), list(text2)))

    return diff


def print_diff(diff):
    for line in diff:
        if line.startswith('+ ') or line.startswith('- ') or line.startswith('? '):
            print(line)


# Main comparison function
def compare_texts(text1, text2):
    print("Comparing by sentences:")
    sentence_diff = compare_sentences(text1, text2)
    print_diff(sentence_diff)

    print("\nComparing by words:")
    word_diff = compare_words(text1, text2)
    print_diff(word_diff)

    print("\nComparing by characters:")
    char_diff = compare_chars(text1, text2)
    print_diff(char_diff)


# Example usage
def compare_pdfs(pdf1_path, pdf2_path):
    # Extract text from both PDFs
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)

    # Compare the texts from both PDFs
    compare_texts(text1, text2)


# Provide the paths to your two PDF files
pdf1_path = '180357_Praca magisterska_poprawiona_30.09.pdf'  # Replace with the path to your first PDF
pdf2_path = '180357_Praca magisterska_poprawiona_30.09_zmiany.pdf'  # Replace with the path to your second PDF

# Run the comparison
compare_pdfs(pdf1_path, pdf2_path)
