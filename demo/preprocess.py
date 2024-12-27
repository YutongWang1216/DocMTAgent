import argparse
import re
import jieba
import chardet
import pysbd
import pdfplumber
from docx import Document
import os

def detect_encoding(file_path):
    """
    Use chardet to detect the file encoding.
    """
    try:
        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"]
            confidence = result["confidence"]
            print(f"Detected encoding: {encoding} (Confidence: {confidence:.2f})")
            return encoding
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        return None


def split_sentences(text, lang):
    """
    Split sentences based on the language:
    - Chinese (lang starts with 'zh'): Use jieba for segmentation by punctuation.
    - Other languages: Use pysbd's Segmenter.
    """
    if lang.startswith("zh"):
        # Use jieba to segment Chinese sentences
        sentences = list(jieba.cut(text, cut_all=False))
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    else:
        try:
            # Use pySBD for sentence segmentation
            segmenter = pysbd.Segmenter(language=lang[:2], clean=True)
            sentences = segmenter.segment(text)
            return [sentence.strip() for sentence in sentences if sentence.strip()]
        except Exception as e:
            print(f"Error during sentence segmentation with pySBD: {e}")
            return []


def extract_text_from_pdf(pdf_path):
    """
    Extract text content from a PDF file using pdfplumber.
    """
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"  
        return text.strip()  
    except Exception as e:
        print(f"Error reading PDF file with pdfplumber: {e}")
        return ""


def extract_text_from_docx(docx_path):
    """
    Extract text content from a Word file.
    """
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        print(f"Error reading Word file: {e}")
        return ""


def process_file(input_path, output_path, lang, encoding=None):
    """
    Read the source file, split sentences based on the language, and write to the output path.
    """
    try:
        # Determine the file type
        is_pdf = input_path.lower().endswith(".pdf")
        is_docx = input_path.lower().endswith(".docx")

        # If it's a PDF file, extract text directly
        if is_pdf:
            print(f"Detected PDF file: {input_path}")
            content = extract_text_from_pdf(input_path)
            if not content.strip():
                raise ValueError("The PDF file contains no extractable text.")

        # If it's a Word file, extract text directly
        elif is_docx:
            print(f"Detected Word file: {input_path}")
            content = extract_text_from_docx(input_path)
            if not content.strip():
                raise ValueError("The Word file contains no extractable text.")

        # If it's a TXT file
        else:
            # If encoding is not provided, detect it automatically
            if not encoding:
                encoding = detect_encoding(input_path)
                if not encoding:
                    raise ValueError("Unable to detect encoding. Please specify the encoding manually.")

            print(f"Reading file: {input_path} with encoding: {encoding}")

            # Read the content of the TXT file
            with open(input_path, "r", encoding=encoding) as infile:
                content = infile.read()

        # Split sentences
        sentences = split_sentences(content, lang)

        # Write to the output file
        with open(output_path, "w", encoding="utf-8") as outfile:  # Output file is saved as UTF-8 by default
            for sentence in sentences:
                outfile.write(sentence + "\n")

        print(f"Processed {len(sentences)} sentences.")
        print(f"Output saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
    except UnicodeDecodeError:
        print(f"Error: Unable to decode the file '{input_path}' using encoding '{encoding}'. Please check the file encoding.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def main():
    """
    Main function to parse command-line arguments and call the file processing logic.
    """
    parser = argparse.ArgumentParser(description="Process a text file to one sentence per line format.")
    parser.add_argument("--input", type=str, required=True, help="Path to the source .txt file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the processed output .txt file")
    parser.add_argument("--lang", type=str, required=True, help="Language code (e.g., 'en', 'zh', 'fr', 'es')")
    parser.add_argument("--encoding", type=str, default=None, help="Optional file encoding. If not provided, it will be auto-detected.")

    args = parser.parse_args()

    print(f"Processing file: {args.input}")
    print(f"Output file will be saved to: {args.output}")
    print(f"Language: {args.lang}")
    # Call the processing function
    process_file(args.input, args.output, args.lang, args.encoding)


if __name__ == "__main__":
    main()