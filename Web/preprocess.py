import PyPDF2
import argparse
import pickle


def extract_text_from_pdf(pdf_file):
    # Initialize a PdfReader object
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    # Initialize a string to store the extracted text
    extracted_text = ""

    # Loop through all pages in the PDF file and extract text
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()

    # Save the extracted text to a text file
    with open("Web/context.txt", "w", encoding="utf-8") as output_file:
        output_file.write(extracted_text)

    with open("Web/context.pk1", 'wb') as file:
        pickle.dump(extracted_text, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default='Web/paper', help='pdf title without the extension')
    args = parser.parse_args()
    # Convert pdf file to a text file
    pdf_file_path = args.context + '.pdf'
    extract_text_from_pdf(pdf_file_path)