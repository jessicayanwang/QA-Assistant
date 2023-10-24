import PyPDF2
import argparse
import pickle


def extract_text_from_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        # Initialize a PdfReader object
        pdf_reader = PyPDF2.PdfReader(file)

        # Initialize a string to store the extracted text
        extracted_text = ""

        # Loop through all pages in the PDF file and extract text
        for page in pdf_reader.pages:
            extracted_text += page.extract_text()

    # Save the extracted text to a text file
    with open("context.txt", "w", encoding="utf-8") as output_file:
        output_file.write(extracted_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=str, default='paper', help='pdf title without the extension')
    args = parser.parse_args()
    # Convert pdf file to a text file
    pdf_file_path = args.context + '.pdf'
    extract_text_from_pdf(pdf_file_path)
    with open('context.txt', 'r') as f:
        context = f.read()

    with open("context.pk1", 'wb') as file:
        pickle.dump(context, file)