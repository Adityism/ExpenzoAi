import pandas as pd
import numpy as np
import re
import joblib
import fitz  # PyMuPDF library
import argparse
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer

# --- 1. CRITICAL: Re-define the Custom Transformer Class ---
# This class definition is ESSENTIAL for joblib to load your custom pipeline.
# It must be present in the script that loads the model.
class SentenceTransformerEmbedder(BaseEstimator, TransformerMixin):
    """A custom scikit-learn compatible transformer for Sentence-BERT models."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        # The fit method is not needed for a pre-trained model, so we just return self.
        return self

    def transform(self, X):
        """Encodes a list of text documents into sentence embeddings."""
        # The input X is expected to be a pandas Series.
        # We convert it to a list for the encoder.
        try:
            return self.model.encode(X.tolist(), show_progress_bar=False)
        except Exception as e:
            print(f"Error during text embedding: {e}", file=sys.stderr)
            return np.array([])

# --- 2. Core Processing Functions ---

def load_model_and_encoder(model_path, encoder_path):
    """Loads the trained model pipeline and target encoder from disk."""
    try:
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        print("✅ Model and encoder loaded successfully.")
        return model, encoder
    except FileNotFoundError:
        print(f"🚨 FATAL ERROR: Model or encoder file not found.", file=sys.stderr)
        print(f"Ensure '{model_path}' and '{encoder_path}' are in the same directory as this script.", file=sys.stderr)
        sys.exit(1) # Exit the script if files are not found.
    except Exception as e:
        print(f"🚨 FATAL ERROR: An error occurred while loading model files: {e}", file=sys.stderr)
        sys.exit(1)


def predict_category(note, amount, model, encoder):
    """Uses the loaded model to predict a category for a single transaction."""
    # The model pipeline expects a DataFrame as input.
    input_data = pd.DataFrame({'Clean_Note': [note], 'Amount': [float(amount)]})
    # The [0] at the end gets the single prediction from the output array.
    predicted_label = model.predict(input_data)[0]
    # The encoder transforms the numeric label back to a string category name.
    predicted_category = encoder.inverse_transform([predicted_label])[0]
    return predicted_category


def extract_text_from_pdf(pdf_path):
    """
    Extracts all text content from a given PDF file, preserving the
    line structure by sorting text blocks.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc.pages():
            # The 'sort=True' argument is the key fix. It reconstructs the
            # text in a logical, top-to-bottom order, preserving lines.
            full_text += page.get_text(sort=True)
            full_text += "\n" # Add a newline after each page
        doc.close()
        return full_text
    except Exception as e:
        print(f"🚨 ERROR: Could not read PDF file '{pdf_path}'. Reason: {e}", file=sys.stderr)
        return None


def process_text(text_content, model, encoder):
    """
    Finds, extracts, and categorizes transactions from a block of text.
    """
    # Split the text content into lines and process each non-empty line
    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
    
    # This regular expression is designed to find lines that describe a transaction.
    # It looks for text followed by a number (integer, with commas, or decimal).
    # Group 1: (.*?) - Captures the transaction description (non-greedy).
    # Group 2: ([\d,]+\.?\d*) - Captures the amount.
    transaction_pattern = re.compile(r"^(.*?)\s+([\d,]+\.?\d*)$")

    matches = []
    for line in lines:
        match = transaction_pattern.match(line)
        if match:
            matches.append(match.groups())

    if not matches:
        print("No potential transactions found in the provided text.")
        return pd.DataFrame()

    print(f"🔍 Found {len(matches)} potential transactions. Categorizing...")

    results = []
    for note, amount_str in matches:
        # Clean the extracted text and amount.
        note = note.strip().replace('\n', ' ')
        amount = float(amount_str.replace(',', ''))

        # Get the category prediction from our trained model.
        category = predict_category(note, amount, model, encoder)
        results.append({
            "Transaction": note,
            "Amount": amount,
            "Predicted Category": category
        })

    return pd.DataFrame(results)


# --- 3. Main Execution Block ---

def main():
    """Main function to run the command-line interface."""
    # Set up the argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(
        description="Categorize expenses from a PDF or raw text using a trained ML model.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting.
    )
    # Create a mutually exclusive group: user must provide either a file or text, but not both.
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        type=str,
        help="Path to the PDF file containing transactions."
    )
    group.add_argument(
        "--text",
        type=str,
        help="A string of raw text containing transactions."
    )
    args = parser.parse_args()

    # --- Load Model ---
    model, encoder = load_model_and_encoder(
        'semantic_expense_model.pkl',
        'semantic_target_encoder.pkl'
    )

    # --- Process Input ---
    input_text = ""
    if args.file:
        print(f"📖 Reading text from PDF file: {args.file}")
        input_text = extract_text_from_pdf(args.file)
        if input_text is None:
            sys.exit(1) # Exit if PDF reading failed.
    elif args.text:
        print("✍️  Processing raw text input.")
        input_text = args.text

    # --- Get and Display Results ---
    if input_text:
        results_df = process_text(input_text, model, encoder)
        if not results_df.empty:
            print("\n" + "="*50)
            print("                CATEGORIZED TRANSACTIONS")
            print("="*50)
            print(results_df.to_string(index=False))
            print("="*50)
    else:
        print("No text could be processed.")


if __name__ == "__main__":
    main()
