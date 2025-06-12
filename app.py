from flask import Flask, render_template, request, jsonify
import os
import sys
import tempfile
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from process import load_model_and_encoder, process_text, extract_text_from_pdf

# Define the SentenceTransformerEmbedder class needed for model loading
class SentenceTransformerEmbedder(BaseEstimator, TransformerMixin):
    """A custom scikit-learn compatible transformer for Sentence-BERT models."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Encodes a list of text documents into sentence embeddings."""
        try:
            return self.model.encode(X.tolist(), show_progress_bar=False)
        except Exception as e:
            print(f"Error during text embedding: {e}", file=sys.stderr)
            return np.array([])

app = Flask(__name__)

# Load the model and encoder when the app starts
model, encoder = load_model_and_encoder(
    'semantic_expense_model.pkl',
    'semantic_target_encoder.pkl'
)

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                # Save the uploaded file temporarily
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, file.filename)
                file.save(temp_path)
                
                # Process the PDF
                text_content = extract_text_from_pdf(temp_path)
                # Clean up
                os.unlink(temp_path)
                os.rmdir(temp_dir)
            else:
                return jsonify({'error': 'Please upload a PDF file'}), 400
        else:
            # Process raw text
            text_content = request.form.get('text', '')
            
        if not text_content:
            return jsonify({'error': 'No content to process'}), 400
            
        # Process the text and get results
        results_df = process_text(text_content, model, encoder)
        
        if results_df.empty:
            return jsonify({'error': 'No transactions found in the content'}), 400
            
        # Convert DataFrame to list of dictionaries for JSON response
        results = results_df.to_dict('records')
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
