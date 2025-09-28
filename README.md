# Expense Categorization

This project is an intelligent expense categorization tool that leverages machine learning and semantic analysis to automatically classify and analyze your financial transactions. It provides a user-friendly web interface for uploading transaction data, viewing categorized results, and giving feedback to improve the model.

## Features
- **Automatic Expense Categorization:** Uses a trained semantic model to classify expenses into categories.
- **PDF and CSV Support:** Upload your transaction data in PDF or CSV format.
- **Interactive Web Interface:** Built with Flask, featuring a modern UI for easy interaction.
- **Feedback Logging:** User feedback is logged to continuously improve the categorization model.
- **Large Transaction Detection:** Identifies and highlights unusually large transactions.

## Project Structure
```
├── app.py                      # Main Flask application
├── embedder.py                 # Embedding logic for semantic model
├── feedback_log.csv            # Stores user feedback for model improvement
├── large_transactions.pdf      # Example output for large transactions
├── pdf.py                      # PDF parsing utilities
├── process.py                  # Core processing and categorization logic
├── requirements.txt            # Python dependencies
├── semantic_expense_model.pkl  # Trained ML model for categorization
├── semantic_target_encoder.pkl # Label encoder for categories
├── utils.py                    # Utility functions
└── templates/
    ├── index.html              # Main web interface
    ├── index2.html             # Alternate interface
    ├── logo.png                # Project logo
    └── styles.css              # Custom styles
```

## Getting Started
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <project-directory>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```
4. **Open your browser:**
   Visit `http://localhost:5000` to use the app.

## Requirements
- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- PyPDF2

(See `requirements.txt` for the full list.)

## Usage
- Upload your transaction file (PDF/CSV) via the web interface.
- View categorized expenses and large transactions.
- Provide feedback to help improve the model.



---

**Developed by Aditya Goyal**
