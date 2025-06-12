# ==============================================================================
#  PDF Transaction Generator
#  Description: This script generates a PDF file with 500 random transactions
#               to be used as a test dataset for the expense categorizer.
#
#  Instructions:
#  1. Install the dependency: pip install reportlab
#  2. Run this script. It will create a file named 'large_transactions.pdf'.
# ==============================================================================

import random
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

print("🚀 Starting PDF generation...")

# --- Data for Random Transaction Generation ---
# We'll use location-specific examples for Agra, India, to make it realistic.
transaction_templates = {
    "Food": [
        "Zomato food delivery", "Swiggy Instamart groceries", "Lunch at The Salt Cafe",
        "Coffee at Starbucks, Fatehabad Road", "Dinner at Pinch of Spice",
        "Weekly vegetable shopping, Sadar Bazaar", "Domino's Pizza order",
        "Brijwasi Sweets purchase", "GMB restaurant bill"
    ],
    "Transportation": [
        "Ola ride to work", "Uber to TDI Mall", "Auto fare to Agra Cantt",
        "Monthly petrol fill-up at HP Petrol Pump", "Parking fee at Sanjay Place",
        "Agra City bus ticket", "EV bike rental charge"
    ],
    "Shopping": [
        "Myntra clothing order", "Amazon electronics purchase", "Shoes from Nike store, Civil Lines",
        "Book from Crossword bookstore", "Home decor from IKEA online", "Big Bazaar household items",
        "Mobile accessory from Gaffar Market"
    ],
    "Bills": [
        "Vodafone Postpaid phone bill", "Airtel Xstream Wifi payment", "Torrent Power electricity bill",
        "Credit card statement payment", "UP Jal Nigam water bill", "Gas cylinder booking"
    ],
    "Entertainment": [
        "Movie tickets at PVR, TDI Mall", "Netflix India subscription", "Spotify Premium yearly plan",
        "Bowling at Timezone, Sarv Multiplex", "Entry ticket to Taj Mahal",
        "BookMyShow event booking"
    ],
    "Health": [
        "Apollo Pharmacy purchase", "Doctor consultation fee at Synergy Plus",
        "Pathkind Labs blood test", "Online medicine order from PharmEasy",
        "Gym membership monthly fee"
    ],
    "Education": [
        "Udemy course fee", "Coursera subscription payment", "Stationery purchase from local shop",
        "Online webinar ticket", "Technical book purchase"
    ],
    "Investment": [
        "SIP installment for Mutual Fund", "Stock purchase via Groww",
        "Contribution to PPF account"
    ]
}

def generate_random_transaction():
    """Selects a random category and item, and generates a random amount."""
    category = random.choice(list(transaction_templates.keys()))
    note = random.choice(transaction_templates[category])
    
    # Generate amount based on category to make it more realistic
    if category in ["Bills", "Shopping", "Investment"]:
        amount = round(random.uniform(500, 7500), 2)
    elif category == "Transportation":
        amount = round(random.uniform(50, 800), 2)
    else: # Food, Health, Entertainment, etc.
        amount = round(random.uniform(100, 2500), 2)
        
    return f"{note} {amount}"

def create_transaction_pdf(filename, num_transactions):
    """Creates a PDF document with a specified number of transactions."""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Set up text object
    text_object = c.beginText()
    text_object.setFont("Helvetica", 10)
    # Start text from top-left, with a margin
    x_margin = 0.5 * inch
    y_margin = 0.5 * inch
    x, y = x_margin, height - y_margin
    text_object.setTextOrigin(x, y)
    
    line_height = 14 # Points
    
    text_object.textLine(f"--- Transaction Statement (Generated for Testing) ---")
    text_object.textLine(" ") # Add a blank line
    
    for i in range(num_transactions):
        # Check if we need to start a new page
        if y < y_margin + line_height:
            c.drawText(text_object)
            c.showPage() # Finalize current page and start a new one
            text_object = c.beginText()
            text_object.setFont("Helvetica", 10)
            x, y = x_margin, height - y_margin
            text_object.setTextOrigin(x, y)

        transaction_line = generate_random_transaction()
        text_object.textLine(transaction_line)
        y -= line_height

    c.drawText(text_object)
    c.save()

if __name__ == "__main__":
    output_file = "large_transactions.pdf"
    num_transactions = 500
    
    try:
        create_transaction_pdf(output_file, num_transactions)
        print(f"✅ Successfully generated {num_transactions} transactions in {output_file}")
    except Exception as e:
        print(f"❌ Error generating PDF: {str(e)}")


