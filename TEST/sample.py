def calculate_discount(price):
    """Apply 10% discount to price"""
    return price * 0.9

def send_email(to, subject, body):
    """Send email to a user"""
    print(f"Sending email to {to}")

def calculate_tax(amount):
    """Calculate 18% GST tax"""
    return amount * 0.18

def get_user(user_id):
    """Fetch user from database"""
    return {"id": user_id, "name": "John"}

def process_payment(amount, user_id):
    """Process payment for a user"""
    tax = calculate_tax(amount)
    discount = calculate_discount(amount)
    final = amount + tax - discount
    return {"user": user_id, "total": final}

def send_invoice(user_id, amount):
    """Send invoice email after payment"""
    user = get_user(user_id)
    send_email(user["name"], "Invoice", f"Your total is {amount}")