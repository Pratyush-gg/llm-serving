"""
Synthetic dataset generator for JSON extraction adapter.
Generates 600 training and 60 held-out examples strictly adhering to ExtractionSchema:
  user: str
  order_id: str
  amount: float
"""

import json
import random
import os

SEED = 42
random.seed(SEED)

FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Casey",
    "Jamie", "Riley", "Avery", "Dakota", "Reese", "Cameron", "Quinn", "Skyler",
    "Logan", "Rowan", "Jesse", "Kendall", "Harper", "Finley", "Hayden", "Emerson",
    "Priya", "Rohan", "Ananya", "Vikram", "Mei", "Chen", "Yuki", "Kenji",
    "Carlos", "Sofia", "Elena", "Mateo", "Fatima", "Tariq", "Amara", "Kwame"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores"
]

ID_PREFIXES = ["ORD", "INV", "TXN", "REF", "PO", "SO", "REC", "ORDER", "CART"]
ID_SEPARATORS = ["-", "_", "#", ""]

TEMPLATES = [
    # E-commerce confirmation
    "Thank you for shopping with us, {user}! Your order {order_id} has been processed for a total charge of ${amount:.2f}.",
    "Order Confirmation: Customer {user}, reference number {order_id}. Total billed to your card: ${amount:.2f}.",
    "Hi {user}, we received your order {order_id}. The final amount due is ${amount:.2f}.",
    
    # Billing / Invoice
    "INVOICE SUMMARY\nClient: {user}\nInvoice No: {order_id}\nTotal Due: ${amount:.2f}\nStatus: Paid",
    "Payment of ${amount:.2f} received from {user} for invoice {order_id}. Thank you for your business.",
    "Billing notification for account {user}. Transaction {order_id} was successfully charged for ${amount:.2f}.",

    # Support / Refund ticket
    "Support Ticket: Customer {user} requested a refund for order {order_id} totaling ${amount:.2f}.",
    "Agent notes: Spoke with {user} regarding disputed charge of ${amount:.2f} under transaction ID {order_id}.",
    "Customer Service Log: Inquired about order {order_id} placed by {user}. Confirmed package value is ${amount:.2f}.",

    # Shipping / Logistics
    "Shipping dispatch notification: Order {order_id} belonging to {user} has shipped. Package declaration value: ${amount:.2f}.",
    "Delivery update for {user}: Package {order_id} is out for delivery. COD amount to collect: ${amount:.2f}.",

    # Informal chat / email
    "Hey team, can someone check {order_id}? {user} mentioned they were overbilled and only expected to pay ${amount:.2f}.",
    "Quick note from accounting: please record payment of ${amount:.2f} against order {order_id} from {user}.",
    "{user} just placed order {order_id} with checkout total ${amount:.2f} via PayPal.",
    "Transaction alert: ${amount:.2f} debited for purchase {order_id} by user {user}."
]

def generate_sample():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    user = f"{first} {last}"
    
    prefix = random.choice(ID_PREFIXES)
    sep = random.choice(ID_SEPARATORS)
    num = random.randint(10000, 999999)
    order_id = f"{prefix}{sep}{num}"
    
    amount = round(random.uniform(5.0, 4999.0), 2)
    template = random.choice(TEMPLATES)
    text = template.format(user=user, order_id=order_id, amount=amount)
    
    prompt = (
        f"Extract the user, order_id, and amount from the following text as a JSON object:\n\n"
        f"Text: \"{text}\"\n\n"
        f"Output format:\n"
        f'{{"user": "<name>", "order_id": "<id>", "amount": <number>}}'
    )
    gold_json = {
        "user": user,
        "order_id": order_id,
        "amount": amount
    }
    completion = json.dumps(gold_json)
    
    return {
        "prompt": prompt,
        "input_text": text,
        "completion": completion,
        "gold_json": gold_json
    }

def main():
    total_needed = 660
    train_count = 600
    holdout_count = 60
    
    unique_samples = []
    seen_texts = set()
    
    while len(unique_samples) < total_needed:
        sample = generate_sample()
        if sample["input_text"] not in seen_texts:
            seen_texts.add(sample["input_text"])
            unique_samples.append(sample)
            
    train_samples = unique_samples[:train_count]
    holdout_samples = unique_samples[train_count:]
    
    os.makedirs("data", exist_ok=True)
    
    train_file = os.path.join("data", "json_train.jsonl")
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")
            
    holdout_file = os.path.join("data", "json_holdout.jsonl")
    with open(holdout_file, "w", encoding="utf-8") as f:
        for s in holdout_samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Generated {len(train_samples)} training records -> {train_file}")
    print(f"Generated {len(holdout_samples)} holdout records -> {holdout_file}")

if __name__ == "__main__":
    main()
