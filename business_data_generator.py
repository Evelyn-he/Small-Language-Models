import csv
import random
from collections import defaultdict
from pathlib import Path

CUSTOMER_DATABASE_PATH = Path(__file__).parent / "data" / "customer_data.csv"
BUSINESS_DATABASE_PATH = Path(__file__).parent / "data" / "business_data.csv"

def generate_product_catalog(input_file, output_file):
    products = {}
    
    with open(input_file, 'r', encoding='iso-8859-1') as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            stock_code = row['StockCode']
            description = row['Description']
            unit_price = float(row['UnitPrice'])
            quantity = int(row['Quantity'])

            if quantity > 0 and description != '':
                if stock_code not in products:
                    products[stock_code] = {
                        'description': description,
                        'price': unit_price
                    }
                elif products[stock_code]['price'] == 0 and unit_price != 0:
                    products[stock_code]['price'] = unit_price
    
    with open(output_file, 'w', newline='', encoding='iso-8859-1') as outfile:
        fieldnames = ['StockCode', 'Description', 'UnitPrice', 'StockQuantity']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for stock_code in sorted(products.keys()):
            writer.writerow({
                'StockCode': stock_code,
                'Description': products[stock_code]['description'],
                'UnitPrice': products[stock_code]['price'],
                'StockQuantity': int(random.triangular(0, 200, 30))
            })

if __name__ == "__main__":
    random.seed(10)

    generate_product_catalog(CUSTOMER_DATABASE_PATH, BUSINESS_DATABASE_PATH)