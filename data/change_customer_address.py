import pandas as pd
from faker import Faker

df = pd.read_csv('customer_data.csv')
fake = Faker('en_CA')  # Canadian locale

# Create a dictionary mapping each unique tracking number to an address
tracking_to_address = {}
for tracking_num in df['TrackingNumber'].unique():
    address = fake.address()
    # Remove all types of newlines and extra whitespace
    address = ' '.join(address.split())
    # Add country at the end
    address = address + ' Canada'
    tracking_to_address[tracking_num] = address

# Assign addresses based on tracking number
df['Address'] = df['TrackingNumber'].map(tracking_to_address)

df.to_csv('updated_file.csv', index=False)


print("Finished mapping addresses to tracking number ...")

df['Address'] = df['TrackingNumber'].map(tracking_to_address)

print("Finished entering addresses ... ")

df.to_csv('updated_customer_data.csv', index=False)

print("Finished Writing ...")