import json

# Load the data card
with open('jssp_llm_format_120k_datacard.json', 'r') as file:
    data_card = json.load(file)

# Accessing basic information
print("Dataset Name:", data_card['Dataset Name'])
print("Number of Entries:", data_card['Number of Entries'])
print("Number of Fields:", data_card['Number of Fields'])

# Accessing field information
fields = data_card['Fields']
for field, info in fields.items():
    print(f"Field: {field}")
    print(f"  Type: {info['Type']}")
    print(f"  Number of Unique Values: {info['Number of Unique Values']}")
    print(f"  Sample Values: {info['Sample Values']}")

# Accessing sample entries
sample_entries = data_card['Sample Entries']
print("Sample Entries:")
for entry in sample_entries:
    print(entry)
