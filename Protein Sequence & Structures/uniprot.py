import requests

# Get the Uniprot erquest URL
api_endpoint = "https://www.uniprot.org/uniprot/"
protein = "P01308"
req_url = api_endpoint + protein + ".fasta"

# Request teh protein from Uniprot & print its FASTA sequence
response = requests.get(req_url)
if response.status_code == 200:
    print(response.text)
else:
    print("Something wrong!")
    print(response.text)

# Save the FASTA sequence into a file
filename = protein + ".fasta"
if response.status_code == 200:
    with open(filename, 'w') as file:
        file.write(response.text)
    print(f"Downloaded {protein} to {filename}")
else:
    print("Something wrong!")
    print(response.status_code)