from Bio import Entrez
from Bio import SeqIO

def retrieve_prot_sequence(protein_id):
    
    # Set email (mandatory for Entrez)
    Entrez.email = "nguyuling@gmail.com"
    
    # Fetch protein
    handle = Entrez.efetch(
        db = 'protein',
        id = protein_id,
        rettype= "fasta",
        retmmode = "text"
    )
    
    # Read the FASTA record from the handle
    record = SeqIO.read(handle, 'fasta')
    handle.close()
    
    # Extract the sequence string
    protein_sequence = str(record.seq)
    
    # Print the formatted data
    print(f"Protein ID: {record.id}")
    print(f"Description: {record.description}")
    print(f"Length: {len(record.seq)} amino acids")
    print("Sequence:")
    print(protein_sequence[:80] + "...")
    
    # Save to file
    filename = protein_id + ".fasta"
    with open(filename, 'w') as file:
        SeqIO.write(record, file, "fasta")
    print(f"Successfully saved protein {protein_id} to {filename}")
    
    return protein_sequence

# Function calling
protein_sequence = retrieve_prot_sequence("P01309")
    