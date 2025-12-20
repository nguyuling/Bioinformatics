from Bio import Entrez
from Bio import SeqIO
from collections import Counter
import pandas as pd

#! From entrez.py
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
    
    return protein_sequence

def generate_basic_analysis(protein_seq):
    
    # Count every unique amino acid character & store a dictionary of tuples (aa, count)
    aa_count = Counter(protein_seq) 
    # Count the length of the protein sequence
    total = len(protein_seq)
    # Calculate the percentage of each amino acid & store a tuples (aa, percentage)
    aa_percent = {aa: (count / total) * 100 for aa, count in aa_count.items()}
    
    # Create a DataFrame
    df = pd.DataFrame({
        'Amino Acid' : list(aa_count.keys()),
        'Count' : list(aa_count.values()),
        'Percentage' : [aa_percent[aa] for aa in aa_count.keys()]
    })
    
    # Sort the DataFrame by 'Count' column in descending order
    df_sorted = df.sort_values(by='Count', ascending=False)
    return df_sorted

# Function calling
protein_sequence = retrieve_prot_sequence("P01309")
analysis_df = generate_basic_analysis(protein_sequence)
print(analysis_df.to_string(index=False)) #so it won't print the index no. column