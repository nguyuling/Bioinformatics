from Bio import Entrez
from Bio import SeqIO
from Bio.SeqUtils import ProtParam
import pandas as pd

def retrieve_prot_sequence(proteins):
    Entrez.email = "nguyuling@graduate.utm.my"
    records = []
    processed = 0
    current = 0
    
    # Fetch all protein sequences and append them
    while processed < len(proteins):
        print(f"Fetching protein {proteins[current]} ...")
        handle = Entrez.efetch(
            db = 'protein',
            id = proteins[current],
            rettype = 'fasta',
            retmode = 'text'
        )
        
        record = SeqIO.read(handle, 'fasta')
        records.append(record)
        processed += 1
        current += 1
    
    handle.close()
    
    filename = 'insulin_family.fasta'
    with open(filename, 'w') as file:
        for record in records:
            file.write(record.format('fasta'))
    
    return records

# Function to analyse a single protein record
def analyse_multiple_proteins(records):
    all_proteins_properties = []
    
    for record in records:
        analysis = ProtParam.ProteinAnalysis(str(record.seq))
        properties = {
            'Protein ID' : record.id,
            'Sequence Length' : f"{analysis.length} amino acids",
            'Molecular Weight' : f"{round(analysis.molecular_weight(), 4)} Da",
            'Isoelectric Point' : round(analysis.isoelectric_point(), 4),
            'Instability Index' : round(analysis.instability_index(), 4),
            'Aromaticity' : round(analysis.aromaticity(), 4),
            'Gravy' : round(analysis.gravy(), 4)
        }
        all_proteins_properties.append(properties)
        
    properties_df = pd.DataFrame(all_proteins_properties)
    return properties_df

# Function calling
proteins = ['P01308', 'P01322', 'P01325', 'P01329']
records = retrieve_prot_sequence(proteins)
properties_df = analyse_multiple_proteins(records)
print(properties_df.to_string(index=False))