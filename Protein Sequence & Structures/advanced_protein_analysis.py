from Bio import Entrez
from Bio import SeqIO
from Bio.SeqUtils import ProtParam

# From protein id, retrieve the protein sequence
def retrieve_prot_sequence(protein_id):
    Entrez.email = "nguyuling@graduate.utm.my"
    
    handle = Entrez.efetch(
        db='protein',
        id=protein_id,
        rettype='fasta',
        retmode='text'
    )
    
    record = SeqIO.read(handle, "fasta")
    handle.close()
    return record

def generate_advanced_analysis(record):
    analysis = ProtParam.ProteinAnalysis(record)
    properties = {
        'Protein ID' : record.id,
        'Sequence' : record.seq,
        'Length' : f"{analysis.length} amino acids",
        'Molecular Weight' : f"{round(analysis.isoelectric_point(), 4)} Da",
        'isoelectric Pouint' : round(analysis.isoelectric_point(), 4),
        'Instability Index' : round(analysis.instability_index(), 4),
        'Aromaticity' : round(analysis.aromaticity(), 4),
        'Gravy' : round(analysis.gravy(), 4)
    }
    return properties

# Function calling
record = retrieve_prot_sequence("P01308")
properties = generate_advanced_analysis(record)
for key, value in properties.items():
    print(f"{key:<25} : {value}")
