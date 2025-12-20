import streamlit as st
import pandas as pd
from Bio import Entrez
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

#! function retrieve_data
def retrieve_data(protein_id: str):
    clean_id = protein_id.strip()
    
    if not clean_id:
        return None # Return None if input is empty

    try:
        # Step 1: Search NCBI to find the native accession ID (Fix for HTTP 400 Error)
        st.info(f"🔎 Searching NCBI for ID: {clean_id}...")
        
        Entrez.email = "nguyuling@gmail.com" # Setting email here to ensure it's available for Entrez.esearch
        search_handle = Entrez.esearch(
            db="protein",
            # Use a robust search term that finds the NCBI ID associated with the input
            term=clean_id + " AND RefSeq[filter]", 
            retmax="1",
            tool="BiopythonLabApp"
        )
        search_record = Entrez.read(search_handle)
        search_handle.close()

        if not search_record["IdList"]:
            st.error(f"Could not find a corresponding NCBI accession for ID: {clean_id}")
            return None
        
        ncbi_id = search_record["IdList"][0] 
        st.info(f"Found NCBI Accession: {ncbi_id}. Starting fetch.")
        
        fetch_handle = Entrez.efetch(
            db="protein",
            id=ncbi_id,
            rettype="fasta",
            retmode="text",
            tool="BiopythonLabApp" 
        )
        record = SeqIO.read(fetch_handle, "fasta")
        fetch_handle.close()
        
        # write the record to a fasta file
        filename = f"{ncbi_id}.fasta"
        SeqIO.write(record, filename, "fasta")
        print(f"Successfully retrieved and saved protein ID {ncbi_id} to {filename}") 
        return record
        
    except Exception as e:
        st.error("Retrieval Failed! An error occurred during the search or fetch process.")
        st.code(f"Error Details: {e}")
        print(f"An error occurred while retrieving or writing the sequence: {e}") 

#! function get_basic_analysis
def get_basic_analysis(record):
    sequence = str(record.seq)
    seq_len = len(sequence)
    seq_analysis = ProteinAnalysis(sequence)
    aa_comp = seq_analysis.count_amino_acids()
    mol_weight = seq_analysis.molecular_weight()
    iso_point = seq_analysis.isoelectric_point()
    return seq_len, aa_comp, mol_weight, iso_point

#! app layout
st.set_page_config(layout="wide")
st.title("Lab 1 - NGU YU LING (A23CS0149)")

uniprot_id = st.text_input("Enter a Uniprot ID", value="P04637")
retrieve = st.button("Retrieve")

#! functions calling
if retrieve: # check if the button is clicked
    
    with st.spinner("Retrieving data from NCBI..."):
        record = retrieve_data(uniprot_id)
    
    if record: # if the retrieval was successful
        st.success(f"Successfully retrieved protein record for ID: **{record.id}**")
        sequence_length, aa_composition, molecular_weight, isoelectric_point = get_basic_analysis(record)
        
        st.header("Raw Record Information")
        col1, col2 = st.columns(2)
        with col1: 
            st.subheader("Field")
            st.write("Raw Record Returned:")
            st.write("Sequence:")
            st.write("Description:")
            st.write("Name:")
        with col2:
            st.subheader("Value")
            st.code(str(record).split('\n')[0] + '...')
            st.code(str(record.seq))
            st.write(record.description)
            st.write(record.name)
        
        st.header("Basic Analysis Result")
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Metric")
            st.write("Sequence Length")
            st.write("Molecular Weight")
            st.write("Isoelectric Point")
        with col4:
            st.subheader("Value")
            st.write(f"**{sequence_length}** amino acids")
            st.write(f"**{molecular_weight:,.2f}** Da")
            st.write(f"**{isoelectric_point:.3f}**") 
        
        st.markdown("---")
        st.write("Amino Acid Composition")
        aa_df = pd.DataFrame(
            data = aa_composition.items(),
            columns = ["Amino Acids", "Count"]
        ).sort_values(by="Count", ascending=False)
        st.dataframe(aa_df, use_container_width = True)
    
    else: # if the retrieval failed
        st.warning("Please check the ID entered or see error message above.")