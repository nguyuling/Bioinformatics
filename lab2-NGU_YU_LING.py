import streamlit as st
from Bio.PDB import PDBList, PDBParser
from Bio.PDB.Structure import Structure
import numpy as np
import py3Dmol
import os
import tempfile
import pandas as pd

# Retrieve and Parse the Protein Structure
@st.cache_data
def get_protein_structure(protID):

    # Create a temporary directory to download the PDB file to
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize PDB tools
        pdb_list = PDBList()
        parser = PDBParser()

        try:
            # Download the PDB file (returns the local path)
            protID_upper = protID.upper()
            file_path = pdb_list.retrieve_pdb_file(
                protID_upper, 
                pdir=temp_dir, 
                file_format='pdb'
            )
            
            # Parse the structure from the file
            structure = parser.get_structure(protID_upper, file_path)
            return structure
            
        except Exception as e:
            st.error(f"Error retrieving/parsing PDB ID '{protID}'. Check if the ID is valid.")
            st.error(f"Detailed error: {e}")
            return None

# Calculate Structural Properties and Create 3D View ---
def get_structure_info(prot_structure):
    
    #Prepare Atom Coordinates and Mass
    atoms = [atom for atom in prot_structure.get_atoms()]
    coords = np.array([atom.get_coord() for atom in atoms])
    masses = np.array([atom.mass for atom in atoms])
    total_mass = np.sum(masses)
    
    if total_mass == 0:
        return {'com': "N/A (No mass data)", 'Rg': "N/A (No mass data)", '3dview': None}
        
    # Calculate Center of Mass (COM)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    com_formatted = f"({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f}) Å"

    # Calculate Radius of Gyration (Rg)
    distances = coords - com
    dist_sq = np.sum(distances**2, axis=1)
    rg_sq = np.sum(masses * dist_sq) / total_mass
    Rg = np.sqrt(rg_sq)
    Rg_formatted = f"{Rg:.3f} Å"
    
    # Create py3Dmol View
    view_component = create_py3dmol_view(prot_structure)    
    return {
        'com': com_formatted,
        'Rg': Rg_formatted,
        '3dview': view_component
    }


def create_py3dmol_view(prot_structure):
    """Creates an interactive 3D viewer component for Streamlit."""
    
    # Extract the PDB ID from the structure object for py3Dmol query
    pdb_id = prot_structure.get_id()
    
    # Initialize the py3Dmol view
    view = py3Dmol.view(
        query=f'pdb:{pdb_id}', 
        width=800, 
        height=500,
        viewer=(500, 500)
    )
    
    # Set display style
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return st.py3dmol(view, height=500, width=800)


# Streamlit Application Layout
def app():
    st.set_page_config(
        page_title="Protein Structure Analyzer",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 Protein Structure Analyzer (PDB)")
    st.markdown("Enter a valid 4-character PDB ID to retrieve and analyze the protein structure. Calculations include the Center of Mass (COM) and Radius of Gyration ($R_g$).")
    
    # Input field and button
    col1, col2 = st.columns([1, 2])
    
    with col1:
        protein_id = st.text_input(
            "Enter PDB ID (e.g., 4HHB)", 
            value="4HHB"
        )
        
        if st.button("Analyze Structure"):
            if not protein_id or len(protein_id) != 4:
                st.warning("Please enter a valid 4-character PDB ID.")
            else:
                st.session_state['run_analysis'] = protein_id.upper()
        
    if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
        protID = st.session_state['run_analysis']
        
        # Retrieve structure
        with st.spinner(f"Downloading and parsing structure for PDB ID: {protID}..."):
            structure = get_protein_structure(protID)
        
        if structure:
            st.success(f"Structure **{protID}** successfully loaded!")
            
            # Analyze structure
            results = get_structure_info(structure)
            
            st.subheader(f"Structural Information for {protID}")
            
            # Display metrics
            col_com, col_rg = st.columns(2)
            
            with col_com:
                st.metric(
                    label="Center of Mass (COM) [Å]", 
                    value=results['com'],
                    help="The mass-weighted average position of all atoms."
                )
            
            with col_rg:
                st.metric(
                    label="Radius of Gyration ($R_g$) [Å]", 
                    value=results['Rg'],
                    help="A measure of the compactness or size of the protein structure."
                )

            # Display 3D View
            st.subheader("3D Visualization")
            results['3dview'] # This displays the py3Dmol component

if __name__ == "__main__":
    app()