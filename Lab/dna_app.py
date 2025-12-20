import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# --- App Configuration ---
st.set_page_config(
    page_title="DNA Nucleotide Counter",
    layout="centered"
)

# --- Title and Description ---
st.title("🧬 DNA Nucleotide Count Web App")
st.markdown("""
This app counts the **nucleotide composition** of a query DNA sequence!
""")

# --- Sidebar for Input ---
st.sidebar.header('User Input Features')

# Default DNA sequence for demonstration
default_dna = (
    "ATGCAGAGATAGCAGCGCGACGATAGACAGACAGCATGCATGC"
    "TACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTA"
)

# Text area for user input
sequence_input = st.sidebar.text_area(
    "Enter DNA Sequence:",
    default_dna,
    height=250
)

# Format the input sequence: remove non-alphabetic characters and convert to uppercase
sequence = sequence_input.replace('\n', '').replace('\r', '').upper()

st.header('Input Sequence')
st.code(sequence)

# --- Main Logic: Nucleotide Count Function ---
def DNA_nucleotide_count(seq):
    # Use collections.Counter for easy counting
    d = Counter(seq) 

    # Create a DataFrame for organized output
    df = pd.DataFrame.from_dict(d, orient='index').T
    
    # Ensure all 4 nucleotides are present, filling missing ones with 0
    nucleotides = ['A', 'T', 'G', 'C']
    for n in nucleotides:
        if n not in df.columns:
            df[n] = 0
            
    # Select and order the columns
    df = df[['A', 'T', 'G', 'C']]
    return df

# Calculate the counts
X = DNA_nucleotide_count(sequence)

# --- Display Results ---

# 1. Table Output
st.subheader('1. Nucleotide Count Table')
st.dataframe(X)

# 2. Bar Chart Output
st.subheader('2. Nucleotide Count Bar Chart')
st.info('Only A, T, G, C nucleotides are included in the chart.')

# Prepare data for plotting
df_plot = X.iloc[0] # Get the first (and only) row
fig, ax = plt.subplots()
ax.bar(df_plot.index, df_plot.values, color=['red', 'green', 'blue', 'orange'])
ax.set_xlabel('Nucleotide')
ax.set_ylabel('Count')
ax.set_title('Nucleotide Composition')
st.pyplot(fig) # Display the Matplotlib figure in Streamlit

# --- Simple Statistics ---
st.subheader('3. Simple Statistics')
total_length = len(sequence)
st.write(f"**Total Sequence Length:** {total_length} bases")
gc_content = ((X['G'].sum() + X['C'].sum()) / total_length) * 100
st.write(f"**GC-Content:** {gc_content:.2f}%")