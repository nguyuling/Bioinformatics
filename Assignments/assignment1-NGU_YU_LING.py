import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple, List

# Set page config
st.set_page_config(
    page_title="Sequence Alignment Tool",
    layout="wide",
)

st.title("Sequence Alignment Tool")
st.markdown("Perform global (Needleman-Wunsch) and local (Smith-Waterman) alignment on DNA/Protein sequences")


#! Base class for sequence alignment algorithms
class SequenceAligner:
    
    def __init__(self, seq1: str, seq2: str, match: int, mismatch: int, gap: int):
        self.seq1 = seq1.upper()
        self.seq2 = seq2.upper()
        self.match = match
        self.mismatch = mismatch
        self.gap = gap
        self.matrix = None
        self.traceback_matrix = None
        self.score = None
        self.aligned_seq1 = None
        self.aligned_seq2 = None
        self.traceback_path = None
    
    # Calculate similarity score between two characters
    def similarity_score(self, char1: str, char2: str) -> int:
        if char1 == char2:
            return self.match
        else:
            return self.mismatch
    
    # Initialize the scoring matrix
    def initialize_matrix(self) -> np.ndarray:
        raise NotImplementedError
    
    def fill_matrix(self):
        """Fill the scoring matrix"""
        raise NotImplementedError
    
    def traceback(self):
        """Perform traceback to get aligned sequences"""
        raise NotImplementedError
    
    # Perform alignment
    def align(self) -> Tuple[str, str, int, List]:
        self.initialize_matrix()
        self.fill_matrix()
        self.traceback()
        return self.aligned_seq1, self.aligned_seq2, self.score, self.traceback_path


#! Global alignment using Needleman-Wunsch algorithm
class NeedlemanWunsch(SequenceAligner):
    
    # Initialize matrix for global alignment
    def initialize_matrix(self):
        rows = len(self.seq1) + 1
        cols = len(self.seq2) + 1
        self.matrix = np.zeros((rows, cols), dtype=int)
        self.traceback_matrix = [[None for _ in range(cols)] for _ in range(rows)]
        
        # Initialize first row and column
        for i in range(rows):
            self.matrix[i, 0] = i * self.gap
            self.traceback_matrix[i][0] = 'U'  # Up
        
        for j in range(cols):
            self.matrix[0, j] = j * self.gap
            self.traceback_matrix[0][j] = 'L'  # Left
    
    # Fill the scoring matrix
    def fill_matrix(self):    
        rows = len(self.seq1) + 1
        cols = len(self.seq2) + 1
        
        for i in range(1, rows):
            for j in range(1, cols):
                # Calculate scores for three options
                match_score = self.matrix[i-1, j-1] + self.similarity_score(self.seq1[i-1], self.seq2[j-1])
                delete_score = self.matrix[i-1, j] + self.gap
                insert_score = self.matrix[i, j-1] + self.gap
                
                # Choose the maximum score
                max_score = max(match_score, delete_score, insert_score)
                self.matrix[i, j] = max_score
                
                # Track the direction
                if max_score == match_score:
                    self.traceback_matrix[i][j] = 'D' # diagonal
                elif max_score == delete_score:
                    self.traceback_matrix[i][j] = 'U' # up
                else:
                    self.traceback_matrix[i][j] = 'L' # left
        
        self.score = int(self.matrix[-1, -1])
    
    # Traceback from bottom-right to top-left
    def traceback(self):
        aligned_seq1 = []
        aligned_seq2 = []
        path = []
        
        i = len(self.seq1)
        j = len(self.seq2)
        
        while i > 0 or j > 0:
            path.append((i, j))
            
            if i == 0:
                aligned_seq1.append('-')
                aligned_seq2.append(self.seq2[j-1])
                j -= 1
            elif j == 0:
                aligned_seq1.append(self.seq1[i-1])
                aligned_seq2.append('-')
                i -= 1
            else:
                direction = self.traceback_matrix[i][j]
                if direction == 'D':
                    aligned_seq1.append(self.seq1[i-1])
                    aligned_seq2.append(self.seq2[j-1])
                    i -= 1
                    j -= 1
                elif direction == 'U':
                    aligned_seq1.append(self.seq1[i-1])
                    aligned_seq2.append('-')
                    i -= 1
                else:  # 'L'
                    aligned_seq1.append('-')
                    aligned_seq2.append(self.seq2[j-1])
                    j -= 1
        
        self.aligned_seq1 = ''.join(reversed(aligned_seq1))
        self.aligned_seq2 = ''.join(reversed(aligned_seq2))
        self.traceback_path = list(reversed(path))


#! Local alignment using Smith-Waterman algorithm
class SmithWaterman(SequenceAligner):
    
    # Initialize matrix for local alignment
    def initialize_matrix(self):
        rows = len(self.seq1) + 1
        cols = len(self.seq2) + 1
        self.matrix = np.zeros((rows, cols), dtype=int)
        self.traceback_matrix = [[None for _ in range(cols)] for _ in range(rows)]
    
    # Fill the scoring matrix
    def fill_matrix(self):
        rows = len(self.seq1) + 1
        cols = len(self.seq2) + 1
        
        for i in range(1, rows):
            for j in range(1, cols):
                # Calculate scores for three options
                match_score = self.matrix[i-1, j-1] + self.similarity_score(self.seq1[i-1], self.seq2[j-1])
                delete_score = self.matrix[i-1, j] + self.gap
                insert_score = self.matrix[i, j-1] + self.gap
                
                # Smith-Waterman: include 0 as option (start new alignment)
                max_score = max(0, match_score, delete_score, insert_score)
                self.matrix[i, j] = max_score
                
                # Track the direction
                if max_score == 0:
                    self.traceback_matrix[i][j] = 'E'  # End
                elif max_score == match_score:
                    self.traceback_matrix[i][j] = 'D'  # Diagonal
                elif max_score == delete_score:
                    self.traceback_matrix[i][j] = 'U'  # Up
                else:
                    self.traceback_matrix[i][j] = 'L'  # Left
        
        # Find the maximum score and its position
        max_score_pos = np.unravel_index(np.argmax(self.matrix), self.matrix.shape)
        self.score = int(self.matrix[max_score_pos])
        self.max_pos = max_score_pos
    
    # Traceback from maximum score position to a cell with 0
    def traceback(self):
        aligned_seq1 = []
        aligned_seq2 = []
        path = []
        
        i, j = self.max_pos
        
        while i > 0 and j > 0:
            path.append((i, j))
            direction = self.traceback_matrix[i][j]
            
            if direction == 'E':
                break
            elif direction == 'D':
                aligned_seq1.append(self.seq1[i-1])
                aligned_seq2.append(self.seq2[j-1])
                i -= 1
                j -= 1
            elif direction == 'U':
                aligned_seq1.append(self.seq1[i-1])
                aligned_seq2.append('-')
                i -= 1
            else:  # 'L'
                aligned_seq1.append('-')
                aligned_seq2.append(self.seq2[j-1])
                j -= 1
        
        self.aligned_seq1 = ''.join(reversed(aligned_seq1))
        self.aligned_seq2 = ''.join(reversed(aligned_seq2))
        self.traceback_path = list(reversed(path))


# Sidebar for input
st.sidebar.header("Configuration")

# Sequence input
seq1 = st.sidebar.text_area("Sequence 1:", value="AGGTAB", height=100)
seq2 = st.sidebar.text_area("Sequence 2:", value="GXTXAYB", height=100)

# Scoring parameters
st.sidebar.subheader("Scoring Parameters")
match_score = st.sidebar.number_input("Match Score:", value=2, step=1)
mismatch_score = st.sidebar.number_input("Mismatch Score:", value=-1, step=1)
gap_score = st.sidebar.number_input("Gap Penalty:", value=-1, step=1)

# Alignment type
alignment_type = st.sidebar.radio("Alignment Type:", ["Global (Needleman-Wunsch)", "Local (Smith-Waterman)"])

# Input validation
seq1_clean = seq1.strip().upper()
seq2_clean = seq2.strip().upper()

if seq1_clean and seq2_clean:
    # Perform alignment
    if alignment_type == "Global (Needleman-Wunsch)":
        aligner = NeedlemanWunsch(seq1_clean, seq2_clean, match_score, mismatch_score, gap_score)
    else:
        aligner = SmithWaterman(seq1_clean, seq2_clean, match_score, mismatch_score, gap_score)
    
    aligned_seq1, aligned_seq2, score, traceback_path = aligner.align()
    
    # Display results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Alignment Results")
        st.metric("Alignment Score", score)
        
        st.write("**Aligned Sequence 1:**")
        st.code(aligned_seq1, language="text")
        
        st.write("**Aligned Sequence 2:**")
        st.code(aligned_seq2, language="text")
        
        # Display similarity information
        st.write("**Alignment Details:**")
        matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
        mismatches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != b and a != '-' and b != '-')
        gaps_seq1 = aligned_seq1.count('-')
        gaps_seq2 = aligned_seq2.count('-')
        
        details_df = pd.DataFrame({
            "Metric": ["Matches", "Mismatches", "Gaps (Seq1)", "Gaps (Seq2)", "Alignment Length"],
            "Count": [matches, mismatches, gaps_seq1, gaps_seq2, len(aligned_seq1)]
        })
        st.dataframe(details_df, width=400)
    
    with col2:
        st.subheader("Scoring Matrix")
        
        # Display matrix - create column names to avoid duplicates
        col_names = [''] + [f"{aligner.seq2[j]}({j})" for j in range(len(aligner.seq2))]
        row_names = ['-'] + [f"{aligner.seq1[i]}({i})" for i in range(len(aligner.seq1))]
        
        matrix_display = []
        for i in range(len(aligner.matrix)):
            row = [row_names[i]] + [int(aligner.matrix[i, j]) for j in range(len(aligner.matrix[0]))]
            matrix_display.append(row)
        
        # Ensure column count matches
        matrix_df = pd.DataFrame(matrix_display)
        st.dataframe(matrix_df, width=600)
        
        # Highlight traceback path on matrix
        st.write("**Matrix with Traceback Path:**")
        highlighted_matrix = np.full(aligner.matrix.shape, "", dtype=object)
        
        for idx, (i, j) in enumerate(traceback_path):
            if idx == 0:
                highlighted_matrix[i, j] = f"START {int(aligner.matrix[i, j])}"
            else:
                highlighted_matrix[i, j] = f"PATH {int(aligner.matrix[i, j])}"
        
        # Display path information
        st.write(f"**Traceback Path (Total steps: {len(traceback_path)}):**")
        st.write(f"Coordinates: {' -> '.join([f'({i},{j})' for i, j in traceback_path[:5]])}{'...' if len(traceback_path) > 5 else ''}")
        
        # Display path visualization
        path_text = " -> ".join([f"({i},{j})" for i, j in traceback_path])
        with st.expander("View Full Traceback Path"):
            st.code(path_text, language="text")
    
    # Visualization of alignment
    st.subheader("Visual Alignment")
    
    col_vis1, col_vis2, col_vis3 = st.columns(3)
    
    with col_vis1:
        st.write("**Sequence 1:**")
        for i in range(0, len(aligned_seq1), 50):
            st.code(aligned_seq1[i:i+50])
    
    with col_vis2:
        st.write("**Match/Mismatch:**")
        match_str = ""
        for a, b in zip(aligned_seq1, aligned_seq2):
            if a == b:
                match_str += "|"
            elif a == '-' or b == '-':
                match_str += "."
            else:
                match_str += "X"
        
        for i in range(0, len(match_str), 50):
            st.code(match_str[i:i+50])
    
    with col_vis3:
        st.write("**Sequence 2:**")
        for i in range(0, len(aligned_seq2), 50):
            st.code(aligned_seq2[i:i+50])

else:
    st.warning("Please enter both sequences to perform alignment")