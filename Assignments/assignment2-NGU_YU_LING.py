import streamlit as st
import numpy as np
import pandas as pd
import time
from typing import List, Tuple

# Page configuration
st.set_page_config(
    page_title="Sequence Analysis Tool",
    layout="wide",
)

st.title("Sequence Analysis Tool")
st.markdown("Analyze DNA/Protein sequences with distance metrics and pattern matching algorithms")


#! Distance Metrics
def hamming_distance(seq1: str, seq2: str) -> int:
    # Calculate Hamming distance between two sequences
    if len(seq1) != len(seq2):
        return -1
    
    distance = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            distance += 1
    return distance

# Calculate edit distance using dynamic programming
def edit_distance(seq1: str, seq2: str) -> Tuple[int, np.ndarray]:
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = 1 + min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1])
    
    return dp[m, n], dp


#! Sequence Exact Matching Algorithms
# Naive string matching algorithm
def naive_matching(reference: str, pattern: str) -> Tuple[List[int], float]:
    start_time = time.time()
    matches = []
    
    for i in range(len(reference) - len(pattern) + 1):
        match = True
        for j in range(len(pattern)):
            if reference[i + j] != pattern[j]:
                match = False
                break
        
        if match:
            matches.append(i)
    
    elapsed_time = time.time() - start_time
    return matches, elapsed_time

# Boyer-Moore algorithm using bad character rule
def boyer_moore_matching(reference: str, pattern: str) -> Tuple[List[int], float]:
    start_time = time.time()
    matches = []
    
    # Build bad character table
    bad_char = {}
    for i in range(len(pattern)):
        bad_char[pattern[i]] = i
    
    i = 0
    while i <= len(reference) - len(pattern):
        j = len(pattern) - 1
        
        # Compare pattern with reference from right to left
        while j >= 0 and pattern[j] == reference[i + j]:
            j -= 1
        
        if j < 0:
            # Pattern found
            matches.append(i)
            i += 1
        else:
            # Shift based on bad character rule
            bad_char_value = bad_char.get(reference[i + j], -1)
            shift = max(1, j - bad_char_value)
            i += shift
    
    elapsed_time = time.time() - start_time
    return matches, elapsed_time


#! Streamlit Application
# Sidebar inputs
st.sidebar.header("Input Sequences")

reference = st.sidebar.text_area(
    "Reference Sequence:",
    value="ATCGATCGATCGATCG",
    height=100
)

pattern = st.sidebar.text_area(
    "Pattern Sequence:",
    value="ATCG",
    height=100
)

reference = reference.strip().upper()
pattern = pattern.strip().upper()

# Create tabs
tab1, tab2 = st.tabs(["Distance Metrics", "Sequence Matching"])

with tab1:
    st.header("Distance Metrics Analysis")
    
    if reference and pattern:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hamming Distance")
            
            if len(reference) == len(pattern):
                h_distance = hamming_distance(reference, pattern)
                st.metric("Hamming Distance", h_distance)
                
                # Show alignment
                st.write("Sequence Comparison:")
                col_ref, col_match, col_pat = st.columns(3)
                with col_ref:
                    st.write("**Reference:**")
                    st.code(reference)
                with col_match:
                    st.write("**Match:**")
                    match_str = ""
                    for i in range(len(reference)):
                        if reference[i] == pattern[i]:
                            match_str += "|"
                        else:
                            match_str += "X"
                    st.code(match_str)
                with col_pat:
                    st.write("**Pattern:**")
                    st.code(pattern)
            else:
                st.warning(f"Sequences must be equal length for Hamming distance. Reference: {len(reference)}, Pattern: {len(pattern)}")
        
        with col2:
            st.subheader("Edit Distance (Levenshtein)")
            
            min_distance, dp_matrix = edit_distance(reference, pattern)
            st.metric("Edit Distance", min_distance)
            
            # Display DP matrix
            st.write("Dynamic Programming Matrix:")
            dp_df = pd.DataFrame(
                dp_matrix
            )
            st.dataframe(dp_df, width=600)
            
            # Additional metrics
            st.write("Distance Metrics:")
            metrics_data = {
                "Metric": ["Edit Distance", "Sequence Length (Reference)", "Sequence Length (Pattern)"],
                "Value": [
                    str(min_distance),
                    str(len(reference)),
                    str(len(pattern))
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, width=400)
    else:
        st.warning("Please enter both sequences")


with tab2:
    st.header("Sequence Exact Matching")
    
    if reference and pattern:
        if len(pattern) > len(reference):
            st.error("Pattern must be shorter than or equal to the reference sequence")
        else:
            # Run both algorithms
            naive_matches, naive_time = naive_matching(reference, pattern)
            bm_matches, bm_time = boyer_moore_matching(reference, pattern)
            
            # Compare results side by side
            col_naive, col_bm = st.columns(2)
            
            with col_naive:
                st.subheader("Naive Algorithm")
                
                st.write(f"**Matches Found:** {len(naive_matches)}")
                st.metric("Execution Time (ms)", f"{naive_time * 1000:.4f}")
                
                if naive_matches:
                    st.write(f"**Match Offsets:** {naive_matches}")
                    
                    # Display matches
                    st.write("Match Visualization:")
                    for offset in naive_matches[:5]:
                        # Show matching portion
                        start = max(0, offset - 2)
                        end = min(len(reference), offset + len(pattern) + 2)
                        context = reference[start:end]
                        match_pos = offset - start
                        highlight = " " * match_pos + "^" * len(pattern)
                        st.code(f"{context}\n{highlight}")
                    
                    if len(naive_matches) > 5:
                        st.info(f"... and {len(naive_matches) - 5} more matches")
                else:
                    st.info("No matches found")
            
            with col_bm:
                st.subheader("Boyer-Moore Algorithm")
                
                st.write(f"**Matches Found:** {len(bm_matches)}")
                st.metric("Execution Time (ms)", f"{bm_time * 1000:.4f}")
                
                if bm_matches:
                    st.write(f"**Match Offsets:** {bm_matches}")
                    
                    # Display matches
                    st.write("Match Visualization:")
                    for offset in bm_matches[:5]:
                        # Show matching portion
                        start = max(0, offset - 2)
                        end = min(len(reference), offset + len(pattern) + 2)
                        context = reference[start:end]
                        match_pos = offset - start
                        highlight = " " * match_pos + "^" * len(pattern)
                        st.code(f"{context}\n{highlight}")
                    
                    if len(bm_matches) > 5:
                        st.info(f"... and {len(bm_matches) - 5} more matches")
                else:
                    st.info("No matches found")
            
            # Performance comparison
            st.subheader("Performance Comparison")
            
            speedup_value = f"{naive_time / bm_time:.2f}x" if bm_time > 0 else "N/A"
            
            comparison_data = {
                "Algorithm": ["Naive", "Boyer-Moore"],
                "Matches Found": [str(len(naive_matches)), str(len(bm_matches))],
                "Execution Time (ms)": [f"{naive_time * 1000:.4f}", f"{bm_time * 1000:.4f}"]
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, width=600)
            
            st.metric("Speedup (Naive / Boyer-Moore)", speedup_value)
            
            # Detailed analysis
            st.subheader("Detailed Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.write("**Reference Sequence:**")
                st.code(reference)
            
            with analysis_col2:
                st.write("**Pattern Sequence:**")
                st.code(pattern)
            
            # Full visualization
            st.write("**Full Match Visualization:**")
            
            # Create a visual representation
            visual = ""
            match_positions = set(naive_matches)
            
            for i in range(len(reference)):
                visual += reference[i]
            
            st.code(visual)
            
            # Show match positions
            positions_str = ""
            for i in range(len(reference)):
                if any(i >= pos and i < pos + len(pattern) for pos in match_positions):
                    positions_str += "^"
                else:
                    positions_str += " "
            st.code(positions_str)
            
            # Statistics
            st.write("**Statistics:**")
            stats_data = {
                "Metric": [
                    "Reference Length",
                    "Pattern Length",
                    "Number of Matches"
                ],
                "Value": [
                    str(len(reference)),
                    str(len(pattern)),
                    str(len(naive_matches))
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, width=400)
    
    else:
        st.warning("Please enter both sequences")
