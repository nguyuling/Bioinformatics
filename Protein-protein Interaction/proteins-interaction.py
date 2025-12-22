import requests
import pandas as pd
import matplotlib.pyplot as plt # Used for generating the visualization (the network plot)
import networkx as nx # Used for creating, analyzing, and working with graph structures (the PPI network)

#! Configuration
# Define the base URL for the STRING API's network endpoint
string_url = "https://string-db.org/api/json/network"
# Define the target protein (TP53 - Human tumor suppressor)
protein_name = "TP53" 
# Define the query parameters for the API request
params = {
    "identifiers" : protein_name, # Search for interactions involving this protein
    "species" : 9606, # Specify the species ID for Human
    "limit" : 20 # Restrict the API to return the top 20 interactions
}

#! Data Retrieval
# Send the HTTP GET request to the API with the specified parameters
response = requests.get(string_url, params=params)

if response.status_code == 200:
    # Parse the JSON data into a Python list of dictionaries
    network = response.json()
    print(f"Downloaded {len(network)} interactions for. {protein_name}")
else:
    print(f"Failed to download. Status code: {response.status_code}")
# Print the raw data of the first interaction for structural inspection
print(network[0])

#! Data Processing (Pandas)
# Convert the list of raw interaction dictionaries into a flat Pandas DataFrame
network_df = pd.json_normalize(network)
# Display the first few rows
network_df.head() 
# Display the total number of rows (interactions) and columns (attributes)
network_df.shape

# Select only the three columns necessary for building a network graph
ppi_data = network_df[['preferredName_A', 'preferredName_B', 'score']]
# Rename the columns for standard network graph terminology
ppi_data.columns = ['protein1', 'protein2', 'score']
# Explicitly print the cleaned top 10 interactions to the terminal
print("\n--- Cleaned Protein-Protein Interactions (Top 10) ---")
print(ppi_data.head(10).to_string(index=False))

#! Network Analysis (NetworkX)
# Create a NetworkX graph object from the DataFrame, defining columns as nodes/edges
network_graph = nx.from_pandas_edgelist(ppi_data, "protein1", "protein2")
# Print a summary of the resulting network's size
print(f"Network built with {network_graph.number_of_nodes()} nodes and {network_graph.number_of_edges()} edges")
# Calculate the degree (total connections) for every protein in the network
network_graph.degree() 

# Calculate the degree centrality to identify the most connected proteins
degree_centrality = nx.degree_centrality(network_graph)
# Sort the proteins to find the top 5 with the highest centrality score
top_5_proteins = sorted(degree_centrality.items(), key=lambda x:-x[1])[:5]
# Print the top 5 proteins along with their scores
print(f"Top 5 proteins, centrality: {top_5_proteins}")
# Extract just the names of the top 5 hub proteins into a list.
high_centrality_nodes = [node for node, centrality in top_5_proteins]
# Print the final list of hub proteins.
print(f"Top 5 proteins: {high_centrality_nodes}")


#! Network Visualization (Matplotlib)
# Determine the visual positions of the nodes using a force-directed layout
slayout = nx.spring_layout(network_graph, seed=125)
# Set the title for the visualization.
plt.title('Protein-Protein Interaction Network', fontsize=16)
# Draw the basic network structure
nx.draw(network_graph, slayout, with_labels=True, node_size=1000, node_color='lightblue', font_size=8)
# Highlight the top 5 hub proteins (from the list created above) in a new color.
nx.draw_networkx_nodes(network_graph, slayout, nodelist=high_centrality_nodes, node_color='orange')
# Display the final plot.
plt.show()