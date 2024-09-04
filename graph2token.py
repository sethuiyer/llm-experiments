import torch
from torch import Tensor
from transformers import BertTokenizer, BertModel
from typing import Dict, List, Tuple, Any
import networkx as nx
import matplotlib.pyplot as plt

class GraphToTokenSequence:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initialize the GraphToTokenSequence with a specified BERT model.
        
        Args:
            model_name (str): The name of the pre-trained BERT model to use.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def levi_graph_conversion(self, graph: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[str]]:
        """
        Transform the original graph to a Levi graph representation.
        
        Args:
            graph (Dict[str, List[Dict[str, str]]]): The original graph structure.
        
        Returns:
            Dict[str, List[str]]: The Levi graph representation.
        """
        levi_graph: Dict[str, List[str]] = {}
        for node, edges in graph.items():
            for edge in edges:
                relation_node = edge['relation']
                target_node = edge['target']
                levi_graph.setdefault(node, []).append(relation_node)
                levi_graph.setdefault(relation_node, []).append(target_node)
        return levi_graph

    def tokenize_nodes(self, levi_graph: Dict[str, List[str]]) -> Tuple[List[int], Dict[str, int]]:
        """
        Tokenize each node in the Levi graph and prepare for positional encoding.
        
        Args:
            levi_graph (Dict[str, List[str]]): The Levi graph representation.
        
        Returns:
            Tuple[List[int], Dict[str, int]]: A tuple containing the list of token IDs and a dictionary of token positions.
        """
        token_ids: List[int] = []
        token_positions: Dict[str, int] = {}
        position = 0
        
        for node, connected_nodes in levi_graph.items():
            node_tokens = self.tokenizer.encode(node, add_special_tokens=False)
            token_ids.extend(node_tokens)
            token_positions[node] = position
            position += len(node_tokens)
            
            for connected_node in connected_nodes:
                connected_node_tokens = self.tokenizer.encode(connected_node, add_special_tokens=False)
                token_ids.extend(connected_node_tokens)
                token_positions[connected_node] = position
                position += len(connected_node_tokens)

        return token_ids, token_positions

    @torch.no_grad()
    def encode_tokens(self, token_ids: List[int]) -> Tensor:
        """
        Convert token IDs to model inputs and encode them using the BERT model.
        
        Args:
            token_ids (List[int]): The list of token IDs to encode.
        
        Returns:
            Tensor: The encoded representation of the input tokens.
        """
        # Prepare input_ids and attention_mask with batch dimension
        input_ids = torch.tensor([token_ids])  # Add batch dimension here
        attention_mask = torch.tensor([[1] * len(token_ids)])  # Same here
        
        # Pass inputs to the model
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def graph_to_sequence(self, graph: Dict[str, List[Dict[str, str]]]) -> Tensor:
        """
        Convert a graph structure to a sequence of encoded tokens.
        
        Args:
            graph (Dict[str, List[Dict[str, str]]]): The input graph structure.
        
        Returns:
            Tensor: The encoded representation of the graph.
        """
        levi_graph = self.levi_graph_conversion(graph)
        token_ids, token_positions = self.tokenize_nodes(levi_graph)
        encoded_tokens = self.encode_tokens(token_ids)
        return encoded_tokens

    def visualize_graph(self, graph: Dict[str, List[Dict[str, str]]]) -> None:
        """
        Visualize the input graph structure using networkx and matplotlib.
        
        Args:
            graph (Dict[str, List[Dict[str, str]]]): The input graph structure.
        """
        G = nx.DiGraph()
        for source, edges in graph.items():
            for edge in edges:
                G.add_edge(source, edge['target'], relation=edge['relation'])

        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, arrows=True)
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title("Graph Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # Example graph structure
    graph = {
        'Dog': [{'relation': 'IsA', 'target': 'Animal'}],
        'Cat': [{'relation': 'IsA', 'target': 'Animal'}],
        'Animal': [{'relation': 'PartOf', 'target': 'Nature'}]
    }

    # Instantiate and use the class
    graph_processor = GraphToTokenSequence()
    
    # Process the graph
    encoded_sequence = graph_processor.graph_to_sequence(graph)
    print("Encoded sequence shape:", encoded_sequence.shape)
    print("Encoded sequence", encoded_sequence)
    
    # Visualize the graph
    graph_processor.visualize_graph(graph)

if __name__ == "__main__":
    main()
