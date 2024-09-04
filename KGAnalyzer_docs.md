
### 1. **Imports and Setup**

```python
import os
import re
import nltk
import networkx as nx
import random
from txtai import LLM, Embeddings
from tqdm import tqdm
from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

nltk.download('punkt', quiet=True)
```

- **Imports**: This section imports various libraries:
  - `os`, `re`: For file and regex operations.
  - `nltk`: For natural language processing tasks (e.g., tokenizing text).
  - `networkx`: For creating and analyzing graphs.
  - `random`: For generating random choices, used in graph traversal.
  - `txtai`: For using large language models (LLMs) and embeddings.
  - `tqdm`: For displaying progress bars in loops.
  - `Counter`, `OrderedDict`: For counting and managing ordered data.
  - `matplotlib.pyplot`: For plotting data visualizations.
  - `pandas`: For reading and manipulating CSV files.
- **NLTK Setup**: Downloads the 'punkt' tokenizer from NLTK, used for sentence tokenization.

### 2. **Class Definition: `KnowledgeGraphAnalyzer`**

```python
class KnowledgeGraphAnalyzer:
    """
    A class used to analyze knowledge graphs.

    Attributes:
    ----------
    embeddings_path : str
        The path to the embeddings file.
    llm_model : str
        The name of the LLM model to use.
    csv_filename : str
        The filename of the CSV file to use.
    embeddings_config : dict
        The configuration for the embeddings.
    embeddings : Embeddings
        The embeddings object.
    llm : LLM
        The LLM object.
    nx_graph : nx.Graph
        The NetworkX graph object.
    id_mapping : dict
        A dictionary mapping node IDs to node objects.

    Methods:
    -------
    process_text_file(file_path)
        Process a text file and generate topics.
    generate_topics(paragraphs)
        Generate topics for a list of paragraphs.
    analyze_graph()
        Analyze the knowledge graph.
    save_embeddings(filename)
        Save the embeddings to a file.
    load_embeddings(filename)
        Load the embeddings from a file.
    graph_qa(question)
        Perform Q&A on the graph.
    """
```

- **Class Overview**: `KnowledgeGraphAnalyzer` is a class for analyzing knowledge graphs using embeddings and large language models.
- **Attributes**:
  - **`embeddings_path`**: Path to the embeddings file.
  - **`llm_model`**: Name of the language model to use.
  - **`csv_filename`**: Filename of the CSV file with data.
  - **`embeddings_config`**: Configuration for embeddings.
  - **`embeddings`**: An object to handle embeddings.
  - **`llm`**: The large language model object.
  - **`nx_graph`**: A NetworkX graph object.
  - **`id_mapping`**: A mapping from node IDs to graph node objects.
- **Methods**: 
  - The methods cover text processing, topic generation, graph analysis, Q&A, and file operations (save/load).

### 3. **Initialization: `__init__` Method**

```python
def __init__(self, embeddings_path="intfloat/e5-large", llm_model="TheBloke/Mistral-7B-OpenOrca-AWQ", 
             embeddings_config=None, csv_filename="hadenpa_lore_expanded.csv"):
    """
    Initialize the KnowledgeGraphAnalyzer object.

    Parameters:
    ----------
    embeddings_path : str
        The path to the embeddings file.
    llm_model : str
        The name of the LLM model to use.
    csv_filename : str
        The filename of the CSV file to use.
    embeddings_config : dict
        The configuration for the embeddings.
    """
    self.embeddings_path = embeddings_path
    self.llm_model = llm_model
    self.csv_filename = csv_filename
    
    if embeddings_config is None:
        embeddings_config = {
            "autoid": "uuid5",
            "instructions": {"query": "query: ", "data": "passage: "},
            "content": True,
            "graph": {"approximate": False, "minscore": 0.7}
        }
    
    self.embeddings = Embeddings(path=self.embeddings_path, **embeddings_config)
    self.llm = LLM(self.llm_model)
    self.llm.generator.llm.pipeline.tokenizer.pad_token_id = self.llm.generator.llm.pipeline.tokenizer.eos_token_id
    self.nx_graph = None
    self.id_mapping = None
```

- **Purpose**: Initializes the `KnowledgeGraphAnalyzer` object with default paths for embeddings and the LLM model.
- **Configuration**:
  - Sets up the embeddings configuration if not provided.
  - Creates `Embeddings` and `LLM` objects using the specified paths and configuration.
  - Initializes an empty graph (`nx_graph`) and an ID mapping dictionary (`id_mapping`).

### 4. **Method: `process_text_file`**

```python
def process_text_file(self, file_path: str, max_paragraph_length: int = 1300, overlap: int = 200) -> list:
    """
    Process a text file and generate topics.

    Parameters:
    ----------
    file_path : str
        The path to the text file.
    max_paragraph_length : int
        The maximum length of a paragraph.
    overlap : int
        The overlap between paragraphs.

    Returns:
    -------
    list
        A list of paragraphs.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    text = re.sub(r'\n+', ' ', text)
    sentences = nltk.sent_tokenize(text)
    
    paragraphs = []
    current_paragraph = ""
    
    for sentence in sentences:
        if len(current_paragraph) + len(sentence) + 1 <= max_paragraph_length:
            current_paragraph += (" " + sentence if current_paragraph else sentence)
        else:
            paragraphs.append(current_paragraph.strip())
            overlap_text = current_paragraph[-overlap:] if len(current_paragraph) > overlap else current_paragraph
            current_paragraph = overlap_text + " " + sentence
    
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    
    final_paragraphs = []
    for paragraph in paragraphs:
        while len(paragraph) > max_paragraph_length:
            split_point = max_paragraph_length - overlap
            final_paragraphs.append(paragraph[:split_point].strip())
            paragraph = paragraph[split_point - overlap:].strip()
        final_paragraphs.append(paragraph.strip())
    
    return final_paragraphs
```

- **Purpose**: Reads a text file, processes it into paragraphs, and ensures each paragraph has a controlled length with overlap for continuity.
- **Steps**:
  - Opens and reads the text file.
  - Tokenizes the text into sentences.
  - Combines sentences into paragraphs with a maximum length (`max_paragraph_length`) and overlap (`overlap`).
  - Ensures paragraphs that exceed the length limit are split appropriately with overlap.
- **Returns**: A list of formatted paragraphs.

### 5. **Method: `generate_topics`**

```python
def generate_topics(self, paragraphs: list, batch_size: int = 5) -> None:
    """
    Generate topics for a list of paragraphs.

    Parameters:
    ----------
    paragraphs : list
        A list of paragraphs.
    batch_size : int
        The batch size for generating topics.
    """
    existing_paragraphs = set()
    if os.path.exists(self.csv_filename):
        df = pd.read_csv(self.csv_filename)
        existing_paragraphs = set(df['Paragraph'].tolist())

    new_paragraphs = [p for p in paragraphs if p not in existing_paragraphs]
    if not new_paragraphs:
        print("No new paragraphs to process.")
        return

    self.embeddings.upsert([(str(i), text, None) for i, text in enumerate(new_paragraphs)])

    batch = []
    for uid in tqdm(range(len(new_paragraphs)), desc="Inferring topics"):
        text = new_paragraphs[uid]
        batch.append((uid, text))
        if len(batch) == batch_size:
            self._process_topic_batch(batch)
            batch = []

    if batch:
        self._process_topic_batch(batch)
```

- **Purpose**: Generates topics for paragraphs and saves new paragraphs into the knowledge graph.
- **Steps**:
  - Checks for existing paragraphs in the specified CSV file to avoid processing duplicates.
  - Filters out new paragraphs and inserts them into the embeddings system.
  - Processes the paragraphs in batches (`batch_size`) and generates topics for each batch.
- **Key Sub-Method**: Calls `_process_topic_batch` to handle each batch of paragraphs.

### 6. **Helper Method: `_process_topic_batch`**

```python
def _process_topic_batch(self, batch: list) -> None:
    """
    Process a batch of paragraphs and generate topics.

    Parameters:
    ----------
    batch : list
        A list of paragraphs.
    """
    prompt_template = "Create a simple, concise topic for the following text. Only return the topic name.\n\nText: {text}"
    prompts = [[{"role": "user", "content": prompt_template.format(text=text)}] for _, text in batch]

    for uid, topic in zip(
        [uid for uid, _ in batch

],
        self.llm(prompts, maxlength=2048, batch_size=len(batch))
    ):
        self.embeddings.graph.addattribute(uid, "topic", topic)
        topics = self.embeddings.graph.topics
        if topics is not None:
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(uid)
```

- **Purpose**: Processes a batch of paragraphs to generate topics using the LLM.
- **Steps**:
  - Prepares prompts for each paragraph in the batch.
  - Uses the LLM to generate concise topics for each paragraph.
  - Adds the topics to the graph nodes as attributes.
  - Updates the graph's topic list with the new topics.

### 7. **Graph Analysis Methods**

These methods analyze the graph, providing insights into its structure, components, and properties.

#### Method: `analyze_graph`

```python
def analyze_graph(self) -> None:
    """
    Analyze the knowledge graph.
    """
    self.nx_graph = nx.Graph(self.embeddings.graph.backend)
    
    print(f"Number of nodes: {self.nx_graph.number_of_nodes()}")
    print(f"Number of edges: {self.nx_graph.number_of_edges()}")
    
    print("\nBasic Graph Properties:")
    self._print_basic_properties()
    
    print("\nDegree Distribution:")
    degree_dist = self._get_degree_distribution()
    self._print_degree_distribution(degree_dist)
    
    print("\nCentrality Analysis:")
    self._centrality_analysis()
    
    print("\nClustering Analysis:")
    self._clustering_analysis()
    
    print("\nConnected Components:")
    self._component_analysis()
    
    print("\nTopic Distribution:")
    self._topic_distribution()
```

- **Purpose**: Runs various analyses on the knowledge graph.
- **Steps**:
  - Creates a NetworkX graph from the embeddings graph backend.
  - Prints graph properties, degree distribution, centrality measures, clustering analysis, and component details.
  - Calls helper methods for detailed analysis in each area.

### 8. **Helper Methods for Graph Analysis**

Several helper methods support the `analyze_graph` method by providing specific analyses:

#### a. **Basic Properties and Degree Distribution**

```python
def _print_basic_properties(self) -> None:
    """
    Print basic graph properties.
    """
    print(f"Is connected: {nx.is_connected(self.nx_graph)}")
    print(f"Diameter: {nx.diameter(self.nx_graph) if nx.is_connected(self.nx_graph) else 'N/A (Graph is not connected)'}")
    print(f"Average shortest path length: {nx.average_shortest_path_length(self.nx_graph) if nx.is_connected(self.nx_graph) else 'N/A (Graph is not connected)'}")
    print(f"Average clustering coefficient: {nx.average_clustering(self.nx_graph)}")

def _get_degree_distribution(self) -> Counter:
    """
    Get the degree distribution of the graph.

    Returns:
    -------
    Counter
        A Counter object representing the degree distribution.
    """
    return Counter(dict(self.nx_graph.degree()).values())

def _print_degree_distribution(self, degree_dist: Counter) -> None:
    """
    Print the degree distribution.

    Parameters:
    ----------
    degree_dist : Counter
        A Counter object representing the degree distribution.
    """
    for degree, count in sorted(degree_dist.items()):
        print(f"Degree {degree}: {count} nodes")
    
    plt.figure(figsize=(10, 6))
    plt.bar(degree_dist.keys(), degree_dist.values())
    plt.xlabel('Degree')
    plt.ylabel('Number of Nodes')
    plt.title('Degree Distribution')
    plt.show()
```

- **Purpose**: Provides an overview of basic properties (e.g., connectivity, diameter) and degree distribution.
- **Visualization**: Degree distribution is visualized using a bar plot.

#### b. **Centrality and Clustering Analysis**

```python
def _centrality_analysis(self, top_n: int = 10) -> None:
    """
    Perform centrality analysis.

    Parameters:
    ----------
    top_n : int
        The number of top nodes to display.
    """
    centrality_measures = {
        "degree": nx.degree_centrality,
        "betweenness": nx.betweenness_centrality,
        "closeness": nx.closeness_centrality
    }

    for measure, func in centrality_measures.items():
        print(f"\nTop {top_n} nodes by {measure} centrality:")
        centrality = func(self.nx_graph)
        for node, value in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            print(f"Node {node}: {value:.4f}")

def _clustering_analysis(self) -> None:
    """
    Perform clustering analysis.
    """
    clustering_coeffs = nx.clustering(self.nx_graph)
    avg_clustering = sum(clustering_coeffs.values()) / len(clustering_coeffs)
    print(f"Average clustering coefficient: {avg_clustering:.4f}")
    
    print("\nTop nodes by local clustering coefficient:")
    for node, coeff in sorted(clustering_coeffs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"Node {node}: {coeff:.4f}")
```

- **Purpose**: Analyzes centrality (importance of nodes) and clustering (local connections) in the graph.
- **Details**:
  - **Centrality**: Evaluates nodes' importance based on degree, betweenness, and closeness centrality.
  - **Clustering**: Computes local clustering coefficients, indicating how nodes cluster together.

#### c. **Component and Topic Distribution Analysis**

```python
def _component_analysis(self) -> None:
    """
    Perform connected component analysis.
    """
    components = list(nx.connected_components(self.nx_graph))
    print(f"Number of connected components: {len(components)}")
    print(f"Size of the largest component: {len(max(components, key=len))}")
    
    component_sizes = [len(c) for c in components]
    plt.figure(figsize=(10, 6))
    plt.hist(component_sizes, bins=20)
    plt.xlabel('Component Size')
    plt.ylabel('Frequency')
    plt.title('Distribution of Connected Component Sizes')
    plt.show()

def _topic_distribution(self, top_n: int = 10, plot_n: int = 20) -> None:
    """
    Perform topic distribution analysis.

    Parameters:
    ----------
    top_n : int
        The number of top topics to display.
    plot_n : int
        The number of topics to plot.
    """
    topic_counts = Counter()
    for node in self.embeddings.graph.scan():
        topic = self.embeddings.graph.attribute(node, "topic")
        if topic:
            topic_counts[topic] += 1
    
    print(f"Top {top_n} most common topics:")
    for topic, count in topic_counts.most_common(top_n):
        print(f"{topic}: {count}")
    
    plt.figure(figsize=(12, 6))
    topics, counts = zip(*topic_counts.most_common(plot_n))
    plt.bar(topics, counts)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Topics')
    plt.ylabel('Number of Nodes')
    plt.title(f'Distribution of Top {len(topics)} Topics')
    plt.tight_layout()
    plt.show()
```

- **Purpose**: Analyzes the connected components (subgraphs) and distribution of topics within the graph.
- **Details**:
  - **Component Analysis**: Shows the number and sizes of connected components (clusters of nodes).
  - **Topic Distribution**: Displays the most common topics in the graph.

### 9. **Embeddings Management: Save and Load**

```python
def save_embeddings(self, filename: str = 'hadenpa.tar.gz') -> None:
    """
    Save the embeddings to a file.

    Parameters:
    ----------
    filename : str
        The filename to save the embeddings to.
    """
    self.embeddings.save(filename)

def load_embeddings(self, filename: str = 'hadenpa.tar.gz') -> None:
    """
    Load the embeddings from a file.

    Parameters:
    ----------
    filename : str
        The filename to load the embeddings from.
    """
    if os.path.exists(filename):
        self.embeddings.load(filename)
        print(f"Embeddings loaded from {filename}")
    else:
        print(f"File {filename} not found. Starting with fresh embeddings.")
```

- **Purpose**: Provides methods to save and load embeddings, facilitating reuse and persistence of the graph's data.

### 10. **Graph Q&A: `graph_qa` Method**

```python
def graph_qa(self, question: str, max_path_length: int = 3, num_paths: int = 3) -> str:
    """
    Perform Q&A on the graph.

    Parameters:
    ----------
    question : str
        The question to answer.
    max_path_length : int
        The maximum length of a path.
    num_paths : int
        The number of paths to consider.

    Returns:
    -------
    str
        The answer to the question.
    """
    if not self.nx_graph or not self.id_mapping:
        self.initialize_graph()

    start_nodes = self.embeddings.search(question, 3)
    paths = []
    for start_node in start_nodes:
        if start_node['id'] in self.id_mapping:
            nx_node = self.id_mapping[start_node['id']]
            paths.extend(self.random_walks(nx_node, max_path_length, num_paths))
    path_info = self.extract_path_info(paths)
    return self.generate_answer(question, path

_info)
```

- **Purpose**: Answers questions by traversing the graph to find relevant paths.
- **Steps**:
  - Searches for relevant nodes in the graph based on the question.
  - Performs random walks from start nodes to explore paths.
  - Extracts information from the paths and generates an answer using the LLM.

### 11. **Helper Methods for Q&A**

#### a. **Random Walks and Path Information**

```python
def random_walks(self, start_node: str, max_length: int, num_paths: int) -> list:
    """
    Perform random walks on the graph.

    Parameters:
    ----------
    start_node : str
        The starting node.
    max_length : int
        The maximum length of a path.
    num_paths : int
        The number of paths to consider.

    Returns:
    -------
    list
        A list of paths.
    """
    paths = []
    visited_global = set()
    for _ in range(num_paths):
        path = [start_node]
        visited_local = set([start_node])
        for _ in range(max_length - 1):
            neighbors = [n for n in self.nx_graph.neighbors(path[-1]) 
                         if n not in visited_local and n not in visited_global]
            if not neighbors:
                break
            next_node = random.choice(neighbors)
            path.append(next_node)
            visited_local.add(next_node)
        visited_global.update(visited_local)
        paths.append(path)
    return paths

def extract_path_info(self, paths: list) -> list:
    """
    Extract information from paths.

    Parameters:
    ----------
    paths : list
        A list of paths.

    Returns:
    -------
    list
        A list of path information.
    """
    seen = set()
    path_info = []
    for path in paths:
        info = []
        for node in path:
            if node not in seen:
                text = self.embeddings.graph.attribute(node, "text")
                topic = self.embeddings.graph.attribute(node, "topic")
                info.append({"node": node, "text": text, "topic": topic})
                seen.add(node)
        if info:
            path_info.append(info)
    return path_info
```

- **Random Walks**: Generates paths from a starting node by randomly traversing neighbors, respecting a maximum path length.
- **Path Information**: Extracts and returns detailed information from the paths, including node texts and topics.

#### b. **Answer Generation**

```python
def generate_answer(self, question: str, path_info: list) -> str:
    """
    Generate an answer to a question.

    Parameters:
    ----------
    question : str
        The question to answer.
    path_info : list
        A list of path information.

    Returns:
    -------
    str
        The answer to the question.
    """
    graph_context = self.format_context(path_info)
    vector_results = self.get_diverse_vector_results(question)
    vector_context = self.format_vector_results(vector_results)
    combined_context = f"{graph_context}\n\nAdditional relevant information:\n{vector_context}"
    
    prompt = f"""Given the following context extracted from a document graph and vector search, answer the question. 
Use only the information provided in the context. If the answer cannot be fully determined from the context, 
say so and provide the best partial answer possible.

Context:
{combined_context}

Question: {question}

Answer: """
    
    with open('prompt.txt', 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Prompt saved to {os.path.abspath('prompt.txt')}")
    
    return self.llm(prompt, max_length=8192)
```

- **Purpose**: Combines context from graph paths and vector search results to form a comprehensive context for answering the question.
- **Steps**:
  - Formats the context from paths and vector results.
  - Creates a prompt for the LLM with the combined context and question.
  - Saves the prompt to a file and passes it to the LLM for generating the answer.

### 12. **Diverse Vector Results**

```python
def get_diverse_vector_results(self, question: str, num_results: int = 5, diversity_threshold: float = 0.7) -> list:
    """
    Get diverse vector results.

    Parameters:
    ----------
    question : str
        The question to answer.
    num_results : int
        The number of results to consider.
    diversity_threshold : float
        The diversity threshold.

    Returns:
    -------
    list
        A list of diverse vector results.
    """
    try:
        all_results = self.embeddings.search(question, num_results * 2)
        diverse_results = OrderedDict()
        for result in all_results:
            if len(diverse_results) >= num_results:
                break
            is_diverse = True
            for dr in diverse_results.values():
                try:
                    similarity = self.embeddings.similarity(result['text'], dr['text'])
                    if isinstance(similarity, list):
                        similarity = similarity[0][1]
                    if not isinstance(similarity, (int, float)):
                        raise TypeError(f"Unexpected similarity type: {type(similarity)}")
                    if similarity >= diversity_threshold:
                        is_diverse = False
                        break
                except Exception as e:
                    print(f"Error calculating similarity: {e}")
                    is_diverse = False
                    break
            if is_diverse:
                diverse_results[result['id']] = result
        return list(diverse_results.values())
    except Exception as e:
        print(f"Error in get_diverse_vector_results: {e}")
        return []
```

- **Purpose**: Retrieves vector search results and filters them to ensure diversity based on a similarity threshold.
- **Steps**:
  - Searches for relevant vectors related to the question.
  - Checks similarity between results to maintain diversity.
  - Returns a filtered list of diverse results.

### 13. **Context Formatting**

```python
def format_vector_results(self, results: list, max_text_length: int = 1200) -> str:
    """
    Format vector results.

    Parameters:
    ----------
    results : list
        A list of vector results.
    max_text_length : int
        The maximum text length.

    Returns:
    -------
    str
        The formatted vector results.
    """
    context = ""
    for i, result in enumerate(results, 1):
        context += f"Document {i}:\n"
        context += f"Text: {result['text'][:max_text_length]}...\n"
        if 'metadata' in result and 'topic' in result['metadata']:
            context += f"Topic: {result['metadata']['topic']}\n"
        context += f"Relevance Score: {result['score']:.4f}\n\n"
    return context

def format_context(self, path_info: list, max_text_length: int = 400) -> str:
    """
    Format context.

    Parameters:
    ----------
    path_info : list
        A list of path information.
    max_text_length : int
        The maximum text length.

    Returns:
    -------
    str
        The formatted context.
    """
    context = "Information extracted from the document graph:\n\n"
    for i, path in enumerate(path_info, 1):
        context += f"Path {i}:\n"
        for node_info in path:
            context += f"- Topic: {node_info['topic']}\n  Text: {node_info['text'][:max_text_length]}...\n"
        context += "\n"
    return context
```

- **Purpose**: Formats the context extracted from the graph and vector results into readable text for the LLM.
- **Steps**:
  - Limits text length to ensure readability.
  - Includes relevant metadata such as topics and relevance scores.

### 14. **Graph Initialization**

```python
def initialize_graph(self) -> None:
    """
    Initialize the graph.
    """
    self.nx_graph = nx.Graph(self.embeddings.graph.backend)
    self.id_mapping = self._create_id_mapping()

def _create_id_mapping(self) -> dict:
    """
    Create an ID mapping.

    Returns:
    -------
    dict
        A dictionary mapping node IDs to node objects.
    """
    return {self.embeddings.graph.attribute(node, "id"): node for node in self.nx_graph.nodes()}
```

- **Purpose**: Initializes the graph from the embeddings backend and creates an ID mapping for nodes.
- **Steps**:
  - Converts the embeddings graph into a NetworkX graph.
  - Maps node IDs to actual node objects for easy reference.

