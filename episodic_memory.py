import numpy as np
import torch
import networkx as nx
from transformers import AutoTokenizer, pipeline
import transformers
from sentence_transformers import SentenceTransformer
from scipy.special import legendre
from scipy.stats import entropy
from functools import lru_cache
import time

class MemoryUnit:
    def __init__(self, tokens, embedding, surprise_score, timestamp):
        self.tokens = tokens
        self.embedding = embedding
        self.surprise_score = surprise_score
        self.timestamp = timestamp
        self.access_count = 0
        self.last_access_time = timestamp

def legendre_paragraph_embedding(sentence_embeddings, num_coefficients=10):
    normalized_embeddings = np.tanh(sentence_embeddings)
    num_sentences, embedding_dim = normalized_embeddings.shape
    positions = np.linspace(-1, 1, num_sentences)
    legendre_values = np.zeros((num_sentences, num_coefficients))
    for n in range(num_coefficients):
        legendre_values[:, n] = legendre(n)(positions)
    paragraph_embedding = np.dot(legendre_values.T, normalized_embeddings)
    return paragraph_embedding.flatten()

class MemoryFormation:
    def __init__(self, model, tokenizer, sentence_transformer, threshold=0.5, max_sentences=15, num_coefficients=10):
        self.model = model
        self.tokenizer = tokenizer
        self.sentence_transformer = sentence_transformer
        self.threshold = threshold
        self.max_sentences = max_sentences
        self.num_coefficients = num_coefficients

    def segment_sequence(self, text):
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()][:self.max_sentences]
        
        memory_units = []
        current_memory = []
        current_sentences = []
        
        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            surprise_score = self.calculate_surprise(tokens)
            
            if surprise_score > self.threshold and current_memory:
                embedding = self.compute_embedding(current_sentences)
                memory_units.append(MemoryUnit(current_memory, embedding, surprise_score, time.time()))
                current_memory = tokens
                current_sentences = [sentence]
            else:
                current_memory.extend(tokens)
                current_sentences.append(sentence)
        
        if current_memory:
            embedding = self.compute_embedding(current_sentences)
            memory_units.append(MemoryUnit(current_memory, embedding, surprise_score, time.time()))
        
        return memory_units

    def calculate_surprise(self, tokens):
        input_ids = self.tokenizer.encode(" ".join(tokens), return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            
            uniform_dist = torch.ones_like(probs) / probs.shape[-1]
            kl_div = entropy(probs[0].cpu().numpy(), uniform_dist[0].cpu().numpy())
            
            surprise = -torch.log(probs[0, input_ids[0, -1]]).item() + kl_div
        return surprise

    def compute_embedding(self, sentences):
        sentence_embeddings = self.sentence_transformer.encode(sentences)
        return legendre_paragraph_embedding(sentence_embeddings, self.num_coefficients)

class MemoryRefinement:
    def __init__(self):
        pass

    def refine_memory_units(self, memory_units):
        similarity_matrix = self.calculate_similarity_matrix(memory_units)
        G = nx.from_numpy_matrix(similarity_matrix)
        refined_memory_units = []
        
        for i, memory_unit in enumerate(memory_units):
            subgraph_nodes = range(i, i+1)
            subgraph = G.subgraph(subgraph_nodes)
            cohesion = nx.average_clustering(subgraph)
            separation = nx.average_shortest_path_length(G) - nx.average_shortest_path_length(subgraph)
            
            if cohesion > separation:
                refined_memory_units.append(memory_unit)
            elif refined_memory_units:
                refined_memory_units[-1].tokens.extend(memory_unit.tokens)
                refined_memory_units[-1].embedding = np.mean([refined_memory_units[-1].embedding, memory_unit.embedding], axis=0)
            else:
                refined_memory_units.append(memory_unit)
        
        return refined_memory_units

    def calculate_similarity_matrix(self, memory_units):
        embeddings = np.array([unit.embedding for unit in memory_units])
        similarity_matrix = np.inner(embeddings, embeddings)
        return similarity_matrix

class MemoryRecall:
    def __init__(self, memory_units):
        self.memory_units = memory_units

    @lru_cache(maxsize=1000)
    def retrieve_memory(self, query_embedding_tuple):
        query_embedding = np.array(query_embedding_tuple)
        retrieved_units = []
        current_time = time.time()
        for unit in self.memory_units:
            similarity = np.inner(query_embedding, unit.embedding)
            if similarity > 0.5:  # Threshold for similarity
                retrieved_units.append((unit, similarity))
                unit.access_count += 1
                unit.last_access_time = current_time
        
        retrieved_units = self.apply_temporal_effects(retrieved_units)
        return tuple(retrieved_units)  # Convert to tuple for caching

    def apply_temporal_effects(self, retrieved_units):
        current_time = time.time()
        for i, (unit, similarity) in enumerate(retrieved_units):
            time_decay = np.exp(-0.1 * (current_time - unit.timestamp))
            access_boost = np.log1p(unit.access_count)
            recency_boost = np.exp(-0.01 * (current_time - unit.last_access_time))
            adjusted_similarity = similarity * time_decay * (1 + access_boost) * recency_boost
            retrieved_units[i] = (unit, adjusted_similarity)
        return sorted(retrieved_units, key=lambda x: x[1], reverse=True)

class MemoryManager:
    def __init__(self, max_memories=1000, prune_threshold=0.1, archive_threshold=0.01):
        self.max_memories = max_memories
        self.prune_threshold = prune_threshold
        self.archive_threshold = archive_threshold
        self.archive = []

    def prune_memories(self, memory_units):
        if len(memory_units) <= self.max_memories:
            return memory_units

        current_time = time.time()
        sorted_memories = sorted(
            memory_units,
            key=lambda x: self.calculate_memory_score(x, current_time),
            reverse=True
        )

        kept_memories = sorted_memories[:self.max_memories]
        pruned_memories = sorted_memories[self.max_memories:]

        for memory in pruned_memories:
            if self.should_archive(memory, current_time):
                self.archive.append(memory)
            elif self.should_forget(memory, current_time):
                continue  # Forget the memory
            else:
                kept_memories.append(memory)
        
        return kept_memories

    def calculate_memory_score(self, memory, current_time):
        time_factor = np.exp(-0.1 * (current_time - memory.timestamp))
        access_factor = np.log1p(memory.access_count)
        recency_factor = np.exp(-0.01 * (current_time - memory.last_access_time))
        return memory.surprise_score * time_factor * (1 + access_factor) * recency_factor

    def should_archive(self, memory, current_time):
        score = self.calculate_memory_score(memory, current_time)
        return self.archive_threshold <= score < self.prune_threshold

    def should_forget(self, memory, current_time):
        score = self.calculate_memory_score(memory, current_time)
        return score < self.archive_threshold

    def consolidate_memories(self, memories):
        if not memories:
            return []

        groups = []
        for memory in memories:
            added = False
            for group in groups:
                if np.inner(memory.embedding, group[0].embedding) > self.prune_threshold:
                    group.append(memory)
                    added = True
                    break
            if not added:
                groups.append([memory])

        consolidated = []
        for group in groups:
            if len(group) > 1:
                tokens = [token for mem in group for token in mem.tokens]
                embedding = np.mean([mem.embedding for mem in group], axis=0)
                surprise_score = np.mean([mem.surprise_score for mem in group])
                timestamp = max(mem.timestamp for mem in group)
                consolidated.append(MemoryUnit(tokens, embedding, surprise_score, timestamp))
            else:
                consolidated.append(group[0])

        return consolidated

class EM_LLM:
    def __init__(self, model_name="sethuiyer/Nandine-7b", sentence_transformer_name="all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = self.pipeline.model
        self.sentence_transformer = SentenceTransformer(sentence_transformer_name)
        self.memory_units = []
        self.memory_recall = MemoryRecall([])
        self.memory_manager = MemoryManager()

    def process_sequence(self, text):
        formation = MemoryFormation(self.model, self.tokenizer, self.sentence_transformer)
        new_memory_units = formation.segment_sequence(text)
        
        refinement = MemoryRefinement()
        refined_units = refinement.refine_memory_units(new_memory_units)
        
        self.update_memories(refined_units)
        self.memory_units = self.memory_manager.prune_memories(self.memory_units)
        self.memory_recall = MemoryRecall(self.memory_units)

    def update_memories(self, new_units):
        for new_unit in new_units:
            updated = False
            for existing_unit in self.memory_units:
                similarity = np.inner(new_unit.embedding, existing_unit.embedding)
                if similarity > 0.8:  # High similarity threshold for updating
                    if self.detect_contradiction(new_unit, existing_unit):
                        self.handle_contradiction(new_unit, existing_unit)
                    else:
                        existing_unit.tokens.extend(new_unit.tokens)
                        existing_unit.embedding = (existing_unit.embedding + new_unit.embedding) / 2
                        existing_unit.surprise_score = max(existing_unit.surprise_score, new_unit.surprise_score)
                        existing_unit.timestamp = time.time()
                    updated = True
                    break
            if not updated:
                self.memory_units.append(new_unit)

    def detect_contradiction(self, new_unit, existing_unit):
        embedding_similarity = np.inner(new_unit.embedding, existing_unit.embedding)
        token_overlap = len(set(new_unit.tokens) & set(existing_unit.tokens)) / len(set(new_unit.tokens) | set(existing_unit.tokens))
        return embedding_similarity < 0.3 or token_overlap < 0.1

    def handle_contradiction(self, new_unit, existing_unit):
        versioned_memory = MemoryUnit(
            tokens=new_unit.tokens,
            embedding=new_unit.embedding,
            surprise_score=new_unit.surprise_score,
            timestamp=new_unit.timestamp
        )
        self.memory_units.append(versioned_memory)

    @lru_cache(maxsize=100)
    def query_memory(self, query_text):
        query_embedding = self.sentence_transformer.encode([query_text])[0]
        query_embedding = legendre_paragraph_embedding(query_embedding.reshape(1, -1))
        
        return self.memory_recall.retrieve_memory(tuple(query_embedding))

    def generate_response(self, prompt, max_new_tokens=256):
        retrieved_memories = self.query_memory(prompt)
        
        weighted_memories = []
        current_time = time.time()
        for unit, similarity in retrieved_memories:
            recency_weight = np.exp(-0.1 * (current_time - unit.last_access_time))
            weighted_memories.append((unit, similarity * recency_weight))
        
        weighted_memories.sort(key=lambda x: x[1], reverse=True)
        
        context = ""
        for unit, weight in weighted_memories:
            context += f"{' '.join(unit.tokens)} "
            if len(context.split()) > 100:
                break
        
        full_prompt = f"{context.strip()}\n\n{prompt}"
        
        messages = [{"role": "user", "content": full_prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(chat_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]

# Usage example
em_llm = EM_LLM()

# Process some text to form memories
em_llm.process_sequence("""
The sun rose over the misty hills, painting the sky in hues of pink and gold. 
Birds began their morning chorus, filling the air with melodious chirps. 
In the distance, a lone deer grazed peacefully in a meadow. 
The dew-covered grass sparkled like diamonds in the early light. 
A gentle breeze rustled through the leaves, carrying the scent of wildflowers. 
The world seemed to awaken slowly, embracing the new day with quiet enthusiasm. 
As the mist began to lift, the landscape revealed its full beauty. 
The rolling hills stretched as far as the eye could see, dotted with ancient oak trees. 
A babbling brook wound its way through the valley, its clear waters reflecting the brightening sky. 
The air was crisp and fresh, invigorating the senses and promising new beginnings.
""")

# Generate a response using the EM-LLM
response = em_llm.generate_response("Describe a peaceful morning scene.")
print(response)
