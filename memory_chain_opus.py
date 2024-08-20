import numpy as np
import torch
import networkx as nx
from transformers import AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from scipy.special import legendre
from scipy.stats import entropy
from functools import lru_cache
import time
import hashlib

# ------------------------------
# Blockchain Components
# ------------------------------

class MemoryBlock:
    def __init__(self, data, previous_hash):
        self.timestamp = time.time()
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_content = str(self.timestamp) + str(self.data) + str(self.previous_hash) + str(self.nonce)
        return hashlib.sha256(block_content.encode()).hexdigest()

    def mine_block(self, difficulty):
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

class MemoryChain:
    def __init__(self, difficulty=4):
        self.chain = [self.create_genesis_block()]
        self.difficulty = difficulty

    def create_genesis_block(self):
        return MemoryBlock("Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

# ------------------------------
# Memory Management Components 
# ------------------------------

class MemoryUnit:
    def __init__(self, tokens, embedding, surprise_score):
        self.memory_chain = MemoryChain()
        initial_block = MemoryBlock({
            'tokens': tokens,
            'embedding': embedding.tolist(),
            'surprise_score': surprise_score
        }, "0")
        self.memory_chain.add_block(initial_block)
        self.access_count = 0
        self.last_access_time = time.time()

    def add_version(self, tokens, embedding, surprise_score):
        new_block = MemoryBlock({
            'tokens': tokens,
            'embedding': embedding.tolist(),
            'surprise_score': surprise_score
        }, self.memory_chain.get_latest_block().hash)
        self.memory_chain.add_block(new_block)

    def get_current_version(self):
        return self.memory_chain.get_latest_block().data

# ------------------------------
# Memory Formation Components
# ------------------------------

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
                memory_units.append(MemoryUnit(current_memory, embedding, surprise_score))
                current_memory = tokens
                current_sentences = [sentence]
            else:
                current_memory.extend(tokens)
                current_sentences.append(sentence)
        
        if current_memory:
            embedding = self.compute_embedding(current_sentences)
            memory_units.append(MemoryUnit(current_memory, embedding, surprise_score))
        
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

# ------------------------------
# EM-LLM System Components
# ------------------------------

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
        self.memory_formation = MemoryFormation(self.model, self.tokenizer, self.sentence_transformer)

    def process_sequence(self, text):
        new_memory_units = self.memory_formation.segment_sequence(text)
        self.update_memories(new_memory_units)

    def update_memories(self, new_units):
        for new_unit in new_units:
            updated = False
            for existing_unit in self.memory_units:
                current_version = existing_unit.get_current_version()
                new_version = new_unit.get_current_version()
                similarity = np.inner(np.array(new_version['embedding']), np.array(current_version['embedding']))
                if similarity > 0.8:  # High similarity threshold for updating
                    if self.detect_contradiction(new_version, current_version):
                        self.handle_contradiction(new_unit, existing_unit)
                    else:
                        merged_tokens = current_version['tokens'] + new_version['tokens']
                        merged_embedding = (np.array(current_version['embedding']) + np.array(new_version['embedding'])) / 2
                        merged_surprise_score = max(current_version['surprise_score'], new_version['surprise_score'])
                        existing_unit.add_version(merged_tokens, merged_embedding, merged_surprise_score)
                    updated = True
                    break
            if not updated:
                self.memory_units.append(new_unit)

    def detect_contradiction(self, new_version, existing_version):
        embedding_similarity = np.inner(np.array(new_version['embedding']), np.array(existing_version['embedding']))
        token_overlap = len(set(new_version['tokens']) & set(existing_version['tokens'])) / len(set(new_version['tokens']) | set(existing_version['tokens']))
        return embedding_similarity < 0.3 or token_overlap < 0.1

    def handle_contradiction(self, new_unit, existing_unit):
        new_version = new_unit.get_current_version()
        existing_unit.add_version(new_version['tokens'], np.array(new_version['embedding']), new_version['surprise_score'])

    def query_memory(self, query_text):
        query_embedding = self.sentence_transformer.encode([query_text])[0]
        query_embedding = legendre_paragraph_embedding(query_embedding.reshape(1, -1))
        
        retrieved_units = []
        for unit in self.memory_units:
            current_version = unit.get_current_version()
            similarity = np.inner(query_embedding, np.array(current_version['embedding']))
            if similarity > 0.5:  # Threshold for similarity
                retrieved_units.append((unit, similarity))
                unit.access_count += 1
                unit.last_access_time = time.time()
        
        return sorted(retrieved_units, key=lambda x: x[1], reverse=True)

    def generate_response(self, prompt, max_new_tokens=256):
        retrieved_memories = self.query_memory(prompt)
        
        context = ""
        for unit, similarity in retrieved_memories:
            current_version = unit.get_current_version()
            context += f"{' '.join(current_version['tokens'])} "
            if len(context.split()) > 100:
                break
        
        full_prompt = f"{context.strip()}\n\n{prompt}"
        
        messages = [{"role": "user", "content": full_prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(chat_prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        return outputs[0]["generated_text"]

# ------------------------------
# Example Usage 
# ------------------------------

# Initialize the EM-LLM System
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

# Example of accessing memory chain
if em_llm.memory_units:
    first_memory = em_llm.memory_units[0]
    print(f"Memory chain length: {len(first_memory.memory_chain.chain)}")
    print(f"Is chain valid: {first_memory.memory_chain.is_chain_valid()}")
    for block in first_memory.memory_chain.chain:
        print(f"Block hash: {block.hash}")        
        print(f"Block data: {block.data['tokens'][:10]}...")  # Print first 10 tokens
