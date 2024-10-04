import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Determines the appropriate device (GPU or CPU) for computation.
    Uses Metal Performance Shaders (MPS) on Macs with M1/M2 chips if available.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    filter_value: float = -float('Inf')
) -> torch.Tensor:
    """
    Filters logits using top-k and top-p (nucleus) sampling.
    
    Args:
        logits (torch.Tensor): Logits distribution shape (vocab_size).
        top_k (Optional[int]): Keep only top_k tokens with highest probability.
        top_p (Optional[float]): Keep the top tokens with cumulative probability >= top_p.
        filter_value (float): Value to set for filtered logits.
        
    Returns:
        torch.Tensor: Filtered logits.
    """
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter sorted indices to original indexing
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

def calculate_entropy(probs: torch.Tensor) -> float:
    """
    Calculates entropy of probabilities.
    
    Args:
        probs (torch.Tensor): Probability distribution over the vocabulary.
        
    Returns:
        float: Entropy value.
    """
    epsilon = 1e-9  # To prevent log(0)
    entropy = -torch.sum(probs * torch.log(probs + epsilon)).item()
    return entropy

def calculate_varentropy(entropy_history: List[float]) -> float:
    """
    Calculates the variance of the entropy history.
    
    Args:
        entropy_history (List[float]): List of past entropy values.
        
    Returns:
        float: Variance of entropy.
    """
    if len(entropy_history) < 2:
        return 0.0  # Default value if insufficient data
    return float(np.var(entropy_history))

def resample(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    num_resamples: int,
    temperature: float,
    top_k: Optional[int] = None,  # Add top_k filtering for resampling
    top_p: Optional[float] = None  # Add top_p filtering for resampling
) -> int:
    """
    Generates multiple resamples by adding noise to logits and selects the most frequent token.
    
    Args:
        model (PreTrainedModel): The language model.
        input_ids (torch.Tensor): Current input IDs.
        attention_mask (torch.Tensor): Current attention mask.
        num_resamples (int): Number of resamples to generate.
        temperature (float): Temperature for scaling logits.
        top_k (Optional[int]): Top-k filtering parameter.
        top_p (Optional[float]): Top-p (nucleus) filtering parameter.
        
    Returns:
        int: Selected token ID.
    """
    token_ids = []  # Store generated token_ids
    device = input_ids.device

    for _ in range(num_resamples):
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            last_token_logits = logits[0, -1, :] / temperature

            # Apply filtering if specified
            if top_k or top_p:
                filtered_logits = top_k_top_p_filtering(last_token_logits, top_k=top_k, top_p=top_p)
            else:
                filtered_logits = last_token_logits

            # Add Gaussian noise
            noise = torch.randn_like(filtered_logits) * 0.1  # Scale noise appropriately
            noisy_logits = filtered_logits + noise

            # Sample from the noisy logits
            probabilities = torch.softmax(noisy_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            token_ids.append(next_token)

    # Return the most frequent token among resamples. Good default if no clear "best".
    return max(set(token_ids), key=token_ids.count)

def should_inject_cot(
    cot_injection_count: int,
    previous_token_text: str,  # Pass the decoded token text directly
    tokens_since_last_injection: int,
    max_cot_injections: int = 3,
    cot_trigger_phrase: str = "wait.."  # Customizable trigger phrase
) -> bool:
    """
    Determines whether to inject the Chain of Thought (CoT) phrase.
    
    Args:
        cot_injection_count (int): Number of times CoT has been injected.
        previous_token_text (str): Decoded text of the previous token.
        tokens_since_last_injection (int): Tokens generated since the last injection.
        max_cot_injections (int): Maximum number of CoT injections allowed.
        cot_trigger_phrase (str): Phrase that triggers CoT injection.
        
    Returns:
        bool: True if CoT should be injected, False otherwise.
    """
    return (
        cot_injection_count < max_cot_injections and
        tokens_since_last_injection > 2 and  # Inject after at least a few tokens
        previous_token_text == cot_trigger_phrase
    )

def dynamic_decoding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 50,
    beam_width: int = 3,
    entropy_bounds: Tuple[float, float] = (0.5, 1.5),
    confidence_levels: Dict[str, Tuple[float, float]] = {  # Adjusted default confidence levels
        "high": (0.7, float('inf')),
        "moderate": (0.4, 0.7),
        "low": (0.0, 0.4),
    },
    temperature_levels: Dict[str, float] = {  # Adjusted default temperature levels
        "greedy": 0.2,  # Lower for more deterministic
        "beam_search": 0.5,
        "nucleus_sampling": 0.8, # Slightly lower for better coherence
    },
    top_k: int = 50,
    top_p: float = 0.9,
    cot_phrase: str = "wait..", # Make CoT phrase a parameter
    max_cot_injections: int = 3, # Parameterize max CoT injections
    num_resamples: int = 5, # Number of resamples
    resampling_top_k: Optional[int] = 50,  # Optional top_k for resampling
    resampling_top_p: Optional[float] = 0.9, # Optional top_p for resampling
) -> Tuple[str, List[float], List[float]]:
    """
    Adaptive decoding function that dynamically switches between decoding strategies
    based on model's entropy and confidence metrics.

    Args:
        model (PreTrainedModel): The language model.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        messages (List[Dict[str, str]]): Input messages with roles and content.
        max_new_tokens (int): Maximum number of tokens to generate.
        beam_width (int): Number of beams for beam search.
        entropy_bounds (Tuple[float, float]): Bounds to categorize entropy.
        confidence_levels (Dict[str, Tuple[float, float]]): Confidence level mappings.
        temperature_levels (Dict[str, float]): Temperature settings for strategies.
        top_k (int): Top-k value for filtering.
        top_p (float): Top-p value for nucleus sampling.
        cot_phrase (str): Phrase to inject as Chain of Thought.
        max_cot_injections (int): Maximum number of CoT injections allowed.
        num_resamples (int): Number of resamples for the resampling strategy.
        resampling_top_k (Optional[int]): Top-k filtering parameter for resampling.
        resampling_top_p (Optional[float]): Top-p (nucleus) filtering parameter for resampling.
    
    Returns:
        Tuple[str, List[float], List[float]]: Generated text, list of entropies, list of confidences.
    """
    device = get_device()
    model.to(device)
    model.eval()

    # Prepare input text
    input_text = ""
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        input_text += f"{role}: {content}\n"
    input_text += "assistant:"

    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    # Set pad_token_id if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize variables
    generated_tokens = []
    total_generated_tokens = 0
    entropy_list = []
    confidence_list = []
    entropy_history = []
    varentropy = 0.0
    cot_injection_count = 0
    tokens_since_last_injection = 0
    previous_token_id = None

    # Encode CoT phrase (optimize: do this only if cot_trigger_phrase is not empty)
    if cot_phrase:
        cot_token_ids = tokenizer.encode(cot_phrase, add_special_tokens=False, return_tensors="pt").to(device)
        cot_token_ids = cot_token_ids.squeeze().tolist()
    else:
        cot_token_ids = []  # Or None, if you prefer

    # Helper function to categorize entropy and varentropy
    def categorize(value, bounds):
        if value < bounds[0]:
            return "low"
        elif value < bounds[1]:
            return "moderate"
        else:
            return "high"

    # Strategy Mapping  (slightly optimized structure)
    strategy_mapping = {
        "high": {
            "low": {"low": "greedy", "moderate": "greedy", "high": "greedy"},
            "moderate": {"low": "beam_search", "moderate": "beam_search", "high": "beam_search"},
            "high": {"low": "nucleus_sampling", "moderate": "nucleus_sampling", "high": "nucleus_sampling"},
        },
        "moderate": {
            "low": {"low": "greedy", "moderate": "greedy", "high": "greedy"},
            "moderate": {"low": "beam_search", "moderate": "beam_search", "high": "beam_search"},
            "high": {"low": "nucleus_sampling", "moderate": "nucleus_sampling", "high": "nucleus_sampling"},
        },
        "low": {
            "low": {"low": "greedy", "moderate": "greedy", "high": "greedy"},
            "moderate": {"low": "beam_search", "moderate": "beam_search", "high": "beam_search"},
            "high": {"low": "nucleus_sampling", "moderate": "nucleus_sampling", "high": "nucleus_sampling"},
        },
    }

    # Strategy Execution Functions
    def greedy_strategy(last_token_logits):
        return torch.argmax(last_token_logits).item()

    def beam_search_strategy(last_token_logits):  # Added last_token_logits for consistency
        beam_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,  # Generate one token at a time
            num_beams=beam_width,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            temperature=temperature_levels["beam_search"],
        )
        return beam_output[0, -1].item()

    def nucleus_sampling_strategy(last_token_logits):
        filtered_logits = top_k_top_p_filtering(
            last_token_logits, top_k=None, top_p=top_p, filter_value=-float('Inf')
        )
        probabilities = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        return next_token

    def resampling_strategy(last_token_logits):  # Added last_token_logits for consistency
        return resample(
            model,
            input_ids,
            attention_mask,
            num_resamples=num_resamples,
            temperature=temperature_levels["nucleus_sampling"],
            top_k=resampling_top_k,
            top_p=resampling_top_p,
        )

    strategy_functions = {
        "greedy": greedy_strategy,
        "beam_search": beam_search_strategy,
        "nucleus_sampling": nucleus_sampling_strategy,
        "resampling": resampling_strategy,
    }

    # Initialize strategy usage counters for logging
    strategy_usage = {
        "greedy": 0,
        "beam_search": 0,
        "nucleus_sampling": 0,
        "resampling": 0,
    }

    while total_generated_tokens < max_new_tokens:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (batch_size, seq_len, vocab_size)
                last_token_logits = logits[0, -1, :] / temperature_levels["greedy"]  # Apply temperature

                # Calculate probabilities
                probs = torch.softmax(last_token_logits, dim=-1)

                # Compute entropy
                entropy = calculate_entropy(probs)
                entropy_history.append(entropy)
                varentropy = calculate_varentropy(entropy_history)

                # Get top two probabilities
                top_probs, top_indices = torch.topk(probs, 2)
                prob_diff = (top_probs[0] - top_probs[1]).item()
                confidence = prob_diff

                confidence_list.append(confidence)
                entropy_list.append(entropy)

                # Determine confidence level
                confidence_level = "low"  # Default
                for level, (lower, upper) in confidence_levels.items():
                    if lower <= confidence < upper:
                        confidence_level = level
                        break

                # CoT Injection (check if cot_token_ids is not empty)
                previous_token_text = tokenizer.decode([previous_token_id], skip_special_tokens=True).strip() if previous_token_id is not None else ""
                inject_cot = False
                if cot_phrase:
                    inject_cot = should_inject_cot(
                        cot_injection_count=cot_injection_count,
                        previous_token_text=previous_token_text,
                        tokens_since_last_injection=tokens_since_last_injection,
                        max_cot_injections=max_cot_injections,
                        cot_trigger_phrase=cot_phrase
                    )

                if inject_cot:
                    logger.info("Injecting Chain of Thought (CoT) phrase.")
                    generated_tokens.extend(cot_token_ids)
                    total_generated_tokens += len(cot_token_ids)

                    # Update input_ids and attention_mask
                    cot_tensor = torch.tensor(cot_token_ids, dtype=torch.long).unsqueeze(0).to(device)
                    input_ids = torch.cat([input_ids, cot_tensor], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones_like(cot_tensor).to(device)], dim=1)

                    cot_injection_count += 1
                    tokens_since_last_injection = 0
                    previous_token_id = cot_token_ids[-1] if cot_token_ids else previous_token_id
                    continue

                # Categorize entropy and varentropy
                entropy_category = categorize(entropy, entropy_bounds)
                varentropy_category = categorize(varentropy, entropy_bounds)

                # Strategy Mapping using nested dictionary lookup
                try:
                    recommended_strategy = strategy_mapping[confidence_level][entropy_category][varentropy_category]
                except KeyError:
                    recommended_strategy = "greedy"  # Default
                strategy_temperature = temperature_levels.get(recommended_strategy, temperature_levels["greedy"])  # Get temperature, default to greedy

                logger.info(f"Strategy selected: {recommended_strategy} with temperature {strategy_temperature}")
                strategy_usage[recommended_strategy] += 1

                # Execute the selected strategy
                if recommended_strategy in strategy_functions:
                    if recommended_strategy in ["greedy", "nucleus_sampling", "resampling"]:
                        next_token_id = strategy_functions[recommended_strategy](last_token_logits)
                    elif recommended_strategy == "beam_search":
                        next_token_id = strategy_functions[recommended_strategy](last_token_logits)
                    else:
                        # Default to greedy if strategy is unrecognized
                        next_token_id = strategy_functions["greedy"](last_token_logits)
                else:
                    # Default to greedy if strategy is unrecognized
                    next_token_id = strategy_functions["greedy"](last_token_logits)

                # Update tokens_since_last_injection
                tokens_since_last_injection += 1

                # Update previous_token_id before appending to generated_tokens
                previous_token_id = next_token_id

                # Append to generated_tokens
                generated_tokens.append(next_token_id)
                total_generated_tokens += 1

                # Update input_ids and attention_mask
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_tensor).to(device)], dim=1)

                # Check for EOS token
                if next_token_id == tokenizer.eos_token_id:
                    logger.info("EOS token detected. Terminating generation.")
                    break

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Log strategy usage statistics
    logger.info("Strategy Usage Statistics:")
    for strategy, count in strategy_usage.items():
        logger.info(f"  {strategy}: {count} times")

    return generated_text, entropy_list, confidence_list

# Example Usage
if __name__ == "__main__":
    # Initialize model and tokenizer
    model_name = "gpt2"  # Replace with "unsloth/Llama-3.2-1B-Instruct" or your desired model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example messages
    messages = [
        {
            "role": "user",
            "content": "Translate the following English text to French: 'Hello, how are you today?'"
        }
    ]

    # Perform adaptive decoding
    generated_text, entropy_list, confidence_list = dynamic_decoding(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=50,
        beam_width=3,
        entropy_bounds=(0.5, 1.5),  # Adjust based on your model's characteristics
        confidence_levels={
            "high": (0.7, float('inf')),
            "moderate": (0.4, 0.7),
            "low": (0.0, 0.4),
        },
        temperature_levels={
            "greedy": 0.2,  # Lower for more deterministic
            "beam_search": 0.5,
            "nucleus_sampling": 0.8, # Slightly lower for better coherence
        },
        top_k=50,
        top_p=0.9,
        cot_phrase="wait..", # Make CoT phrase a parameter
        max_cot_injections=3, # Parameterize max CoT injections
        num_resamples=5, # Number of resamples
        resampling_top_k=50,  # Optional top_k for resampling
        resampling_top_p=0.9, # Optional top_p for resampling
    )

    # Output the results
    print("\nGenerated Text:")
    print(generated_text)
    print("\nEntropy History:")
    print(entropy_list)
    print("\nConfidence History:")
    print(confidence_list)

