import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List, Dict, Tuple, Optional
import re

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
    logits = logits.clone()

    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

def prepare_next_token(next_token: torch.Tensor) -> torch.Tensor:
    """
    Reshapes the token to prepare for input concatenation.
    """
    return next_token.view(1, 1)

def dynamic_decoding(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 50,
    beam_width: int = 3,
    lookahead_depth: int = 2,
    entropy_threshold: float = 0.7,
    confidence_threshold: float = 0.6,
    hesitation_entropy_threshold: float = 1.0,
    hesitation_confidence_threshold: float = 0.5,
    top_k: int = 50,
    top_p: float = 0.9,
    epsilon: float = 1e-8,
) -> Tuple[str, List[float], List[float]]:
    """
    Adaptive decoding function that dynamically switches between decoding strategies
    based on model's entropy and confidence metrics.
    """
    device = get_device()
    model.to(device)
    model.eval()

    # Optionally, script the model with TorchScript to optimize
    #model = torch.jit.script(model)

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
    hesitation_count = 0

    # Encode hesitation phrase
    hesitation_phrase = " let me think again..."
    hesitation_token_ids = tokenizer.encode(hesitation_phrase, return_tensors="pt").to(device)

    while total_generated_tokens < max_new_tokens:
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Enable mixed precision
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                last_token_logits = logits[0, -1, :]

                # Calculate probabilities
                probs = torch.softmax(last_token_logits, dim=-1)

                # Compute entropy
                entropy = -torch.sum(probs * torch.log(probs + epsilon)).item()

                # Get top two probabilities
                top_probs, top_indices = torch.topk(probs, 2)
                prob_diff = (top_probs[0] - top_probs[1]).item()

                # Compute confidence
                confidence = prob_diff

                # Log entropy and confidence
                entropy_list.append(entropy)
                confidence_list.append(confidence)
                print(f"Token {total_generated_tokens}: Entropy = {entropy:.4f}, Confidence = {confidence:.4f}")

                # Decision logic for adaptive decoding
                if entropy > hesitation_entropy_threshold and confidence < hesitation_confidence_threshold:
                    if hesitation_count < 2:
                        # Insert hesitation phrase for first two instances
                        print("High entropy and low confidence detected. Inserting hesitation phrase.")
                        generated_tokens.extend(hesitation_token_ids.squeeze().tolist())
                        total_generated_tokens += hesitation_token_ids.size(1)

                        # Update input_ids and attention_mask
                        input_ids = torch.cat([input_ids, hesitation_token_ids], dim=1)
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones_like(hesitation_token_ids).to(device)], dim=1
                        )
                        hesitation_count += 1
                        continue
                    else:
                        # Use beam search for subsequent instances
                        print("High entropy detected, using beam search.")
                        beam_output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=min(lookahead_depth, max_new_tokens - total_generated_tokens),
                            num_beams=beam_width,
                            num_return_sequences=beam_width,
                            early_stopping=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                        best_beam_idx = torch.argmax(beam_output.sequences_scores)
                        best_sequence = beam_output.sequences[best_beam_idx]

                        # Extract new tokens
                        new_tokens = best_sequence[len(input_ids[0]):]
                        generated_tokens.extend(new_tokens.tolist())
                        total_generated_tokens += new_tokens.size(0)

                        # Update input_ids and attention_mask
                        input_ids = torch.cat([input_ids, new_tokens.unsqueeze(0)], dim=1)
                        attention_mask = torch.cat(
                            [attention_mask, torch.ones((1, new_tokens.size(0)), dtype=torch.long).to(device)], dim=1
                        )

                        if tokenizer.eos_token_id in new_tokens:
                            print("EOS token detected. Terminating generation.")
                            break

                elif entropy > 0.3 and confidence < 0.7:
                    print("Moderate uncertainty detected, using top-k sampling.")
                    filtered_logits = top_k_top_p_filtering(
                        logits=last_token_logits, top_k=top_k, top_p=None
                    )
                    probabilities = torch.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                    next_token = prepare_next_token(next_token)

                    generated_tokens.append(next_token.item())
                    total_generated_tokens += 1

                    # Update input_ids and attention_mask
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), dtype=torch.long).to(device)], dim=1
                    )

                elif entropy > 0.3 and confidence < 0.5:
                    print("Moderate uncertainty detected, using nucleus sampling.")
                    filtered_logits = top_k_top_p_filtering(
                        logits=last_token_logits, top_k=None, top_p=top_p
                    )
                    probabilities = torch.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                    next_token = prepare_next_token(next_token)

                    generated_tokens.append(next_token.item())
                    total_generated_tokens += 1

                    # Update input_ids and attention_mask
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), dtype=torch.long).to(device)], dim=1
                    )

                else:
                    print("Low uncertainty detected, using greedy decoding.")
                    next_token = top_indices[0]
                    next_token = prepare_next_token(next_token)

                    generated_tokens.append(next_token.item())
                    total_generated_tokens += 1

                    # Update input_ids and attention_mask
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), dtype=torch.long).to(device)], dim=1
                    )

                    if next_token.item() == tokenizer.eos_token_id:
                        print("EOS token detected. Terminating generation.")
                        break

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text, entropy_list, confidence_list

