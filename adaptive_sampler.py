"""
Author : Sethu Iyer
Date : 2024-07-28
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
import numpy as np
from typing import List, Tuple
from collections import Counter
import threading
import time
from rich import print
from rich.markdown import Markdown
from rich.table import Table
import matplotlib.pyplot as plt

# Constants
LN_2 = np.log(2)

# Helper functions for entropy and varentropy calculations
def calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1):
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=axis)
    return entropy, varentropy

def calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor):
    """Calculate metrics required for the sampling strategy."""
    entropy, varentropy = calculate_varentropy_logsoftmax(logits)

    attention_probs = F.softmax(attention_scores, dim=-1)
    attn_entropy = -torch.sum(
        attention_probs * torch.log2(torch.clamp(attention_probs, min=1e-10, max=1.0)), dim=-1
    )
    attn_varentropy = torch.var(attn_entropy, dim=1)

    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(
        torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2)
    )

    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    # Compute confidence (A - B)
    last_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
    probs = F.softmax(last_logits, dim=-1)
    top_probs, _ = probs.topk(2, dim=-1)
    A = top_probs[:, 0]  # Confidence of the top token
    B = top_probs[:, 1]  # Confidence of the second top token
    confidence = A - B

    return {
        "logits_entropy": torch.mean(entropy),
        "logits_varentropy": torch.mean(varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength,
        "confidence": torch.mean(confidence),
    }

def _sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int, min_p: float):
    """Sample the next token using adjusted sampling parameters."""
    batch_size = logits.shape[0]
    logit = logits[:, -1, :]  # Assume logits is (batch_size, seq_len, vocab_size)

    # Apply temperature scaling
    logit = logit / temperature

    # Convert logits to probabilities
    probs = F.softmax(logit, dim=-1)

    # Apply min_p filtering
    if min_p > 0.0:
        p_max, _ = probs.max(dim=-1, keepdim=True)
        indices_to_remove = probs < (min_p * p_max)
        logit = logit.masked_fill(indices_to_remove, float('-inf'))
        probs = F.softmax(logit, dim=-1)

    # Apply top-k filtering
    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=-1)
    else:
        top_k_probs = probs
        top_k_indices = torch.arange(probs.size(-1), device=probs.device).unsqueeze(0).expand(batch_size, -1)

    # Apply top-p (nucleus) filtering
    sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)

    # Normalize the probabilities
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from the adjusted probabilities
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token_indices = sorted_indices.gather(dim=-1, index=next_token)

    # Map back to original vocabulary indices
    next_token_g = top_k_indices.gather(dim=-1, index=next_token_indices)

    return next_token_g

@dataclass(frozen=True)
class SamplerConfig:
    """
    Configuration for the sampler with all hyperparameters.
    """
    temp: float = 0.666
    top_p: float = 0.90
    top_k: int = 27
    min_p: float = 0.03

    low_ent_thresh: float = 0.1
    low_vent_thresh: float = 0.1
    med_ent_thresh: float = 3.0
    high_ent_thresh: float = 5.0
    high_vent_thresh: float = 5.0

    # Confidence bands thresholds
    confidence_very_low_thresh: float = 0.05
    confidence_low_thresh: float = 0.15
    confidence_moderate_thresh: float = 0.25

    # Coefficients for temperature and scoring adjustments
    helv_attn_ent_offset: float = 1.3
    helv_attn_ent_coef: float = 0.2

    lehv_interaction_strength_offset: float = 1.2
    lehv_interaction_strength_coef: float = 0.3

    hehv_attn_ent_coef: float = 0.2
    hehv_attn_vent_offset: float = 2.0
    hehv_attn_vent_coef: float = 0.5

    n_adaptive_samples: int = 5

    # Adaptive sampling parameters
    ada_temp_logits: float = 0.3
    ada_temp_attn: float = 0.2
    ada_temp_agree: float = 0.2
    ada_temp_confidence: float = 0.5  # New parameter for confidence

    ada_top_p: float = 0.1
    ada_top_k_int: float = 0.3
    ada_top_k_agree: float = 0.2
    ada_min_p: float = 0.5

    ada_score_logits_ent: float = 0.1
    ada_score_attn_ent: float = 0.2
    ada_score_logits_vent: float = 0.3
    ada_score_attn_vent: float = 0.4
    ada_score_agree: float = 0.5
    ada_score_int: float = 0.6
    ada_score_confidence: float = 0.5  # New parameter for confidence

    # Weights for metrics in relevance score calculation
    metric_weights: dict = field(default_factory=lambda: {
        "logits_entropy": 1.0,
        "logits_varentropy": 1.0,
        "attn_entropy": 1.0,
        "attn_varentropy": 1.0,
        "agreement": 1.0,
        "interaction_strength": 1.0,
        "confidence": 1.0,
    })

# Define sampling strategies as dataclasses
@dataclass
class SamplingStrategy:
    name: str
    xp_cost: float  # XP cost attribute

    def condition(self, metrics, cfg):
        """Determine if this strategy should be applied based on the metrics and configuration."""
        raise NotImplementedError

    def compute_relevance(self, metrics, cfg):
        """Compute the relevance score for the strategy based on the current metrics."""
        raise NotImplementedError

    def action(self, logits, gen_tokens, metrics, cfg):
        """Define the action to take when this strategy is applied."""
        raise NotImplementedError

@dataclass
class FlowingWithUnspokenIntent(SamplingStrategy):
    name: str = "Flowing with Unspoken Intent"
    xp_cost: float = 1.0  # Minimal cost for greedy decoding

    def condition(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()
        return (
            ent < cfg.low_ent_thresh
            and vent < cfg.low_vent_thresh
            and confidence >= cfg.confidence_moderate_thresh
        )

    def compute_relevance(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()

        relevance_score = 0.0
        # Calculate deviation below thresholds
        if ent < cfg.low_ent_thresh:
            deviation = cfg.low_ent_thresh - ent
            relevance_score += deviation * cfg.metric_weights["logits_entropy"]
        if vent < cfg.low_vent_thresh:
            deviation = cfg.low_vent_thresh - vent
            relevance_score += deviation * cfg.metric_weights["logits_varentropy"]
        if confidence >= cfg.confidence_moderate_thresh:
            deviation = confidence - cfg.confidence_moderate_thresh
            relevance_score += deviation * cfg.metric_weights["confidence"]

        return relevance_score

    def action(self, logits, gen_tokens, metrics, cfg):
        # Greedy sampling
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token

@dataclass
class ProceedingWithConfidence(SamplingStrategy):
    name: str = "Proceeding with Confidence"
    xp_cost: float = 1.5  # Slightly higher cost due to sampling

    def condition(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()
        return (
            ent < cfg.low_ent_thresh
            and vent < cfg.low_vent_thresh
            and cfg.confidence_low_thresh <= confidence < cfg.confidence_moderate_thresh
        )

    def compute_relevance(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()

        relevance_score = 0.0
        # Calculate deviation below thresholds
        if ent < cfg.low_ent_thresh:
            deviation = cfg.low_ent_thresh - ent
            relevance_score += deviation * cfg.metric_weights["logits_entropy"]
        if vent < cfg.low_vent_thresh:
            deviation = cfg.low_vent_thresh - vent
            relevance_score += deviation * cfg.metric_weights["logits_varentropy"]
        if confidence >= cfg.confidence_low_thresh:
            deviation = confidence - cfg.confidence_low_thresh
            relevance_score += deviation * cfg.metric_weights["confidence"]

        return relevance_score

    def action(self, logits, gen_tokens, metrics, cfg):
        # Low-temperature sampling
        confidence = metrics["confidence"].item()
        temperature = max(0.7, cfg.temp * (1 - cfg.ada_temp_confidence * confidence))
        next_token = _sample(
            logits,
            temperature=temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p
        )
        return next_token

@dataclass
class TreadingCarefully(SamplingStrategy):
    name: str = "Treading Carefully, Asking Clarifying Questions"
    xp_cost: float = 2.0  # Additional cost for inserting tokens and adjustments
    clarifying_question_token: int = 2564  # Adjust as needed

    def condition(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()
        return (
            ent > cfg.high_ent_thresh
            and vent < cfg.low_vent_thresh
            and confidence < cfg.confidence_very_low_thresh
        )

    def compute_relevance(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()

        relevance_score = 0.0
        # Calculate deviation above thresholds
        if ent > cfg.high_ent_thresh:
            deviation = ent - cfg.high_ent_thresh
            relevance_score += deviation * cfg.metric_weights["logits_entropy"]
        if vent < cfg.low_vent_thresh:
            deviation = cfg.low_vent_thresh - vent
            relevance_score += deviation * cfg.metric_weights["logits_varentropy"]
        if confidence < cfg.confidence_very_low_thresh:
            deviation = cfg.confidence_very_low_thresh - confidence
            relevance_score += deviation * cfg.metric_weights["confidence"]

        return relevance_score

    def action(self, logits, gen_tokens, metrics, cfg):
        if not (gen_tokens[:, -1] == self.clarifying_question_token).any():
            # Insert a clarifying question token
            next_token = torch.full(
                (gen_tokens.size(0), 1),
                self.clarifying_question_token,
                dtype=torch.long,
                device=logits.device
            )
        else:
            # Increase temperature based on attention entropy and confidence
            attn_ent = metrics["attn_entropy"].item()
            confidence = metrics["confidence"].item()
            temp_adj = cfg.helv_attn_ent_offset + cfg.helv_attn_ent_coef * attn_ent - cfg.ada_temp_confidence * confidence
            temperature = min(1.5, cfg.temp * temp_adj)
            next_token = _sample(
                logits,
                temperature=temperature,
                top_p=cfg.top_p * (1 - confidence),  # Reduce top_p inversely to confidence
                top_k=cfg.top_k,
                min_p=cfg.min_p
            )
        return next_token

@dataclass
class ExploringForks(SamplingStrategy):
    name: str = "Exploring Forks in the Path"
    xp_cost: float = 3.0  # Higher cost due to increased temperature and top-k

    def condition(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()
        return (
            ent < cfg.high_ent_thresh
            and vent > cfg.high_vent_thresh
            and cfg.confidence_low_thresh <= confidence < cfg.confidence_moderate_thresh
        )

    def compute_relevance(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()

        relevance_score = 0.0
        # Calculate deviation below and above thresholds
        if ent < cfg.high_ent_thresh:
            deviation = cfg.high_ent_thresh - ent
            relevance_score += deviation * cfg.metric_weights["logits_entropy"]
        if vent > cfg.high_vent_thresh:
            deviation = vent - cfg.high_vent_thresh
            relevance_score += deviation * cfg.metric_weights["logits_varentropy"]
        if confidence >= cfg.confidence_low_thresh:
            deviation = confidence - cfg.confidence_low_thresh
            relevance_score += deviation * cfg.metric_weights["confidence"]

        return relevance_score

    def action(self, logits, gen_tokens, metrics, cfg):
        agreement = metrics["agreement"].item()
        interaction_strength = metrics["interaction_strength"].item()
        confidence = metrics["confidence"].item()
        temp_adj = cfg.lehv_interaction_strength_offset + cfg.lehv_interaction_strength_coef * interaction_strength - cfg.ada_temp_confidence * confidence
        temperature = min(1.5, cfg.temp * temp_adj)
        top_k_adj = max(5, int(cfg.top_k * (1 + 0.5 * (1 - agreement))))
        next_token = _sample(
            logits,
            temperature=temperature,
            top_p=cfg.top_p,
            top_k=top_k_adj,
            min_p=cfg.min_p
        )
        return next_token

@dataclass
class NavigatingUncertainty(SamplingStrategy):
    name: str = "Navigating Uncertainty"
    xp_cost: float = 5.0  # Moderate resampling increases cost

    def condition(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()
        return (
            ent > cfg.med_ent_thresh
            and vent > cfg.high_vent_thresh
            and cfg.confidence_very_low_thresh <= confidence < cfg.confidence_low_thresh
        )

    def compute_relevance(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()

        relevance_score = 0.0
        # Calculate deviation above thresholds
        if ent > cfg.med_ent_thresh:
            deviation = ent - cfg.med_ent_thresh
            relevance_score += deviation * cfg.metric_weights["logits_entropy"]
        if vent > cfg.high_vent_thresh:
            deviation = vent - cfg.high_vent_thresh
            relevance_score += deviation * cfg.metric_weights["logits_varentropy"]
        if confidence < cfg.confidence_low_thresh:
            deviation = cfg.confidence_low_thresh - confidence
            relevance_score += deviation * cfg.metric_weights["confidence"]

        return relevance_score

    def action(self, logits, gen_tokens, metrics, cfg):
        # Moderate resampling strategy
        attn_ent = metrics["attn_entropy"].item()
        attn_vent = metrics["attn_varentropy"].item()
        confidence = metrics["confidence"].item()

        # Adjust temperature and top_p
        temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent - cfg.ada_temp_confidence * confidence
        temperature = max(1.5, cfg.temp * temp_adj)
        top_p_adj = min(0.95, cfg.top_p + cfg.hehv_attn_ent_coef * attn_ent)

        samples = []
        sample_scores = []

        for _ in range(cfg.n_adaptive_samples):
            sample_next_token = _sample(
                logits,
                temperature=temperature,
                top_p=top_p_adj,
                top_k=cfg.top_k,
                min_p=cfg.min_p
            )

            # Compute score for sample
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            token_one_hot = F.one_hot(sample_next_token.squeeze(-1), num_classes=logits.size(-1))
            log_prob = torch.sum(log_probs * token_one_hot, dim=-1)
            confidence_score = (
                (1 - metrics["logits_entropy"]) * cfg.ada_score_logits_ent
                + (1 - metrics["attn_entropy"]) * cfg.ada_score_attn_ent
                + (1 - metrics["logits_varentropy"]) * cfg.ada_score_logits_vent
                + (1 - metrics["attn_varentropy"]) * cfg.ada_score_attn_vent
                + metrics["agreement"] * cfg.ada_score_agree
                + metrics["interaction_strength"] * cfg.ada_score_int
                + metrics["confidence"] * cfg.ada_score_confidence
            )
            score = log_prob + confidence_score
            samples.append(sample_next_token)
            sample_scores.append(score)

        sample_scores_tensor = torch.stack(sample_scores)
        best_sample_idx = torch.argmax(sample_scores_tensor, dim=0)
        next_token = samples[best_sample_idx]
        return next_token

@dataclass
class ResamplingInTheMist(SamplingStrategy):
    name: str = "Resampling in the Mist"
    xp_cost: float = 7.0  # Aggressive resampling is costly

    def condition(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()
        return (
            ent > cfg.med_ent_thresh
            and vent > cfg.high_vent_thresh
            and confidence < cfg.confidence_very_low_thresh
        )

    def compute_relevance(self, metrics, cfg):
        ent = metrics["logits_entropy"].item()
        vent = metrics["logits_varentropy"].item()
        confidence = metrics["confidence"].item()

        relevance_score = 0.0
        # Calculate deviation above thresholds
        if ent > cfg.med_ent_thresh:
            deviation = ent - cfg.med_ent_thresh
            relevance_score += deviation * cfg.metric_weights["logits_entropy"]
        if vent > cfg.high_vent_thresh:
            deviation = vent - cfg.high_vent_thresh
            relevance_score += deviation * cfg.metric_weights["logits_varentropy"]
        if confidence < cfg.confidence_very_low_thresh:
            deviation = cfg.confidence_very_low_thresh - confidence
            relevance_score += deviation * cfg.metric_weights["confidence"]

        return relevance_score

    def action(self, logits, gen_tokens, metrics, cfg):
        # Aggressive resampling strategy
        attn_ent = metrics["attn_entropy"].item()
        attn_vent = metrics["attn_varentropy"].item()
        confidence = metrics["confidence"].item()

        # Adjust temperature and top_p
        temp_adj = cfg.hehv_attn_vent_offset + cfg.hehv_attn_vent_coef * attn_vent - cfg.ada_temp_confidence * confidence
        temperature = max(2.0, cfg.temp * temp_adj)
        top_p_adj = min(1.0, cfg.top_p + cfg.hehv_attn_ent_coef * attn_ent)

        samples = []
        sample_scores = []

        for _ in range(cfg.n_adaptive_samples):
            sample_next_token = _sample(
                logits,
                temperature=temperature,
                top_p=top_p_adj,
                top_k=cfg.top_k,
                min_p=cfg.min_p
            )

            # Compute score for sample
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            token_one_hot = F.one_hot(sample_next_token.squeeze(-1), num_classes=logits.size(-1))
            log_prob = torch.sum(log_probs * token_one_hot, dim=-1)
            confidence_score = (
                (1 - metrics["logits_entropy"]) * cfg.ada_score_logits_ent
                + (1 - metrics["attn_entropy"]) * cfg.ada_score_attn_ent
                + (1 - metrics["logits_varentropy"]) * cfg.ada_score_logits_vent
                + (1 - metrics["attn_varentropy"]) * cfg.ada_score_attn_vent
                + metrics["agreement"] * cfg.ada_score_agree
                + metrics["interaction_strength"] * cfg.ada_score_int
                + metrics["confidence"] * cfg.ada_score_confidence
            )
            score = log_prob + confidence_score
            samples.append(sample_next_token)
            sample_scores.append(score)

        sample_scores_tensor = torch.stack(sample_scores)
        best_sample_idx = torch.argmax(sample_scores_tensor, dim=0)
        next_token = samples[best_sample_idx]
        return next_token

@dataclass
class AdaptiveSampling(SamplingStrategy):
    name: str = "Adaptive Sampling"
    xp_cost: float = 4.0  # Default adaptive strategy cost

    def condition(self, metrics, cfg):
        # This strategy is used when none of the above conditions are met
        return True  # Always true if reached

    def compute_relevance(self, metrics, cfg):
        # Since Adaptive Sampling is the default, assign a base relevance score
        return 1.0

    def action(self, logits, gen_tokens, metrics, cfg):
        ent = metrics["logits_entropy"]
        vent = metrics["logits_varentropy"]
        attn_ent = metrics["attn_entropy"]
        attn_vent = metrics["attn_varentropy"]
        agreement = metrics["agreement"]
        interaction_strength = metrics["interaction_strength"]
        confidence = metrics["confidence"]

        logits_uncertainty = ent + vent
        attn_uncertainty = attn_ent + attn_vent

        # Compute temperature (scalar)
        temperature_tensor = cfg.temp * (
            1
            + cfg.ada_temp_logits * logits_uncertainty
            + cfg.ada_temp_attn * attn_uncertainty
            - cfg.ada_temp_agree * agreement
            - cfg.ada_temp_confidence * confidence
        )
        temperature = temperature_tensor.item()  # Convert tensor to float

        # Compute top_p (scalar)
        top_p_tensor = torch.clamp(
            cfg.top_p * (1 + cfg.ada_top_p * attn_vent),
            min=0.1,
            max=1.0
        )
        top_p = top_p_tensor.item()  # Convert tensor to float

        # Compute top_k (integer)
        top_k_tensor = torch.clamp(
            torch.round(
                cfg.top_k
                * (
                    1
                    + cfg.ada_top_k_int * interaction_strength
                    - cfg.ada_top_k_agree * agreement
                )
            ),
            min=1,
            max=100,
        )
        top_k = int(top_k_tensor.item())  # Convert tensor to int

        # Compute min_p (scalar)
        min_p_tensor = torch.clamp(
            cfg.min_p * (1 - cfg.ada_min_p * logits_uncertainty),
            min=0.01,
            max=0.5
        )
        min_p = min_p_tensor.item()  # Convert tensor to float

        # Now proceed with the sampling process using the computed parameters
        samples = []
        sample_scores = []
        for _ in range(cfg.n_adaptive_samples):
            sample_next_token = _sample(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
            )

            # Compute score for sample
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            token_one_hot = F.one_hot(sample_next_token.squeeze(-1), num_classes=logits.size(-1))
            log_prob = torch.sum(log_probs * token_one_hot, dim=-1)
            confidence_score = (
                (1 - ent) * cfg.ada_score_logits_ent
                + (1 - attn_ent) * cfg.ada_score_attn_ent
                + (1 - vent) * cfg.ada_score_logits_vent
                + (1 - attn_vent) * cfg.ada_score_attn_vent
                + agreement * cfg.ada_score_agree
                + interaction_strength * cfg.ada_score_int
                + confidence * cfg.ada_score_confidence
            )
            score = log_prob + confidence_score
            samples.append(sample_next_token)
            sample_scores.append(score)

        # Select the sample with the highest score
        sample_scores_tensor = torch.stack(sample_scores)
        best_sample_idx = torch.argmax(sample_scores_tensor, dim=0)
        next_token = samples[best_sample_idx]
        return next_token

# Updated sample function utilizing the strategies with computational budget
def sample(gen_tokens, logits, attention_scores, cfg, remaining_xp):
    """Select the next token based on entropy, varentropy, and confidence metrics, considering XP budget."""
    metrics = calculate_metrics(logits, attention_scores)

    # List of strategies with their indices
    strategies_with_indices = [
        (1, FlowingWithUnspokenIntent()),
        (2, ProceedingWithConfidence()),
        (3, TreadingCarefully()),
        (4, ExploringForks()),
        (5, NavigatingUncertainty()),
        (6, ResamplingInTheMist()),
        (7, AdaptiveSampling()),  # Should be the last one as a fallback
    ]

    # Initialize an empty list to store strategies along with their utility scores
    strategy_options = []

    # Evaluate each strategy
    for idx, strategy in strategies_with_indices:
        # Check if the strategy's condition is met and if we have enough XP
        if strategy.condition(metrics, cfg) and strategy.xp_cost <= remaining_xp:
            # Compute the Relevance Score for the strategy
            relevance_score = strategy.compute_relevance(metrics, cfg)
            # Compute the Utility Score
            utility_score = relevance_score / strategy.xp_cost
            # Store the strategy along with its utility score and index
            strategy_options.append((utility_score, idx, strategy))

    if not strategy_options:
        # If no strategies are affordable, default to greedy decoding
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return next_token, 0, remaining_xp

    # Select the strategy with the highest utility score
    best_strategy = max(strategy_options, key=lambda x: x[0])
    best_utility, idx, selected_strategy = best_strategy
    next_token = selected_strategy.action(logits, gen_tokens, metrics, cfg)

    # Deduct the XP cost
    remaining_xp -= selected_strategy.xp_cost

    return next_token, idx, remaining_xp  # Return the strategy index and updated XP

# ThinkingAssistant class with XP budget integration
class ThinkingAssistant:
    def __init__(self, model=None, tokenizer=None, model_name="gpt2", device="cuda"):
        self.device = device

        # Allow user to pass pre-loaded model and tokenizer, or load them if not provided
        if model and tokenizer:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.sys_message = ''' 
You are an AI trained to provide clear and informative responses. For any instruction or question, you will provide a thorough and detailed answer.
If you do not know the answer, politely inform the user and suggest seeking further assistance.
'''

    def format_prompt(self, instruction):
        # Format the instruction using a chat template with system and user roles
        messages = [
            {"role": "system", "content": self.sys_message.strip()},
            {"role": "user", "content": instruction.strip()}
        ]
        # For simplicity, concatenate the messages
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return prompt

    def generate_response(self, instruction, max_new_tokens=512):
        # Prepare the chat-style prompt
        prompt = self.format_prompt(instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Initialize generation parameters and prepare the sampling configuration
        cfg = SamplerConfig()
        gen_tokens = inputs['input_ids']

        # Initialize the strategy index tracker (for monitoring purposes)
        strategy_indices = []
        generated_text = ""
        strategy_names = []
        strategy_map = {
            1: "Flowing with Unspoken Intent",
            2: "Proceeding with Confidence",
            3: "Treading Carefully",
            4: "Exploring Forks in the Path",
            5: "Navigating Uncertainty",
            6: "Resampling in the Mist",
            7: "Adaptive Sampling",
            0: "Default"
        }

        # Initialize XP budget
        total_xp = 100.0  # Starting XP
        remaining_xp = total_xp

        # Calculate per-token minimum XP to avoid running out
        min_xp_per_token = total_xp / max_new_tokens

        # Generate tokens step by step using the custom sampling strategy
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if remaining_xp < min_xp_per_token:
                    # If remaining XP is too low, use the least costly strategy
                    selected_strategy = FlowingWithUnspokenIntent()
                    next_token = selected_strategy.action(None, gen_tokens, None, cfg)  # logits and metrics are not available
                    strategy_idx = 1  # Index for FlowingWithUnspokenIntent
                    remaining_xp -= selected_strategy.xp_cost
                else:
                    # Get model outputs with attentions
                    outputs = self.model(gen_tokens, output_attentions=True)
                    logits = outputs.logits

                    # Gather attention scores
                    attention_scores = outputs.attentions  # Tuple of tensors
                    attention_scores = torch.stack(attention_scores, dim=0)
                    num_layers, batch_size, num_heads, seq_len, _ = attention_scores.size()
                    attention_scores = attention_scores.permute(1, 2, 0, 3, 4).reshape(batch_size, -1, seq_len, seq_len)

                    # Use the custom sampling method to choose the next token
                    next_token, strategy_idx, remaining_xp = sample(
                        gen_tokens, logits, attention_scores, cfg, remaining_xp
                    )

                strategy_indices.append(strategy_idx)
                strategy_names.append(strategy_map.get(strategy_idx, "Unknown"))

                # Append the next token to the generated tokens
                gen_tokens = torch.cat([gen_tokens, next_token], dim=-1)

                # Decode the newly generated token and append to generated text
                new_text = self.tokenizer.decode(
                    next_token[0],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                generated_text += new_text  # Do not strip spaces

                # Check if the next token is the end-of-sequence token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # If no XP remains, stop generation
                if remaining_xp <= 0:
                    break

        # Combine the initial prompt and the generated text
        full_text = self.tokenizer.decode(gen_tokens[0], skip_special_tokens=True).strip()
        return full_text, generated_text, strategy_names

    def pretty_print_response(self, instruction, max_new_tokens=512):
        from rich.console import Console
        console = Console()
        full_response, generated_text, strategies_used = self.generate_response(instruction, max_new_tokens)

        # Display the assistant's response using rich markdown
        console.print(Markdown(f"## Assistant's Response:\n{generated_text}"))

        # Display the strategies used in a table
        strategy_counts = Counter(strategies_used)
        table = Table(title="Strategies Used During Generation")

        table.add_column("Strategy", style="cyan")
        table.add_column("Count", style="magenta")

        for strategy, count in strategy_counts.items():
            table.add_row(strategy, str(count))

        console.print(table)

        # Plot the strategies used over time
        plt.figure(figsize=(10, 4))
        strategy_indices_numeric = [int(k) for k in strategies_used if k.isdigit()]
        plt.plot(range(len(strategy_indices_numeric)), strategy_indices_numeric, marker='o')
        plt.xlabel('Token Position')
        plt.ylabel('Strategy Used (Index)')
        plt.title('Strategies Used Over Token Generation')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def normal_print_response(self, instruction, max_new_tokens=512):
        full_response, generated_text, strategies_used = self.generate_response(instruction, max_new_tokens)
        print("Assistant's Response:")
        print(generated_text)

    def streaming_print_response(self, instruction, max_new_tokens=512, delay=0.1):
        from rich.console import Console
        console = Console()
        prompt = self.format_prompt(instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        cfg = SamplerConfig()
        gen_tokens = inputs['input_ids']

        strategy_names = []
        strategy_map = {
            1: "Flowing with Unspoken Intent",
            2: "Proceeding with Confidence",
            3: "Treading Carefully",
            4: "Exploring Forks in the Path",
            5: "Navigating Uncertainty",
            6: "Resampling in the Mist",
            7: "Adaptive Sampling",
            0: "Default"
        }

        generated_text = ""

        # Initialize XP budget
        total_xp = 100.0  # Starting XP
        remaining_xp = total_xp

        # Calculate per-token minimum XP to avoid running out
        min_xp_per_token = total_xp / max_new_tokens

        def generate_tokens():
            nonlocal gen_tokens, generated_text, remaining_xp
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    if remaining_xp < min_xp_per_token:
                        # If remaining XP is too low, use the least costly strategy
                        selected_strategy = FlowingWithUnspokenIntent()
                        next_token = selected_strategy.action(None, gen_tokens, None, cfg)  # logits and metrics are not available
                        strategy_idx = 1  # Index for FlowingWithUnspokenIntent
                        remaining_xp -= selected_strategy.xp_cost
                    else:
                        # Get model outputs with attentions
                        outputs = self.model(gen_tokens, output_attentions=True)
                        logits = outputs.logits

                        # Gather attention scores
                        attention_scores = outputs.attentions  # Tuple of tensors
                        attention_scores = torch.stack(attention_scores, dim=0)
                        num_layers, batch_size, num_heads, seq_len, _ = attention_scores.size()
                        attention_scores = attention_scores.permute(1, 2, 0, 3, 4).reshape(batch_size, -1, seq_len, seq_len)

                        # Use the custom sampling method to choose the next token
                        next_token, strategy_idx, remaining_xp = sample(
                            gen_tokens, logits, attention_scores, cfg, remaining_xp
                        )

                    strategy_names.append(strategy_map.get(strategy_idx, "Unknown"))

                    # Append the next token to the generated tokens
                    gen_tokens = torch.cat([gen_tokens, next_token], dim=-1)

                    # Decode the newly generated token and append to generated text
                    new_text = self.tokenizer.decode(
                        next_token[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    generated_text += new_text  # Do not strip spaces
                    console.print(new_text, end='', flush=True)

                    # Check if the next token is the end-of-sequence token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break

                    # If no XP remains, stop generation
                    if remaining_xp <= 0:
                        break

                    # Introduce delay for streaming effect
                    time.sleep(delay)

            # After generation, display strategies used
            strategy_counts = Counter(strategy_names)
            table = Table(title="Strategies Used During Generation")

            table.add_column("Strategy", style="cyan")
            table.add_column("Count", style="magenta")

            for strategy, count in strategy_counts.items():
                table.add_row(strategy, str(count))

            console.print(table)

            # Plot the strategies used over time
            plt.figure(figsize=(10, 4))
            strategy_indices_numeric = [int(k) for k in strategy_names if k.isdigit()]
            plt.plot(range(len(strategy_indices_numeric)), strategy_indices_numeric, marker='o')
            plt.xlabel('Token Position')
            plt.ylabel('Strategy Used (Index)')
            plt.title('Strategies Used Over Token Generation')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

        # Start token generation in a separate thread
        thread = threading.Thread(target=generate_tokens)
        thread.start()
        thread.join()

