"""
RLHF from Scratch on DistilGPT2

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_distilgpt2_tokenizer
from transformers import AutoTokenizer

def load_distilgpt2_tokenizer(model_name="sshleifer/tiny-gpt2"):
    # TODO: load and return the Hugging Face tokenizer for the given model name.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer

# Step 2 - load_distilgpt2_model
from transformers import AutoModelForCausalLM

def load_distilgpt2_model(model_name="sshleifer/tiny-gpt2"):
    # TODO: load a causal LM by name and return it in eval mode
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    return model

# Step 3 - set_pad_token_to_eos
def set_pad_token_to_eos(tokenizer):
    # TODO: assign tokenizer.pad_token = tokenizer.eos_token and return the tokenizer

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer

# Step 4 - generate_and_decode
def generate_and_decode(model, tokenizer, prompt, max_new_tokens=8):
    # TODO: tokenize prompt, generate continuation greedily, decode and return as a string

    # tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # move input to device of model
    inputs = {
        name: tensor.to(model.device)
        for name, tensor in inputs.items()
    }

    # greedy generation; model.generate returns (batch, seq)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    out = tokenizer.decode(output_ids[0])

    return out

# Step 5 - greedy_decode
import torch

def greedy_decode(logits):
    """Return the argmax token id from a single-row logits vector."""
    # TODO: return the token id with the largest logit as a Python int

    max_idx = torch.argmax(logits, axis = -1)

    return int(max_idx.item())

# Step 6 - sample_with_temperature
def sample_with_temperature(logits, temperature):
    # TODO: rescale logits by temperature, softmax, and sample one token id

    scaled_logits = logits / temperature
    norm_logits = scaled_logits - torch.max(scaled_logits, axis = -1)[0]

    exp_logits = torch.exp(norm_logits)
    probs = exp_logits / torch.sum(exp_logits, axis = -1)

    token_id = torch.multinomial(
        probs,
        num_samples=1,
    )

    return int(token_id.item())

# Step 7 - top_k_filter
def top_k_filter(logits, k):
    # TODO: keep the k largest entries of logits and set the rest to -inf.
    vocab_size = logits.shape[-1]
    if k >= vocab_size:
        return logits

    top_values, top_indices = torch.topk(logits, k=k, dim=-1)
    
    filtered_logits = torch.full_like(logits, -torch.inf)
    filtered_logits.scatter_(
        dim=-1,
        index=top_indices,
        src=top_values,
    )

    return filtered_logits

# Step 8 - top_p_filter
def top_p_filter(logits, p):
    # TODO: mask logits outside the smallest cumulative-probability nucleus of size p.

    logits = torch.tensor(logits)
    if p >= 1:
        return logits
    if p <= 0:
        return torch.full_like(logits, -torch.inf)

    # Sort
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # remove tokens after cumulative probability exceeds p.
    remove_mask = cumulative_probs > p

    # retain the first token with cum prob > p
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    filtered_logits = logits.clone()
    indices_to_remove = sorted_indices[remove_mask]
    filtered_logits[indices_to_remove] = -torch.inf

    return filtered_logits

# Step 9 - build_synthetic_instruction_dataset
def build_synthetic_instruction_dataset():
    # TODO: return a small in-memory list of {'prompt', 'response'} dicts for SFT

    data = [
        {
            "prompt": "How are you?",
            "response": "I am doing well, thank you. How can I help you?",
        },
        {
            "prompt": "I like apples.",
            "response": "Apples are nutritious and can be used in many recipes.",
        },
        {
            "prompt": "How should I respond to an illegal request?",
            "response": "Decline the request and suggest a safe, legal alternative.",
        },
        {
            "prompt": "Who are you",
            "response": "I am burger king",
        },
    ]

    return data

# Step 10 - format_example
def format_example(example):
    # TODO: render {'prompt','response'} into one training string with role markers
    return (
        f"### Instruction:\n"
        f"{example['prompt']}\n\n"
        f"### Response:\n"
        f"{example['response']}"
    )

# Step 11 - apply_template
def apply_template(examples):
    # TODO: apply format_example to each item in examples and return the list of strings.
    
    return [format_example(example) for example in examples]

# Step 12 - tokenize_example (not yet solved)
# TODO: implement

# Step 13 - build_labels (not yet solved)
# TODO: implement

# Step 14 - mask_prompt_labels (not yet solved)
# TODO: implement

# Step 15 - pad_batch (not yet solved)
# TODO: implement

# Step 16 - make_attention_mask (not yet solved)
# TODO: implement

# Step 17 - collate_lm_batch (not yet solved)
# TODO: implement

# Step 18 - iterate_minibatches (not yet solved)
# TODO: implement

# Step 19 - train_val_split (not yet solved)
# TODO: implement

# Step 20 - shift_logits_and_labels (not yet solved)
# TODO: implement

# Step 21 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 22 - adamw_update (not yet solved)
# TODO: implement

# Step 23 - linear_warmup_schedule (not yet solved)
# TODO: implement

# Step 24 - clip_grad_norm (not yet solved)
# TODO: implement

# Step 25 - accumulate_gradients (not yet solved)
# TODO: implement

# Step 26 - sft_train_step (not yet solved)
# TODO: implement

# Step 27 - evaluate_loss (not yet solved)
# TODO: implement

# Step 28 - lora_delta (not yet solved)
# TODO: implement

# Step 29 - lora_linear_forward (not yet solved)
# TODO: implement

# Step 30 - init_lora_weights (not yet solved)
# TODO: implement

# Step 31 - freeze_base_params (not yet solved)
# TODO: implement

# Step 32 - count_trainable_params (not yet solved)
# TODO: implement

# Step 33 - merge_lora (not yet solved)
# TODO: implement

# Step 34 - build_synthetic_preference_dataset (not yet solved)
# TODO: implement

# Step 35 - format_preference (not yet solved)
# TODO: implement

# Step 36 - reward_head_forward (not yet solved)
# TODO: implement

# Step 37 - pairwise_reward_loss (not yet solved)
# TODO: implement

# Step 38 - reward_bce_loss (not yet solved)
# TODO: implement

# Step 39 - pairwise_accuracy (not yet solved)
# TODO: implement

# Step 40 - reward_train_step (not yet solved)
# TODO: implement

# Step 41 - sequence_logprob (not yet solved)
# TODO: implement

# Step 42 - per_token_kl (not yet solved)
# TODO: implement

# Step 43 - compute_returns (not yet solved)
# TODO: implement

# Step 44 - gae_advantages (not yet solved)
# TODO: implement

# Step 45 - policy_ratio (not yet solved)
# TODO: implement

# Step 46 - clipped_surrogate (not yet solved)
# TODO: implement

# Step 47 - value_function_loss (not yet solved)
# TODO: implement

# Step 48 - entropy_bonus (not yet solved)
# TODO: implement

# Step 49 - ppo_loss (not yet solved)
# TODO: implement

# Step 50 - kl_penalized_reward (not yet solved)
# TODO: implement

# Step 51 - batch_sequence_logprob (not yet solved)
# TODO: implement

# Step 52 - dpo_logratios (not yet solved)
# TODO: implement

# Step 53 - dpo_ref_logratios (not yet solved)
# TODO: implement

# Step 54 - dpo_loss (not yet solved)
# TODO: implement

# Step 55 - ipo_loss (not yet solved)
# TODO: implement

# Step 56 - kto_loss (not yet solved)
# TODO: implement

# Step 57 - orpo_loss (not yet solved)
# TODO: implement

# Step 58 - simpo_loss (not yet solved)
# TODO: implement

# Step 59 - build_eval_prompt_set (not yet solved)
# TODO: implement

# Step 60 - generate_completions (not yet solved)
# TODO: implement

# Step 61 - score_with_reward (not yet solved)
# TODO: implement

# Step 62 - win_rate (not yet solved)
# TODO: implement

# Step 63 - stream_tokens (not yet solved)
# TODO: implement

# Step 64 - apply_stop_tokens (not yet solved)
# TODO: implement

# Step 65 - chat (not yet solved)
# TODO: implement

