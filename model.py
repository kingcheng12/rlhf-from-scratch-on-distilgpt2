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

# Step 12 - tokenize_example
def tokenize_example(tokenizer, text, max_length=64):
    # TODO: encode `text` with truncation at max_length, no padding, return list[int]

    inputs = tokenizer(text,
                        truncation=True,
                        max_length=max_length,
                        padding=False,
                        )

    return inputs["input_ids"]

# Step 13 - build_labels
def build_labels(input_ids):
    # TODO: return a fresh list equal to input_ids to serve as next-token labels

    return input_ids.copy()

# Step 14 - mask_prompt_labels
def mask_prompt_labels(labels, prompt_length):
    # TODO: replace the first prompt_length entries of labels with -100 and return the new list
    if prompt_length > len(labels):
        return [-100] * len(labels)

    masked = [-100] * prompt_length

    out = masked + labels[prompt_length:].copy()
    
    return out

# Step 15 - pad_batch
def pad_batch(sequences, pad_id):
    # TODO: right-pad a list of token id sequences to the longest length using pad_id

    max_len = max([len(s) for s in sequences])

    out = []
    for s in sequences:
        pad_s = s.copy()
        if len(s) < max_len:
            pad_s += [pad_id]*(max_len - len(s))
        out.append(pad_s)
    
    return out

# Step 16 - make_attention_mask
def make_attention_mask(padded_ids, pad_id):
    # TODO: return a same-shape 0/1 mask with 1 where token != pad_id else 0

    padded_ids = torch.tensor(padded_ids)

    return (padded_ids != pad_id).to(torch.long).tolist()

# Step 17 - collate_lm_batch
def collate_lm_batch(batch, pad_id):
    # TODO: pad input_ids and labels, build attention mask, return dict of LongTensors
    
    input_id_sequences = [example["input_ids"] for example in batch]
    label_sequences = [example["labels"] for example in batch]

    padded_input_ids = pad_batch(input_id_sequences, pad_id)
    padded_labels = pad_batch(label_sequences, -100)

    attention_mask = make_attention_mask(
        padded_input_ids,
        pad_id,
    )

    return {
        "input_ids": torch.tensor(
            padded_input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            padded_labels,
            dtype=torch.long,
        ),
    }

# Step 18 - iterate_minibatches
import random

def iterate_minibatches(examples, batch_size, seed=0):
    # TODO: yield shuffled minibatches of size batch_size from examples (deterministic per seed).

    indices = list(range(len(examples)))

    rng = random.Random(seed)
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        yield [examples[i] for i in batch_indices]

# Step 19 - train_val_split
import random

def train_val_split(examples, val_ratio=0.2, seed=0):
    # TODO: deterministically split examples into (train, val) using seed and val_ratio
    indices = list(range(len(examples)))

    rng = random.Random(seed)
    rng.shuffle(indices)

    num_val = int(len(examples) * val_ratio)

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]  

    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]

    return train_examples, val_examples

# Step 20 - shift_logits_and_labels
def shift_logits_and_labels(logits, labels):
    # TODO: drop the last logit position and the first label position so token t predicts t+1

    shift_logits = logits[:, :-1, :].clone()
    shift_labels = labels[:, 1:].clone()

    return shift_logits, shift_labels

# Step 21 - cross_entropy_loss
import torch
import torch.nn.functional as F

def cross_entropy_loss(shift_logits, shift_labels):
    """Mean next-token cross-entropy, ignoring label positions equal to -100."""
    # TODO: reduce (B, T-1, V) logits and (B, T-1) labels to a scalar loss tensor.
    
    vocab_size = shift_logits.shape[-1]

    flat_logits = shift_logits.reshape(-1, vocab_size)
    flat_labels = shift_labels.reshape(-1)

    return F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=-100,
        reduction="mean",
    )

# Step 22 - adamw_update
import torch

def adamw_update(param, grad, state, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
    """Apply one in-place AdamW step to `param` using `grad` and persistent `state`."""
    # TODO: initialize state on first call, then update moments and apply the decoupled AdamW step
    beta1, beta2 = betas

    if not state:
        state["step"] = 0
        state["m"] = torch.zeros_like(param)
        state["v"] = torch.zeros_like(param)

    state["step"] += 1
    step = state["step"]

    m = state["m"]
    v = state["v"]

    with torch.no_grad():
        # First moment.
        m.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Second moment.
        v.mul_(beta2).addcmul_(
            grad,
            grad,
            value=1 - beta2,
        )

        # Bias correction.
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)

        # Decoupled weight decay.
        param.mul_(1 - lr * weight_decay)

        # Adam update.
        denominator = torch.sqrt(v_hat) + eps
        param.addcdiv_(
            m_hat,
            denominator,
            value=-lr,
        )

    return state

# Step 23 - linear_warmup_schedule
def linear_warmup_schedule(step, warmup_steps):
    # TODO: return a linear warmup multiplier in [0, 1] given the current step and warmup window.
    if warmup_steps == 0:
        return 1
    return min(1, step/warmup_steps)

# Step 24 - clip_grad_norm
import math

def clip_grad_norm(grads, max_norm):
    # TODO: compute the global L2 norm of grads and rescale in place if it exceeds max_norm.
    
    l2_grads = [torch.sum(grad ** 2).item() for grad in grads]
    l2_total = math.sqrt(sum(l2_grads))

    if l2_total > max_norm:
        scale = max_norm / l2_total
        for grad in grads:
            grad.mul_(scale)

    return float(l2_total)

# Step 25 - accumulate_gradients
import torch

def accumulate_gradients(grad_list):
    """Average a list of equally-shaped gradient tensors across micro-batches."""
    # TODO: average a list of equally-shaped gradient tensors and return the mean tensor
    
    return torch.stack(grad_list, dim=0).mean(dim=0)

# Step 26 - sft_train_step
import torch

def sft_train_step(model, batch, optimizer):
    """Run one SFT forward/backward/step and return the loss as a float."""
    # TODO: forward the batch, compute shifted cross-entropy loss, backprop, step optimizer
    model.train()
    optimizer.zero_grad()
    
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = batch['attention_mask']

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    logits = outputs.logits

    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    loss = cross_entropy_loss(shift_logits, shift_labels)

    loss.backward()
    optimizer.step()

    return loss.item()

# Step 27 - evaluate_loss
import torch

def evaluate_loss(model, batches):
    """Mean LM loss over validation batches, no grad."""
    # TODO: iterate batches under no_grad, shift logits/labels, average cross-entropy.

    model.eval()
    losses = []

    with torch.no_grad():
        for batch in batches:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            labels = batch["labels"]

            shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
            loss = cross_entropy_loss(shift_logits, shift_labels)

            losses.append(loss.item())
    
    return sum(losses)/len(losses)

# Step 28 - lora_delta
def lora_delta(A, B, alpha, r):
    # TODO: build the scaled low-rank weight update from factors A and B.

    delta = (alpha / r) * B @ A

    return delta.to(dtype=A.dtype)

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

