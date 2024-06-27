from functools import partial
import random
from transformers import AutoTokenizer

def create_prompt_formats(example,tokenizer):

    """
    Creates prompt formats for training the model.

    Args:
        example (dict): A single example from the dataset.

    Returns:
        dict: The example with added formatted text.
    """

    # Standard user prompts with different variations
    user_variations = [
        "Instruct: Provide a solution schedule for the JSSP problem below, also indicate the makespan.",
        "Task: Provide the steps of a solution for the JSSP problem and determine the makespan.",
        "Command: Give a detailed solution to tackle the JSSP problem, focusing on optimizing the makespan."
    ]
    
    # Extracting the prompt and output directly from the example dictionary
    prompt = example["prompt_machines_first"]
    output = example["output"]
    
    # Randomly selecting standard messages
    user_standard = random.choice(user_variations)

    # Creating a messages array with user and assistant roles
    messages = [
        {"role": "system", "content": "You are expert in Job Shop Scheduling Problem"},
        {"role": "user", "content": " " + user_standard + " \n" + prompt + " "},
        {"role": "assistant", "content": " " + output + " "}
    ]

    # Apply the chat template from the tokenizer with settings
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    
    return example


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes a batch of examples.

    Args:
        batch (dict): A batch of examples.
        tokenizer (AutoTokenizer): The tokenizer to use.
        max_length (int): Maximum length of the tokenized sequences.

    Returns:
        dict: The tokenized batch.
    """
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int,seed, dataset):
    """"
    Formats and tokenizes the dataset for training.

    Args:
        tokenizer (AutoTokenizer): The tokenizer to use.
        max_length (int): Maximum number of tokens to emit from tokenizer.
        seed (int): Seed for shuffling the dataset.
        dataset: The dataset to preprocess.

    Returns:
        Dataset: The preprocessed dataset.
    """
    
    # Add prompt to each sample
    print("Preprocessing dataset...")
    _create_prompt_formats = partial(create_prompt_formats, tokenizer=tokenizer)
    dataset = dataset.map(_create_prompt_formats)
    
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=[ 'prompt_machines_first', 'output'],
        num_proc=1
    )

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset



