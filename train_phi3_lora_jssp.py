from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
import transformers 
import torch
from functools import partial
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils.data_preprocessing import preprocess_dataset
from utils.helping_functions import print_number_of_trainable_model_parameters

seed = 42
set_seed(seed)
#tensorboard --logdir=peft-phi2-jssp-training-consise/logs

custom_dataset_name = './jssp_llm_format_120k.json'

dataset = load_dataset("json", data_files=custom_dataset_name)
print(dataset)

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

#use device_map = {"": 0} if you want to train on specific GPU
device_map="cuda",

model_name='microsoft/Phi-3-mini-128k-instruct'

original_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                      device_map='auto',
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True,
                                                      )

tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)#,padding_side="left",add_eos_token=True,add_bos_token=True,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 40000
# tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'


eval_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)# add_bos_token=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.unk_token
# tokenizer.pad_token = tokenizer.eos_token

def gen(model,p, maxlen=1000, sample=False):
    """
    Generates text using the model based on the provided prompt.

    Args:
        model: The pre-trained language model.
        p (str): The prompt text.
        maxlen (int, optional): Maximum length of the generated text. Defaults to 1000.
        sample (bool, optional): Whether to use sampling. Defaults to False.

    Returns:
        list: A list of generated text sequences.
    """
    toks = eval_tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample,num_return_sequences=1).to('cpu')
    return eval_tokenizer.batch_decode(res,skip_special_tokens=True)


index = 0

VAL_SET_SIZE = 2000

train_val = dataset["train"].train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
dataset = train_val

prompt = dataset['test'][index]['prompt_machines_first']
summary = dataset['test'][index]['output']

formatted_prompt = f"Instruct: Provide a schedule for the following JSSP problem .\n{prompt}\nOutput:\n"
res = gen(original_model,formatted_prompt,2000,)
#print(res[0])
output = res[0].split('Output:\n')[1]

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{formatted_prompt}')
print(dash_line)
print(f'BASELINE SOLUTION:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')




max_length = 40000
print('Max length to be used is ; ', max_length)

train_dataset = preprocess_dataset(tokenizer, max_length,seed, dataset['train'])
eval_dataset = preprocess_dataset(tokenizer, max_length,seed, dataset['test'])

print(f"Shapes of the datasets:")
print(f"Training: {train_dataset.shape}")
print(f"Validation: {eval_dataset.shape}")
print(train_dataset)

# #### 8. Setup the PEFT/LoRA model for Fine-Tuning
# Now, let's perform Parameter Efficient Fine-Tuning (PEFT) fine-tuning. PEFT is a form of instruction fine-tuning that is much more efficient than full fine-tuning.
# PEFT is a generic term that includes Low-Rank Adaptation (LoRA) and prompt tuning (which is NOT THE SAME as prompt engineering!). In most cases, when someone says PEFT, they typically mean LoRA. 
# LoRA, in essence, enables efficient model fine-tuning using fewer computational resources, often achievable with just a single GPU. Following LoRA fine-tuning for a specific task or use case, 
# the outcome is an unchanged original LLM and the emergence of a considerably smaller "LoRA adapter," often representing a single-digit percentage of the original LLM size (in MBs rather than GBs).
# 
# During inference, the LoRA adapter must be combined with its original LLM. The advantage lies in the ability of many LoRA adapters to reuse the original LLM, thereby reducing overall memory 
# requirements when handling multiple tasks and use cases.
# 
# Note the rank (r) hyper-parameter, which defines the rank/dimension of the adapter to be trained.
# r is the rank of the low-rank matrix used in the adapters, which thus controls the number of parameters trained.
# A higher rank will allow for more expressivity, but there is a compute tradeoff.
# 
# alpha is the scaling factor for the learned weights. The weight matrix is scaled by alpha/r, and thus a higher value for alpha assigns more weight to the LoRA activations.


print(print_number_of_trainable_model_parameters(original_model))



config = LoraConfig(
    r=128, #Rank
    lora_alpha=256,
    target_modules=[
        'qkv_proj',
        'o_proj',
        'fc1',
        'fc2',
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)

# 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
original_model.gradient_checkpointing_enable()

# 2 - Using the prepare_model_for_kbit_training method from PEFT
original_model = prepare_model_for_kbit_training(original_model)

peft_model = get_peft_model(original_model, config)


# Once everything is set up and the base model is prepared, we can use the print_trainable_parameters() 
# helper function to see how many trainable parameters are in the model.

print(print_number_of_trainable_model_parameters(peft_model))


output_dir = './testing_final_code_peft-phi3-jssp/'


MICRO_BATCH_SIZE = 2 
BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LOG_STEP = 50
peft_training_args = TrainingArguments(
    output_dir = output_dir,
    warmup_steps=1,
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    per_device_eval_batch_size=MICRO_BATCH_SIZE,   # batch size for evaluation*
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    # max_steps=3,
    num_train_epochs=2,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    logging_steps=LOG_STEP,
    logging_dir=output_dir+"logs",
    save_strategy="steps",
    save_steps=LOG_STEP,
    evaluation_strategy="steps",
    eval_steps=LOG_STEP,
    do_eval=True,
    gradient_checkpointing=True,
    # report_to="tensorboard",
    overwrite_output_dir = 'True',
    group_by_length=True,
    save_total_limit=50,
    # output_dir="phi2_lora-jssp_machines_first_lora_r_{32}",
    fp16=True,
)

peft_model.config.use_cache = False

peft_trainer = transformers.Trainer(
    model=peft_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=peft_training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

peft_training_args.device

peft_trainer.train()

# peft_trainer.train(resume_from_checkpoint=True)

# Free memory for merging weights
del original_model
del peft_trainer
torch.cuda.empty_cache()