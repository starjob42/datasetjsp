from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel
from utils.helping_functions import save_results
from utils.solution_generation import  read_matrix_form_jssp, generate_multiple_solutions

seed = 42
set_seed(seed)

dev_map = f"cuda:0"
# dev_map = f"auto"

checkpoint_path = "microsoft/Phi-3-mini-128k-instruct"
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map = dev_map
)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)


tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
tokenizer.model_max_length = 40000
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'left'


new_adapter_path = "./peft-phi3-jssp_orig_like_chat_dense_256_large_data_true_b4_/checkpoint-1750"

finetuned_model = PeftModel.from_pretrained(model,
                                  new_adapter_path,
                                  torch_dtype=torch.float16,
                                  is_trainable=False,
                                  device_map = dev_map,
                                  )
finetuned_model = finetuned_model.merge_and_unload()

custom_dataset_name = './test_2000.json'

# Load dataset
dataset = load_dataset("json", data_files=custom_dataset_name)

print(dataset)


def main_loop(start, end, model, tokenizer, dataset, num_solutions, temperature, top_k, top_p, max_len):
    """
    Main loop to generate and validate solutions for a range of JSSP problems in the dataset.

    Args:
        start (int): Starting index in the dataset.
        end (int): Ending index in the dataset.
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
        dataset: The dataset containing JSSP problems.
        num_solutions (int): Number of solutions to generate per problem.
        temperature (float): Sampling temperature.
        top_k (int): Top-k sampling parameter.
        top_p (float): Top-p (nucleus) sampling parameter.
        max_len (int): The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens. 
    Returns:
        None
    """
    results = []
    index = start
    while index < end:
        try:
            for index in tqdm(range(index, end)):
                jssp_problem = dataset['train'][index]['prompt_machines_first']
                baseline_sol = dataset['train'][index]['output']
                problem_in_matrix_form = dataset['train'][index]['matrix']

                n, m, inst_for_ortools, real_makespan, sol, machine_to_tasks = read_matrix_form_jssp(matrix_content=problem_in_matrix_form)
                if n ==3 and m ==3:

                    best_gap, is_feasible_list, gap_list, llm_makespan_list,calculated_makespan_list, time_list, peft_model_text_output = generate_multiple_solutions(
                        model, tokenizer, jssp_problem, inst_for_ortools, real_makespan, dev_map=dev_map,sample=True, num_solutions=num_solutions,top_k=top_k, top_p=top_p,temperature=temperature,max_len=max_len)

                else:
                    continue

                results.append((n, m, best_gap, is_feasible_list, gap_list, time_list, llm_makespan_list , calculated_makespan_list, peft_model_text_output))
            
                save_results(results, start, num_solutions, temperature, top_p, top_k)
        
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory at index {index}, restarting from the next index.")
            torch.cuda.empty_cache()
            start = index + 1
            continue
        except Exception as e:
            print(f"An error occurred at index {index}: {e}")
            break

    save_results(results, start, num_solutions, temperature, top_p, top_k)



start = 0
end = len(dataset['train'])
num_solutions = 20
top_k =  50
temperature =  0.2
top_p =  0.8
max_len = 400000

main_loop(start, end, model, tokenizer, dataset, num_solutions, temperature, top_k, top_p, max_len)


# Parameters and Their Effects
# Temperature:

# Definition: Controls the randomness of predictions by scaling the logits before applying softmax. A higher temperature (e.g., 1.0) makes the output more random, while a lower temperature (e.g., 0.1) makes it more deterministic.
# Effect: Lower temperatures make the model more conservative and focused, reducing the diversity but increasing the quality of predictions for more constrained problems. Higher temperatures increase the diversity but can lead to less coherent solutions.
# Top-k Sampling:

# Definition: Limits the sampling pool to the top k predictions. For example, if k=50, only the top 50 predictions are considered at each step.
# Effect: Smaller k values (e.g., 10) make the output more focused and less diverse, while larger k values (e.g., 100) increase diversity. Too large a k can introduce noise and irrelevant options.
# Top-p (Nucleus) Sampling:

# Definition: Considers the smallest set of predictions whose cumulative probability exceeds p. For example, if p=0.9, it includes the smallest number of predictions that together have a 90% probability.
# Effect: Lower p values (e.g., 0.8) result in more focused and high-probability outputs, while higher p values (e.g., 0.95) include more diverse and less probable options.
