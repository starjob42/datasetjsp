from io import StringIO
import numpy as np
import re
import torch
import time

def read_matrix_form_jssp(matrix_content: str, sep: str = ' '):
    """
    Reads the JSSP problem from a string and returns problem data in various formats.

    Args:
        matrix_content (str): The JSSP problem in string format.
        sep (str, optional): The separator used in the string. Defaults to ' '.

    Returns:
        tuple: Contains the number of jobs, number of machines, instance for OR-Tools,
               makespan, solution matrix, and machine to tasks dictionary.
    """

    f = StringIO(matrix_content)
    
    # Load the shape
    n, m = map(int, next(f).split(sep))

    # Load the instance
    instance_lines = [next(f).strip() for _ in range(n)]
    instance = np.array([line.split(sep) for line in instance_lines if line], dtype=np.int16)
    inst_for_ortools = instance.reshape((n, m, 2))

    # Try to read makespan, handle the case if it's missing or not a digit
    try:
        ms = float(next(f).strip())
    except (StopIteration, ValueError):
        ms = None  # Set to None or another appropriate value indicating absence

    # Attempt to read the solution, handle the case if it's not present
    solution_lines = []
    try:
        while True:
            line = next(f).strip()
            if line:
                solution_lines.append(line)
            else:
                break
    except StopIteration:
        pass  # No more lines to read, proceed with possibly an empty solution_lines list

    sol = np.array([line.split(sep) for line in solution_lines if line], dtype=np.int32) if solution_lines else None

    initial_operation_matrix = np.arange(n*m).reshape((n, m))

    # Create machine_to_tasks dictionary
    machine_to_tasks = {}
    if sol is not None:
        for machine_index, machine_row in enumerate(sol):
            tasks_list = []
            for job_value in machine_row:
                job, task = np.where(initial_operation_matrix == job_value)
                if job.size > 0 and task.size > 0:
                    tasks_list.append((job[0], task[0]))  # Assuming job and task indices are found correctly
            machine_to_tasks[machine_index] = tasks_list

    return n, m, inst_for_ortools.tolist(), ms, sol, machine_to_tasks

def parse_solution(text):
    """
    Parses the solution text and extracts operation details and makespan.

    Args:
        text (str): The solution text to parse.

    Returns:
        tuple: Contains a list of operations and the makespan.
    """

    pattern = r"Job (\d+) Operation (\d+) on Machine (\d+) : (\d+) \+ (\d+) -> (\d+)"
    operations = re.findall(pattern, text)

    makespan_pattern = r"Makespan:\s+(\d+(\.\d+)?)"
    makespan_match = re.search(makespan_pattern, text)
    makespan = float(makespan_match.group(1)) if makespan_match else None
    # print('LLM makespan : ', makespan)

    return [{
        'Job': int(job),
        'Operation': int(operation),
        'Machine': int(machine),
        'Start Time': int(start_time),
        'Duration': int(duration),
        'End Time': int(end_time)
    } for job, operation, machine, start_time, duration, end_time in operations],makespan

def validate_jssp_solution(operations, problem_data):
    """
    Validates the JSSP solution against the problem data.

    Args:
        operations (list): The list of operations in the solution.
        problem_data (list): The JSSP problem data.

    Returns:
        tuple: Contains a boolean indicating feasibility, a message, and the calculated makespan.
    """

    for op in operations:
        job = op['Job']
        operation = op['Operation']
        machine = op['Machine']
        expected_machine, expected_duration = problem_data[job][operation]
        # print(job, operation)
        # print('expected_machine ', expected_machine)
        # print('expected_duration ', expected_duration)

        if expected_machine != machine or expected_duration != op['Duration']:
            return False, f"Mismatch in data for Job {job} Operation {operation}: Expected Machine {expected_machine} and Duration {expected_duration}, got Machine {machine} and Duration {op['Duration']}"

    # Organize operations by machines to check for conflicts
    machines = {}
    for op in operations:
        if op['Machine'] not in machines:
            machines[op['Machine']] = []
        machines[op['Machine']].append(op)
    
    # Check for overlapping times on each machine
    for machine_ops in machines.values():
        sorted_ops = sorted(machine_ops, key=lambda x: x['Start Time'])
        for i in range(len(sorted_ops) - 1):
            if sorted_ops[i]['End Time'] > sorted_ops[i + 1]['Start Time']:
                return False, f"Overlap on machine {sorted_ops[i]['Machine']} between operations {sorted_ops[i]['Operation']} and {sorted_ops[i + 1]['Operation']}"

    # Check operation precedence within each job
    jobs = {}
    for op in operations:
        if op['Job'] not in jobs:
            jobs[op['Job']] = []
        jobs[op['Job']].append(op)

    for job_ops in jobs.values():
        sorted_ops = sorted(job_ops, key=lambda x: x['Operation'])
        for i in range(len(sorted_ops) - 1):
            if sorted_ops[i]['End Time'] > sorted_ops[i + 1]['Start Time']:
                return False, f"Operation order error in job {sorted_ops[i]['Job']} at Operation {sorted_ops[i]['Operation']}"

    # Calculate makespan
    makespan = max(op['End Time'] for op in operations)
    # print('Calculated makespan : ', makespan)
    return True, "Solution satisfies all constraints", makespan


def apply_chat_template_inference(prompt, tokenizer, index=0):

    """
    Applies a chat template to the prompt for the model.

    Args:
        prompt (str): The JSSP problem prompt.
        tokenizer: The tokenizer to use.
        index (int, optional): The index of the user variation to use. Defaults to 0.

    Returns:
        str: The formatted text for the model.
    """

    # Standard user prompts with different variations
    user_variations = [
        "Instruct: Provide a solution schedule for the JSSP problem below, also indicate the makespan.",
        "Task: Provide the steps of a solution for the JSSP problem and determine the makespan.",
        "Command: Give a detailed solution to tackle the JSSP problem, focusing on optimizing the makespan."
    ]
    
    user_standard = user_variations[index]
    user_question = {"role": "user", "content": f"{user_standard}\n{prompt}"}

    # Creating a messages array with user and assistant roles
    messages = [
        {"role": "system", "content": "You are an expert in Job Shop Scheduling Problem"},
        user_question
    ]

    # Apply the chat template from the tokenizer with settings
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return formatted_text


def gen(model, prompt, tokenizer, dev_map, maxlen=1000, sample=True, num_return_sequences=5, temperature=1.0, top_k=50, top_p=0.95):
    """
    Generates text using the model based on the provided prompt.

    Args:
        model: The pre-trained language model.
        prompt (str): The prompt text.
        tokenizer: The tokenizer corresponding to the pre-trained model.
        dev_map (str): The device to run the model on. Defaults to "cuda".
        maxlen (int, optional): Maximum length of the generated text. Defaults to 1000.
        sample (bool, optional): Whether to use sampling. Defaults to True.
        num_return_sequences (int, optional): Number of sequences to return. Defaults to 5.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        top_k (int, optional): Top-k sampling parameter. Defaults to 50.
        top_p (float, optional): Top-p (nucleus) sampling parameter. Defaults to 0.95.
        max_len (int): The maximum length the generated tokens can have. Corresponds to the length of the input prompt + max_new_tokens. 

    Returns:
        list: A list of generated text sequences.
    """
    toks = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        res = model.generate(
            **toks.to(dev_map),
            max_new_tokens=maxlen,
            do_sample=sample,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ).to('cpu')
    return tokenizer.batch_decode(res, skip_special_tokens=True)



def generate_multiple_solutions(model, tokenizer, jssp_problem, inst_for_ortools, real_makespan, dev_map="cuda",sample=True, num_solutions=5,top_k=50, top_p=90,  temperature=0.42,max_len=40000):
    """
    Generates multiple solutions for the JSSP problem and evaluates their quality.

    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer for the model.
        jssp_problem (str): The JSSP problem prompt.
        inst_for_ortools (list): Instance formatted for OR-Tools.
        real_makespan (float): The real makespan of the problem.
        num_solutions (int, optional): Number of solutions to generate. Defaults to 5.
        top_k (int, optional): Top-k sampling parameter. Defaults to 50.
        top_p (float, optional): Top-p (nucleus) sampling parameter. Defaults to 90.
        temperature (float, optional): Sampling temperature. Defaults to 0.42.

    Returns:
        tuple: Contains best gap, feasibility list, gap list, LLM makespan list,
               calculated makespan list, time list, and PEFT model text output.
    """
    
    current_prompt = jssp_problem
    prompt = apply_chat_template_inference(current_prompt, tokenizer)
    gap_list = []
    llm_makespan_list = []
    calculated_makespan_list = []

    time_list = []
    is_feasible_list = []
    min_gap_list = None
    
    start_time = time.time()

    peft_model_res = gen(model=model, prompt=prompt, tokenizer=tokenizer, dev_map=dev_map, maxlen=max_len, sample=sample, num_return_sequences=num_solutions, temperature=temperature, top_k=top_k, top_p=top_p)
    
    for peft_model_text_output in peft_model_res:
        try:
            operations, model_makespan = parse_solution(peft_model_text_output)
            is_feasible, message, calculated_makespan = validate_jssp_solution(operations, inst_for_ortools)
            elapsed_time = time.time() - start_time
            time_list.append(elapsed_time)    

            is_feasible_list.append(is_feasible)
            if not is_feasible:
                continue

            gap = (calculated_makespan - real_makespan) / real_makespan
            gap_formatted = f"{gap:.2f}"
            gap_list.append(gap_formatted)
            llm_makespan_list.append(model_makespan)
            calculated_makespan_list.append(calculated_makespan)
        except Exception as e:
            continue
    
        if len(gap_list) == 0:
            min_gap_list = None
            continue
        else:
            min_gap_list = min(gap_list)

    return min_gap_list, is_feasible_list, gap_list, llm_makespan_list,calculated_makespan_list, time_list,peft_model_text_output
