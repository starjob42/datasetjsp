import csv 

def print_number_of_trainable_model_parameters(model):
    """
    Prints the number of trainable parameters in the model.

    Args:
        model: The model to evaluate.

    Returns:
        str: A string summarizing the number of trainable and total parameters.
    """

    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

def write_csv(results, filename='llm_jssp_results.csv'):
    """
    Writes the results to a CSV file.

    Args:
        results (list): The list of results to write.
        filename (str, optional): The name of the output file. Defaults to 'llm_jssp_results.csv'.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['num_jobs','num_machines', 'best_gap', 'is_feasible_list', 'gap_list', 'time_list', 'llm_makespan_list' , 'calculated_makespan_list', 'peft_model_text_output'])
            for result in results:
                writer.writerow(result)
        print(f"Results successfully saved to {filename}")
    except Exception as e:
        print(f"Failed to write CSV file {filename}. Error: {e}")


def save_results(results, start, num_solutions, temperature, top_p, top_k):
    filename = f'./validation_{start}_results_num_sol_{num_solutions}_t_{temperature}_p_{top_p}_k_{top_k}.csv'
    write_csv(results, filename=filename)
