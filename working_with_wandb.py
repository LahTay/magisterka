import wandb
import pandas as pd
import os
import shutil
import json
import tqdm
from tabulate import tabulate

def initialize_api():
    """
    Initializes the Weights & Biases API.

    Returns:
        wandb.Api: An instance of the wandb API.
    """
    return wandb.Api()

def get_runs(api, project_name, group_name):
    """
    Fetches all runs from a specific group within a project.

    Args:
        api (wandb.Api): The wandb API instance.
        project_name (str): The name of the wandb project.
        group_name (str): The name of the group within the project.

    Returns:
        list: A list of wandb.Run objects belonging to the specified group.
    """
    runs = api.runs(f"{project_name}", filters={"group": group_name})
    return runs

def serialize_filters(filters):
    """
    Serializes the filters value into a string suitable for folder names.

    Args:
        filters (Any): The filters value from run.config, which can be a number, tuple, or other types.

    Returns:
        str: A serialized string representation of filters.
    """
    if isinstance(filters, (int, float)):
        return str(filters)
    elif isinstance(filters, tuple):
        # Convert the tuple to a string with elements separated by underscores
        return "_".join(map(str, filters))
    else:
        # For other types, convert to string directly
        return str(filters)

def sanitize_filename(filename):
    """
    Sanitizes a string to be used as a folder name by removing or replacing invalid characters.

    Args:
        filename (str): The original filename.

    Returns:
        str: A sanitized filename.
    """
    # Define a set of valid characters (alphanumeric and some symbols)
    valid_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    # Replace invalid characters with underscores
    sanitized = ''.join(c if c in valid_chars else '_' for c in filename)
    return sanitized

def prettify_json_files(directory):
    """
    Finds all JSON files in the given directory and reformats them to be more readable.

    Args:
        directory (str): The path to the directory containing JSON files.
    """
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                json_path = os.path.join(root, filename)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    with open(json_path, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"Prettified JSON file: {json_path}")
                except Exception as e:
                    print(f"Error processing JSON file {json_path}: {e}")

def download_and_save_artifacts(runs, table_key):
    """
    Downloads artifacts that have the classification_report_table and saves them
    into folders named with 'model_type' and 'filters', within the artifacts folder.
    If multiple runs have the same 'model_type' and 'filters', their artifacts are
    collected into the same folder. Artifacts from different runs are saved into
    subfolders named with the run's ID.

    Args:
        runs (list): A list of wandb.Run objects.
        table_key (str): The key to identify the artifacts to download.
    """
    artifacts_base_dir = 'artifacts'
    os.makedirs(artifacts_base_dir, exist_ok=True)

    for run in tqdm.tqdm(runs):
        # Extract 'model_type' and 'filters' from run.config
        model_type = run.config.get('model', 'unknown_model')
        filters = run.config.get('filters', 'unknown_filters')

        # Serialize filters to a string suitable for folder names
        filters_str = serialize_filters(filters)

        # Sanitize the folder name components
        model_type_sanitized = sanitize_filename(str(model_type))
        filters_str_sanitized = sanitize_filename(filters_str)

        # Destination folder name with 'model_type' and 'filters'
        dest_folder_name = f"{model_type_sanitized}_{filters_str_sanitized}"
        dest_folder_path = os.path.join(artifacts_base_dir, dest_folder_name)

        # Ensure the destination folder exists
        os.makedirs(dest_folder_path, exist_ok=True)

        # Fetch the table artifacts
        artifacts = run.logged_artifacts()
        table_artifacts = [artifact for artifact in artifacts if table_key in artifact.name]

        if not table_artifacts:
            print(f"No artifacts named '{table_key}' found for run {run.name}")
            continue  # Skip this run if no matching artifacts are found

        for artifact in table_artifacts:
            # Download the artifact to a temporary directory
            try:
                temp_dir = artifact.download()
                # Prettify JSON files in the temporary directory
                prettify_json_files(temp_dir)

                # Create a subfolder for this run within the destination folder
                run_folder_name = f"run_{sanitize_filename(run.id)}"
                run_folder_path = os.path.join(dest_folder_path, run_folder_name)
                os.makedirs(run_folder_path, exist_ok=True)

                # Move the contents of temp_dir into run_folder_path
                for item in os.listdir(temp_dir):
                    s = os.path.join(temp_dir, item)
                    if item.endswith('.table.json'):
                        # Remove '.table' from the filename
                        new_item_name = item.replace('.table.json', '.json')
                    else:
                        new_item_name = item
                    d = os.path.join(run_folder_path, new_item_name)
                    if os.path.isdir(s):
                        shutil.move(s, d)
                    else:
                        shutil.move(s, d)

                # Remove the temporary directory
                shutil.rmtree(temp_dir)

                print(f"Artifact for run {run.name} saved to '{run_folder_path}'")

            except Exception as e:
                print(f"Error processing artifact for run {run.name}: {e}")
                continue


def process_artifacts_directory(artifacts_dir='artifacts'):
    """
    Processes all model_type_filters folders inside the artifacts directory.
    """
    for model_folder in os.listdir(artifacts_dir):
        model_folder_path = os.path.join(artifacts_dir, model_folder)
        if os.path.isdir(model_folder_path):
            print(f"Processing folder: {model_folder_path}")
            process_model_folder(model_folder_path)


def process_model_folder(model_folder_path):
    """
    Processes all classification tables inside a model_type_filters folder,
    computes the average, and saves the averaged table as averaged_table.json.
    """
    # Initialize data structures
    data_per_label = {}  # {label: {metric: [values]}}
    labels = set()
    metrics = []
    columns = []
    n_tables = 0

    # For each run subfolder
    for run_subfolder in os.listdir(model_folder_path):
        run_folder_path = os.path.join(model_folder_path, run_subfolder)
        if os.path.isdir(run_folder_path):
            # Find the classification table JSON file
            for file_name in os.listdir(run_folder_path):
                if file_name.endswith('.json') and 'classification_report_table' in file_name:
                    table_json_path = os.path.join(run_folder_path, file_name)
                    # Read and parse the JSON file
                    try:
                        with open(table_json_path, 'r') as f:
                            table_json = json.load(f)
                    except Exception as e:
                        print(f"Error reading {table_json_path}: {e}")
                        continue

                    # Extract columns and data
                    columns = table_json.get('columns', [])
                    data = table_json.get('data', [])
                    # Initialize metrics list (excluding 'Label')
                    if not metrics and columns:
                        metrics = columns[1:]  # Exclude 'Label'

                    # Process the data
                    for row in data:
                        label = row[0]
                        values = row[1:]
                        labels.add(label)
                        if label not in data_per_label:
                            data_per_label[label] = {metric: [] for metric in metrics}
                        for metric, value in zip(metrics, values):
                            data_per_label[label][metric].append(value)
                    n_tables += 1
                    break  # Assume only one table per run folder

    if n_tables == 0:
        print(f"No classification tables found in {model_folder_path}")
        return

    # Compute averages
    averaged_data = []
    for label in sorted(labels):
        row = [label]
        for metric in metrics:
            values = data_per_label[label][metric]
            avg_value = sum(values) / len(values) if values else None
            row.append(avg_value)
        averaged_data.append(row)

    # Build the averaged classification table
    averaged_table = {
        "_type": "table",
        "columns": columns,
        "data": averaged_data,
        "ncols": len(columns),
        "nrows": len(averaged_data)
    }

    # Save the averaged table as averaged_table.json inside the model folder
    averaged_table_path = os.path.join(model_folder_path, 'averaged_table.json')
    try:
        with open(averaged_table_path, 'w') as f:
            json.dump(averaged_table, f, indent=4)
        print(f"Averaged table saved to {averaged_table_path}")
    except Exception as e:
        print(f"Error saving averaged table to {averaged_table_path}: {e}")


def print_metrics_for_artifacts(artifacts_dir='artifacts'):
    """
    Processes each model_type_filters folder and prints the metrics Informedness,
    Markedness, MCC, and CohenKappa for every instrument and average in a pretty table.
    """
    for model_folder in os.listdir(artifacts_dir):
        model_folder_path = os.path.join(artifacts_dir, model_folder)
        if os.path.isdir(model_folder_path):
            print(f"\nFolder: {model_folder}")
            averaged_table_path = os.path.join(model_folder_path, 'averaged_table.json')
            if os.path.exists(averaged_table_path):
                try:
                    with open(averaged_table_path, 'r') as f:
                        averaged_table = json.load(f)
                    print_metrics_table(averaged_table)
                except Exception as e:
                    print(f"Error reading {averaged_table_path}: {e}")
            else:
                print(f"Averaged table not found in {model_folder_path}")

def print_metrics_table(averaged_table):
    """
    Prints a nicely formatted table of selected metrics from the averaged table.

    Args:
        averaged_table (dict): The averaged classification table data.
    """
    # Extract columns and data
    columns = averaged_table.get('columns', [])
    data = averaged_table.get('data', [])

    # Metrics to display
    selected_metrics = ['Informedness', 'Markedness', 'MCC', 'CohenKappa']

    # Find indices of selected metrics
    metric_indices = [columns.index(metric) for metric in selected_metrics if metric in columns]

    # Prepare table data
    table_data = []
    for row in data:
        label = row[0]
        metrics_values = [row[index] for index in metric_indices]
        table_data.append([label] + metrics_values)

    # Prepare headers
    headers = ['Label'] + selected_metrics

    # Print the table using tabulate
    print(tabulate(table_data, headers=headers, tablefmt='grid', floatfmt=".4f"))


def main(mode):

    if mode == 'download_artifacts':
        # Replace with your project name and group name
        project_name = "magisterka-instrument-detection"
        group_name = "Final with tuned hiperparameters"
        table_key = 'classification_report_table'

        # Initialize the API and fetch runs
        api = initialize_api()
        runs = get_runs(api, project_name, group_name)

        # Download and save artifacts with renamed folders and prettified JSON files
        download_and_save_artifacts(runs, table_key)

    elif mode == 'process_artifacts':
        artifacts_dir = 'artifacts'
        process_artifacts_directory(artifacts_dir)
    elif mode == 'print_metrics':
        artifacts_dir = 'artifacts'
        print_metrics_for_artifacts(artifacts_dir)
    else:
        raise Exception("Wrong mode")


if __name__ == "__main__":
    # 'download_artifacts', or 'process_artifacts', or 'print_metrics'
    main('print_metrics')
