import os

def load_data(file_path):
    """
    Load data from a specified file.

    Parameters:
    file_path (str): The path to the file to be loaded.

    Returns:
    data: The data loaded from the file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r') as file:
        data = file.read()
    
    return data

def write_data(file_path, data):
    """
    Write data to a specified file.

    Parameters:
    file_path (str): The path to the file where data will be written.
    data: The data to be written to the file.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    with open(file_path, 'w') as file:
        file.write(data)