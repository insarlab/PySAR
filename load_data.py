import os
import logging

# Configure logging
logging.basicConfig(filename='load_data.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_file(file_path):
    """
    Load the contents of a file.

    :param file_path: Path to the file to be loaded.
    :return: Contents of the file.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    try:
        with open(file_path, 'r') as file:
            data = file.read()
            logging.info(f"File loaded successfully: {file_path}")
            return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None

def write_file(file_path, data):
    """
    Write data to a file.

    :param file_path: Path to the file where data will be written.
    :param data: Data to write to the file.
    """
    try:
        with open(file_path, 'w') as file:
            file.write(data)
            logging.info(f"File written successfully: {file_path}")
    except Exception as e:
        logging.error(f"Error writing to file {file_path}: {e}")

# Example usage
if __name__ == "__main__":
    file_path = 'example.txt'
    data = load_file(file_path)
    if data is not None:
        write_file('output.txt', data)