import sys
import subprocess
import os
import numpy as np
import time
from sys import executable, exit
from packaging.version import Version

# Input directory to read test input from
input_dir = sys.argv[1]

# Output data directory to which to write predictions
output_dir = sys.argv[2]

submission_dir = sys.argv[3]

sys.path.append(output_dir)
sys.path.append(submission_dir)


def load_dataset(filename):
    """
    Load test dataset from file.

    Args:
        filename: Name of the test data file

    Returns:
        test_data: Samples to test with shape 
                  (num_samples, num_timestep, num_channels, num_bands)
    """
    test_file = os.path.join(input_dir, filename)
    # Open the file and load the data
    test_data = np.load(test_file)['arr_0']
    return test_data


def install_from_whitelist(req_file):
    """
    Install packages from requirements file if they are in the whitelist.

    Args:
        req_file: Path to requirements.txt file
    """
    whitelist = open("/app/program/whitelist.txt", 'r').readlines()
    whitelist = [i.rstrip('\n') for i in whitelist]

    for package in open(req_file, 'r').readlines():
        package = package.rstrip('\n')
        package_version = package.split("==")
        if len(package_version) > 2:
            # Invalid format, don't use
            print(
                f"requested package {package} has invalid format, "
                f"will install latest version (of {package_version[0]}) "
                f"if allowed")
            package = package_version[0]
        elif len(package_version) == 2:
            version_str = package_version[1]
            Version(version_str)

        if package_version[0] in whitelist:
            # Package must be in whitelist, so format check unnecessary
            subprocess.check_call(
                [executable, "-m", "pip", "install", package])
            print(f"{package_version[0]} installed")
        else:
            exit(
                f"{package_version[0]} is not an allowed package. "
                f"Please contact the organizers on GitHub to request "
                f"acceptance.")


def print_pretty(text):
    """Print formatted section header."""
    print("-------------------")
    print("#---", text)
    print("-------------------")


def save_prediction(prediction_nparray, dataset_name):
    """
    Save prediction array to npz file.

    Args:
        prediction_nparray: Prediction array to save
        dataset_name: Name of the dataset for file naming
    """
    # Save the prediction to a npz file
    np.savez(os.path.join(
        output_dir, f'test_{dataset_name}.predictions'), prediction_nparray)


def main():
    """
    Run the complete prediction pipeline:
    1. Install required packages
    2. Load test datasets
    3. Load trained model
    4. Make predictions
    5. Save results
    """
    start = time.time()

    # Install packages if requirements.txt exists
    requirements_file = os.path.join(submission_dir, "requirements.txt")
    if os.path.isfile(requirements_file):
        install_from_whitelist(requirements_file)

    duration = time.time() - start
    print_pretty(f'Duration of the package installation: {duration}')

    start = time.time()

    print_pretty('Reading Data')
    # Import Model after paths are set up
    from model import Model

    # Load all test datasets
    affi_test = load_dataset('test_data_affi_masked.npz')
    beignet_test = load_dataset('test_data_beignet_masked.npz')
    affi_test_private = load_dataset(
        'test_data_affi_2024-03-20_private_masked.npz')
    beignet_test_private_1 = load_dataset(
        'test_data_beignet_2022-06-01_private_masked.npz')
    beignet_test_private_2 = load_dataset(
        'test_data_beignet_2022-06-02_private_masked.npz')

    print_pretty('Loading trained Model')
    m_affi = Model('affi')
    m_affi.load()

    print_pretty('Making Prediction for affi')
    # Generate predictions for affi datasets
    prediction_affi = m_affi.predict(affi_test)
    prediction_affi_private = m_affi.predict(affi_test_private)

    print_pretty('Saving Prediction for affi')
    # Save affi predictions
    save_prediction(prediction_affi, 'affi')
    save_prediction(prediction_affi_private, 'affi_private')

    m_beignet = Model('beignet')
    m_beignet.load()

    print_pretty('Making Prediction for beignet')
    prediction_beignet = m_beignet.predict(beignet_test)
    prediction_beignet_private_1 = m_beignet.predict(beignet_test_private_1)
    prediction_beignet_private_2 = m_beignet.predict(beignet_test_private_2)

    print_pretty('Saving Prediction for beignet')
    save_prediction(prediction_beignet, 'beignet')
    save_prediction(prediction_beignet_private_1, 'beignet_private_1')
    save_prediction(prediction_beignet_private_2, 'beignet_private_2')

    # Print total duration in seconds
    duration = time.time() - start
    print_pretty(f'Total duration: {duration:.2f} seconds')


if __name__ == '__main__':

    main()
