from model import Model
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
    filenames: load test dataset
    return:
    test_data: samples to test with shape (num_samples, num_timestep, num_channels, num_bands)
    """
    test_file = os.path.join(input_dir, filename)
    # Open the file and load the data
    test_data = np.load(test_file)['arr_0']
    return test_data


def install_from_whitelist(req_file):
    whitelist = open("/app/program/whitelist.txt", 'r').readlines()
    whitelist = [i.rstrip('\n') for i in whitelist]
    # print(whitelist)

    for package in open(req_file, 'r').readlines():
        package = package.rstrip('\n')
        package_version = package.split("==")
        if len(package_version) > 2:
            # invalid format, don't use
            print(
                f"requested package {package} has invalid format, will install latest version (of {package_version[0]}) if allowed")
            package = package_version[0]
        elif len(package_version) == 2:
            version_str = package_version[1]
            Version(version_str)
            # try:
            #     Version(version_str)
            # except InvalidVersion:
            #     print(f"requested package {package} has invalid version, will install latest version (of {package_version[0]}) if allowed")
            #     package = package_version[0]

        # print("accepted package name: ", package)
        # print("package name ", package_version[0])
        if package_version[0] in whitelist:
            # package must be in whitelist, so format check unnecessary
            subprocess.check_call(
                [executable, "-m", "pip", "install", package])
            print(f"{package_version[0]} installed")
        else:
            exit(
                f"{package_version[0]} is not an allowed package. Please contact the organizers on GitHub to request acceptance of the package.")


def print_pretty(text):
    print("-------------------")
    print("#---", text)
    print("-------------------")


def save_prediction(prediction_nparray, dataset_name):

    # Save the prediction to a npz file
    np.savez(os.path.join(
        output_dir, f'test_{dataset_name}.predictions'), prediction_nparray)


def main():
    """
     Run the pipeline
     > Load
     > Predict
     > Save
    """

    start = time.time()

    requirements_file = os.path.join(submission_dir, "requirements.txt")
    if os.path.isfile(requirements_file):
        install_from_whitelist(requirements_file)

    duration = time.time() - start
    print_pretty(f'Duration of the package installation: {duration}')

    start = time.time()

    print_pretty('Reading Data')
    affi_test = load_dataset('test_data_affi.npz')
    beignet_test = load_dataset('test_data_beignet.npz')
    affi_test_private = load_dataset('test_data_affi_2024-03-20_private.npz')
    beignet_test_private_1 = load_dataset(
        'test_data_beignet_2022-06-01_private.npz')
    beignet_test_private_2 = load_dataset(
        'test_data_beignet_2022-06-02_private.npz')

    print_pretty('Loading trained Model')
    m = Model()
    m.load()

    print_pretty('Making Prediction')
    print(affi_test[0])
    prediction_affi = m.predict(affi_test)
    print(prediction_affi[0])
    prediction_beignet = m.predict(beignet_test)
    prediction_affi_private = m.predict(affi_test_private)
    prediction_beignet_private_1 = m.predict(beignet_test_private_1)
    prediction_beignet_private_2 = m.predict(beignet_test_private_2)

    print_pretty('Saving Prediction')
    save_prediction(prediction_affi, 'affi')
    save_prediction(prediction_beignet, 'beignet')
    save_prediction(prediction_affi_private, 'affi_private')
    save_prediction(prediction_beignet_private_1, 'beignet_private_1')
    save_prediction(prediction_beignet_private_2, 'beignet_private_2')

    duration = time.time() - start
    print_pretty(f'Total duration: {duration}')

    # Compare affi_test and prediction_affi and check if they are the same with np
    print(np.array_equal(affi_test, prediction_affi))
    print(np.array_equal(beignet_test, prediction_beignet))
    print(np.array_equal(affi_test_private, prediction_affi_private))
    print(np.array_equal(beignet_test_private_1, prediction_beignet_private_1))
    print(np.array_equal(beignet_test_private_2, prediction_beignet_private_2))


if __name__ == '__main__':
    main()
