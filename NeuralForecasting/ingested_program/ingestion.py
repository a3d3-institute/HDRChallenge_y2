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


def load_dataset():
    """
    filenames: load test dataset
    return:
    test_data: samples to test with shape (num_samples, num_timestep, num_channels, num_bands)
    """
    filename = 'beignet'
    lfp_array = np.load(f'lfp_{filename}.npy')
    indices = np.load(f'tvts_{filename}_split.npz')
    testing_index = indices['testing_index']
    test_data = lfp_array[testing_index]

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


def save_prediction(prediction_nparray):

    # Save the 3 dim nparray to a file
    np.save(os.path.join(output_dir, 'test.predictions'), prediction_nparray)


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
    X_test = load_dataset()

    print_pretty('Loading trained Model')
    m = Model()
    m.load()

    print_pretty('Making Prediction')
    prediction_array = m.predict(X_test)

    print_pretty('Saving Prediction')
    save_prediction(prediction_array)

    duration = time.time() - start
    print_pretty(f'Total duration: {duration}')


if __name__ == '__main__':
    main()
