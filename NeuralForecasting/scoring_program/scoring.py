import sys
import os
import json
import numpy as np
import torch

# Directory to read labels from
input_dir = sys.argv[1]
solutions = os.path.join(input_dir, 'ref')
prediction_dir = os.path.join(input_dir, 'res')

# Directory to output computed score into
output_dir = sys.argv[2]


def calculate_mse(array1, array2):
    """Calculate MSE between two arrays of the same shape"""
    if array1.shape != array2.shape:
        raise ValueError(
            f"Shapes don't match: {array1.shape} vs {array2.shape}")

    # Convert to tensors if needed
    # 10 steps are the initial steps as input for the model
    # Remove the first 10 steps from the score calculation
    array1 = torch.tensor(array1[:, 10:])
    array2 = torch.tensor(array2[:, 10:])

    # Calculate MSE
    mse = torch.nn.functional.mse_loss(array1, array2, reduction='mean')
    return mse.item()


def read_prediction(dataset_name):
    prediction_file = os.path.join(
        prediction_dir, f'test_{dataset_name}.predictions.npz')

    # Check if file exists
    if not os.path.isfile(prediction_file):
        print('[-] Test prediction file not found!')
        print(prediction_file)
        return

    predicted_array = np.load(prediction_file)['arr_0']

    return predicted_array


def read_solution(dataset_name):
    solution_file = os.path.join(solutions, f'test_data_{dataset_name}.npz')

    # Check if file exists
    if not os.path.isfile(solution_file):
        print('[-] Test solution file not found!')
        return

    test_array = np.load(solution_file)['arr_0']

    return test_array


def save_score(MSE_affi, MSE_beignet, MSE_affi_private, MSE_beignet_private_1, MSE_beignet_private_2,
               total_MSR):
    score_file = os.path.join(output_dir, 'scores.json')

    # Create a dictionary of scores
    scores = {
        'MSE_affi': MSE_affi,
        'MSE_beignet': MSE_beignet,
        'MSE_affi_private': MSE_affi_private,
        'MSE_beignet_private_1': MSE_beignet_private_1,
        'MSE_beignet_private_2': MSE_beignet_private_2,
        'total_MSR': total_MSR
    }

    with open(score_file, 'w') as f_score:
        f_score.write(json.dumps(scores))
        f_score.close()


def print_pretty(text):
    print("-------------------")
    print("#---", text)
    print("-------------------")


def main():

    # Read prediction and solution
    print_pretty('Reading prediction')
    prediction_affi = read_prediction('affi')
    prediction_beignet = read_prediction('beignet')
    prediction_affi_private = read_prediction('affi_private')
    prediction_beignet_private_1 = read_prediction('beignet_private_1')
    prediction_beignet_private_2 = read_prediction('beignet_private_2')

    print_pretty('Reading solution')
    solution_affi = read_solution('affi')
    solution_beignet = read_solution('beignet')
    solution_affi_private = read_solution('affi_2024-03-20_private')
    solution_beignet_private_1 = read_solution('beignet_2022-06-01_private')
    solution_beignet_private_2 = read_solution('beignet_2022-06-02_private')

    # Compute MSE
    print_pretty('Computing MSE score')
    MSE_affi = calculate_mse(solution_affi, prediction_affi)
    MSE_beignet = calculate_mse(solution_beignet, prediction_beignet)
    MSE_affi_private = calculate_mse(
        solution_affi_private, prediction_affi_private)
    MSE_beignet_private_1 = calculate_mse(
        solution_beignet_private_1, prediction_beignet_private_1)
    MSE_beignet_private_2 = calculate_mse(
        solution_beignet_private_2, prediction_beignet_private_2)

    # Compute the total MSE
    total_MSR = (MSE_affi + MSE_beignet + MSE_affi_private +
                 MSE_beignet_private_1 + MSE_beignet_private_2) / 5
    # Write Score
    print_pretty('Saving prediction')
    save_score(MSE_affi, MSE_beignet, MSE_affi_private,
               MSE_beignet_private_1, MSE_beignet_private_2, total_MSR)

    # Print all the scores
    print_pretty(f'MSE_affi: {MSE_affi}')
    print_pretty(f'MSE_beignet: {MSE_beignet}')
    print_pretty(f'MSE_affi_private: {MSE_affi_private}')
    print_pretty(f'MSE_beignet_private_1: {MSE_beignet_private_1}')
    print_pretty(f'MSE_beignet_private_2: {MSE_beignet_private_2}')

    print_pretty(f'total_MSR: {total_MSR}')


if __name__ == '__main__':
    main()
