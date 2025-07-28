import numpy as np
import os
import pickle
import torch


def load_dataset(filename):
    """
    filenames: choice from beignet or affi to load corresponding dataset
    """
    if filename == 'affi' or filename == 'beignet':
        lfp_array = np.load(f'lfp_{filename}.npy')
        indices = np.load(f'tvts_{filename}_split.npz')
        testing_index = indices['testing_index']
        training_index = indices['train_index']
        val_index = indices['val_index']
        training_data = lfp_array[training_index]
        test_data = lfp_array[testing_index]
        val_data = lfp_array[val_index]
        print('Total number of samples', len(lfp_array), 'Num Training', len(training_data),
              'Num test', len(test_data), 'Num val', len(val_data))
        return training_data, test_data, val_data
    else:
        raise NotImplementedError('No such a dataset')


def prepare_masked_data(data, init_steps=10):
    """
    data: numpy array of shape (num_samples, num_timestep, num_channels, num_bands)
    init_steps: number of initial steps to use as input
    return: numpy array of shape (num_samples, num_timestep, num_channels, num_bands)
    """

    # Convert to tensor
    test_data_tensor = torch.tensor(data)
    future_step = data.shape[1] - init_steps
    test_data_tensor = torch.cat([test_data_tensor[:, :init_steps], torch.repeat_interleave(test_data_tensor[:, init_steps-1:init_steps],
                                                                                            future_step, dim=1)], dim=1)
    # convert back to numpy
    test_data_tensor = test_data_tensor.numpy()

    return test_data_tensor


def process_public_dataset(filename):
    """
    filename: choice from affi or beignet to load corresponding dataset
    """

    # Public dataset
    train_data, test_data, val_data = load_dataset(
        filename)  # B * T * C * F

    test_data_masked = prepare_masked_data(test_data)
    val_data_masked = prepare_masked_data(val_data)
# Print fraction of train / test / val to total data
    print(
        f"Fraction of train data: {len(train_data) / (len(train_data) + len(test_data) + len(val_data)):.2f}")
    print(
        f"Fraction of test data: {len(test_data) / (len(train_data) + len(test_data) + len(val_data)):.2f}")
    print(
        f"Fraction of val data: {len(val_data) / (len(train_data) + len(test_data) + len(val_data)):.2f}")

    # Save the data to a file only if the file does not exist
    np.savez(f'./postprocessed_dataset/train_data_{filename}.npz', train_data)
    np.savez(f'./postprocessed_dataset/test_data_{filename}.npz', test_data)
    np.savez(f'./postprocessed_dataset/val_data_{filename}.npz', val_data)
    np.savez(f'./postprocessed_dataset/test_data_{filename}_masked.npz',
             test_data_masked)
    np.savez(f'./postprocessed_dataset/val_data_{filename}_masked.npz',
             val_data_masked)


def process_private_dataset(filename):
    """
    filename: choice from beignet or affi to load corresponding dataset
    """

    # Grab data name from filename
    data_name = filename.split('_')[0]

    # Grab data date from filename
    data_date = filename.split('_')[1]

    # Print data name and date
    print(f"Data name: {data_name}")
    print(f"Data date: {data_date}")

    # Private dataset
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        lfp_array = np.array(data['lfp'])
        lfp_array = np.swapaxes(lfp_array, 2, 3)
        print(lfp_array.shape)

        # Print number of samples
        print(len(lfp_array))

        # Split into train (20%), test (20%), val (60%) randomly with seed 42
        np.random.seed(42)
        indices = np.random.permutation(len(lfp_array))
        train_data = lfp_array[indices[:int(len(lfp_array) * 0.2)]]
        test_data = lfp_array[indices[int(
            len(lfp_array) * 0.2):int(len(lfp_array) * 0.4)]]
        val_data = lfp_array[indices[int(len(lfp_array) * 0.4):]]
        print(train_data.shape)
        print(test_data.shape)
        print(val_data.shape)

        test_data_masked = prepare_masked_data(test_data)
        val_data_masked = prepare_masked_data(val_data)

        # Print fraction of train / test / val to total data
        print(
            f"Fraction of train data: {len(train_data) / (len(train_data) + len(test_data) + len(val_data)):.2f}")
        print(
            f"Fraction of test data: {len(test_data) / (len(train_data) + len(test_data) + len(val_data)):.2f}")
        print(
            f"Fraction of val data: {len(val_data) / (len(train_data) + len(test_data) + len(val_data)):.2f}")

        # Save the data to a file only if the file does not exist
        np.savez(
            f'./postprocessed_dataset/train_data_{data_name}_{data_date}_private.npz', train_data)
        np.savez(
            f'./postprocessed_dataset/test_data_{data_name}_{data_date}_private.npz', test_data)
        np.savez(
            f'./postprocessed_dataset/val_data_{data_name}_{data_date}_private.npz', val_data)
        np.savez(
            f'./postprocessed_dataset/test_data_{data_name}_{data_date}_private_masked.npz', test_data_masked)
        np.savez(
            f'./postprocessed_dataset/val_data_{data_name}_{data_date}_private_masked.npz', val_data_masked)


if __name__ == "__main__":

    # Process public datasets
    process_public_dataset('affi')
    process_public_dataset('beignet')

    # Process private datasets
    process_private_dataset('beignet_2022-06-02_5423_subset.pkl')
    process_private_dataset('beignet_2022-06-01_5405_subset.pkl')
    process_private_dataset('affi_2024-03-20_15499_subset.pkl')
