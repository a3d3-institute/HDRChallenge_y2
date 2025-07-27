import numpy as np
import os
import pickle


def load_dataset(filename):
    """
    filenames: choice from beignet or affi to load corresponding dataset
    return:
    training_data: samples to train with shape (num_samples, num_timestep, num_channels, num_bands)
    test_data: samples to test with shape (num_samples, num_timestep, num_channels, num_bands)
    val_data: samples to train with shape (num_samples, num_timestep, num_channels, num_bands)
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


# Public dataset
train_data_affi, test_data_affi, val_data_affi = load_dataset(
    'affi')  # B * T * C * F

# Print the shape of the data
print(train_data_affi.shape)
print(test_data_affi.shape)
print(val_data_affi.shape)

# Print fraction of train / test / val to total data
print(
    f"Fraction of train data: {len(train_data_affi) / (len(train_data_affi) + len(test_data_affi) + len(val_data_affi)):.2f}")
print(
    f"Fraction of test data: {len(test_data_affi) / (len(train_data_affi) + len(test_data_affi) + len(val_data_affi)):.2f}")
print(
    f"Fraction of val data: {len(val_data_affi) / (len(train_data_affi) + len(test_data_affi) + len(val_data_affi)):.2f}")


# Save the data to a file only if the file does not exist
np.savez('./postprocessed_dataset/train_data_affi.npz', train_data_affi)
np.savez('./postprocessed_dataset/test_data_affi.npz', test_data_affi)
np.savez('./postprocessed_dataset/val_data_affi.npz', val_data_affi)


train_data_beignet, test_data_beignet, val_data_beignet = load_dataset(
    'beignet')  # B * T * C * F

# Print the shape of the data
print(train_data_beignet.shape)
print(test_data_beignet.shape)
print(val_data_beignet.shape)

# Print fraction of train / test / val to total data
print(
    f"Fraction of train data: {len(train_data_beignet) / (len(train_data_beignet) + len(test_data_beignet) + len(val_data_beignet)):.2f}")
print(
    f"Fraction of test data: {len(test_data_beignet) / (len(train_data_beignet) + len(test_data_beignet) + len(val_data_beignet)):.2f}")
print(
    f"Fraction of val data: {len(val_data_beignet) / (len(train_data_beignet) + len(test_data_beignet) + len(val_data_beignet)):.2f}")


# Save the data to a file only if the file does not exist
np.savez('./postprocessed_dataset/train_data_beignet.npz', train_data_beignet)
np.savez('./postprocessed_dataset/test_data_beignet.npz', test_data_beignet)
np.savez('./postprocessed_dataset/val_data_beignet.npz', val_data_beignet)


def process_private_dataset(filename):
    """
    filename: choice from beignet or affi to load corresponding dataset
    return:
    train_data: samples to train with shape (num_samples, num_timestep, num_channels, num_bands)
    test_data: samples to test with shape (num_samples, num_timestep, num_channels, num_bands)
    val_data: samples to train with shape (num_samples, num_timestep, num_channels, num_bands)
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


if __name__ == "__main__":

    # Process private datasets
    process_private_dataset('beignet_2022-06-02_5423_subset.pkl')
    process_private_dataset('beignet_2022-06-01_5405_subset.pkl')
    process_private_dataset('affi_2024-03-20_15499_subset.pkl')
