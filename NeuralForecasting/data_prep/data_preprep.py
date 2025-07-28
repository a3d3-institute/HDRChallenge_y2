"""
Neural Signal Forecasting Data Preprocessing Module

This module handles the preprocessing of neural signal data for forecasting 
tasks. It supports both public datasets (affi, beignet) and private datasets 
with different data formats and splitting strategies.

Date: 2025-07-28
Version: 0.1
"""

import numpy as np
import os
import pickle
import torch
from typing import Tuple


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split neural signal dataset into training, testing, and validation 
    sets.

    This function loads pre-split datasets that have predefined 
    train/test/validation indices stored in .npz files. The data is loaded from 
    .npy files and split according to the stored indices.

    Args:
        filename (str): Dataset name, either 'affi' or 'beignet'

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - training_data: Training samples with shape 
              (num_train_samples, num_timesteps, num_channels, num_bands)
            - test_data: Test samples with shape 
              (num_test_samples, num_timesteps, num_channels, num_bands)
            - val_data: Validation samples with shape 
              (num_val_samples, num_timesteps, num_channels, num_bands)

    Raises:
        NotImplementedError: If the specified dataset is not supported
        FileNotFoundError: If required data files are missing

    Example:
        >>> train_data, test_data, val_data = load_dataset('beignet')
        >>> print(f"Training samples: {train_data.shape}")
        >>> print(f"Test samples: {test_data.shape}")
        >>> print(f"Validation samples: {val_data.shape}")
    """
    if filename not in ['affi', 'beignet']:
        raise NotImplementedError(
            f'Dataset "{filename}" is not supported. Use "affi" or "beignet"')

    try:
        # Load the main data array
        lfp_array = np.load(f'lfp_{filename}.npy')
        print(f"Loaded data for {filename}: {lfp_array.shape}")

        # Load the pre-defined train/test/validation split indices
        indices = np.load(f'tvts_{filename}_split.npz')
        testing_index = indices['testing_index']
        training_index = indices['train_index']
        val_index = indices['val_index']

        # Split the data according to the indices
        training_data = lfp_array[training_index]
        test_data = lfp_array[testing_index]
        val_data = lfp_array[val_index]

        # Print dataset statistics
        total_samples = len(lfp_array)
        print(f"Dataset {filename} statistics:")
        print(f"  Total samples: {total_samples}")
        train_pct = len(training_data) / total_samples
        test_pct = len(test_data) / total_samples
        val_pct = len(val_data) / total_samples
        print(f"  Training samples: {len(training_data)} ({train_pct:.1%})")
        print(f"  Test samples: {len(test_data)} ({test_pct:.1%})")
        print(f"  Validation samples: {len(val_data)} ({val_pct:.1%})")

        return training_data, test_data, val_data

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Required data files for dataset '{filename}' not found. "
            f"Ensure 'lfp_{filename}.npy' and 'tvts_{filename}_split.npz' exist."
        ) from e


def prepare_masked_data(data: np.ndarray, init_steps: int = 10) -> np.ndarray:
    """
    Prepare masked data for multi-step forecasting by repeating the last known 
    values.

    This function creates input sequences for multi-step forecasting where only 
    the first 'init_steps' time steps contain real data, and the remaining time 
    steps are filled with repeated copies of the last known value. This is a 
    common technique in neural forecasting to provide the model with a complete 
    input sequence while masking future information.

    Args:
        data (np.ndarray): Input data with shape 
                          (num_samples, num_timesteps, num_channels, num_bands)
        init_steps (int): Number of initial time steps to use as real data. 
                         The remaining steps will be filled with repeated last values.
                         Default is 10.

    Returns:
        np.ndarray: Masked data with shape 
                   (num_samples, num_timesteps, num_channels, num_bands)
                   where time steps after init_steps contain repeated last known values.

    Example:
        >>> original_data = np.random.rand(100, 20, 89, 9)  # 100 samples, 20 steps
        >>> masked_data = prepare_masked_data(original_data, init_steps=5)
        >>> print(f"Original shape: {original_data.shape}")
        >>> print(f"Masked shape: {masked_data.shape}")
        >>> # First 5 time steps are real data, last 15 are repeated values
    """
    if data.ndim != 4:
        raise ValueError(
            f"Expected 4D array, got {data.ndim}D array with shape {data.shape}")

    if init_steps >= data.shape[1]:
        raise ValueError(
            f"init_steps ({init_steps}) must be less than number of time steps "
            f"({data.shape[1]})")

    # Convert to tensor for efficient operations
    data_tensor = torch.tensor(data, dtype=torch.float32)

    # Calculate how many future steps we need to fill
    future_steps = data.shape[1] - init_steps

    # Create the masked input by concatenating:
    # 1. Real data: first init_steps time steps
    # 2. Repeated data: last known value repeated for future_steps
    masked_tensor = torch.cat([
        data_tensor[:, :init_steps],  # Real data
        torch.repeat_interleave(
            data_tensor[:, init_steps-1:init_steps],  # Last known value
            future_steps, dim=1  # Repeat for future steps
        )
    ], dim=1)

    # Convert back to numpy
    masked_data = masked_tensor.numpy()

    print(f"Prepared masked data: {data.shape} -> {masked_data.shape} "
          f"(init_steps={init_steps}, future_steps={future_steps})")

    return masked_data


def process_public_dataset(filename: str) -> None:
    """
    Process public datasets (affi, beignet) and save preprocessed data.

    This function loads public datasets, applies masking for multi-step forecasting,
    and saves the processed data to .npz files. Public datasets come with
    pre-defined train/test/validation splits.

    Args:
        filename (str): Dataset name, either 'affi' or 'beignet'

    Returns:
        None

    Side Effects:
        Creates the following files in './postprocessed_dataset/':
        - train_data_{filename}.npz
        - test_data_{filename}.npz  
        - val_data_{filename}.npz
        - test_data_{filename}_masked.npz
        - val_data_{filename}_masked.npz

    Example:
        >>> process_public_dataset('beignet')
        >>> # Creates: train_data_beignet.npz, test_data_beignet.npz, etc.
    """
    print(f"Processing public dataset: {filename}")

    # Load the dataset with pre-defined splits
    train_data, test_data, val_data = load_dataset(filename)

    # Apply masking for multi-step forecasting
    test_data_masked = prepare_masked_data(test_data)
    val_data_masked = prepare_masked_data(val_data)

    # Calculate and print data distribution
    total_samples = len(train_data) + len(test_data) + len(val_data)
    train_fraction = len(train_data) / total_samples
    test_fraction = len(test_data) / total_samples
    val_fraction = len(val_data) / total_samples

    print(f"Data distribution for {filename}:")
    print(f"  Training: {train_fraction:.1%} ({len(train_data)} samples)")
    print(f"  Testing: {test_fraction:.1%} ({len(test_data)} samples)")
    print(f"  Validation: {val_fraction:.1%} ({len(val_data)} samples)")

    # Ensure output directory exists
    os.makedirs('./postprocessed_dataset/', exist_ok=True)

    # Save processed data
    np.savez(f'./postprocessed_dataset/train_data_{filename}.npz', train_data)
    np.savez(f'./postprocessed_dataset/test_data_{filename}.npz', test_data)
    np.savez(f'./postprocessed_dataset/val_data_{filename}.npz', val_data)
    np.savez(
        f'./postprocessed_dataset/test_data_{filename}_masked.npz',
        test_data_masked)
    np.savez(
        f'./postprocessed_dataset/val_data_{filename}_masked.npz',
        val_data_masked)

    print(f"Successfully saved processed data for {filename}")


def process_private_dataset(filename: str) -> None:
    """
    Process private datasets from pickle files and save preprocessed data.

    This function loads private datasets from pickle files, applies random
    train/test/validation splits, and saves the processed data. Private datasets
    use a different splitting strategy (20% train, 20% test, 60% validation)
    compared to public datasets.

    Args:
        filename (str): Path to the pickle file containing private dataset.
                       Expected format: '{dataset_name}_{date}_{sample_count}_subset.pkl'
                       Example: 'beignet_2022-06-02_5423_subset.pkl'

    Returns:
        None

    Side Effects:
        Creates the following files in './postprocessed_dataset/':
        - train_data_{dataset_name}_{date}_private.npz
        - test_data_{dataset_name}_{date}_private.npz
        - val_data_{dataset_name}_{date}_private.npz
        - test_data_{dataset_name}_{date}_private_masked.npz
        - val_data_{dataset_name}_{date}_private_masked.npz

    Raises:
        FileNotFoundError: If the pickle file doesn't exist
        KeyError: If the pickle file doesn't contain expected 'lfp' key

    Example:
        >>> process_private_dataset('beignet_2022-06-02_5423_subset.pkl')
        >>> # Creates: train_data_beignet_2022-06-02_private.npz, etc.
    """
    print(f"Processing private dataset: {filename}")

    # Parse filename to extract dataset information
    filename_parts = filename.split('_')
    if len(filename_parts) < 3:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            f"Expected format: '{{dataset_name}}_{{date}}_{{sample_count}}_subset.pkl'")

    data_name = filename_parts[0]  # e.g., 'beignet'
    data_date = filename_parts[1]  # e.g., '2022-06-02'

    print(f"Dataset: {data_name}, Date: {data_date}")

    try:
        # Load data from pickle file
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        # Extract LFP data and transpose axes for consistency
        lfp_array = np.array(data['lfp'])
        # Swap frequency and channel dimensions
        lfp_array = np.swapaxes(lfp_array, 2, 3)

        print(f"Loaded data: {lfp_array.shape}")
        print(f"Total samples: {len(lfp_array)}")

        # Create random train/test/validation split (20%/20%/60%)
        np.random.seed(42)  # For reproducible splits
        indices = np.random.permutation(len(lfp_array))

        train_end = int(len(lfp_array) * 0.2)
        test_end = int(len(lfp_array) * 0.4)

        train_data = lfp_array[indices[:train_end]]
        test_data = lfp_array[indices[train_end:test_end]]
        val_data = lfp_array[indices[test_end:]]

        print("Split data:")
        print(f"  Training: {train_data.shape}")
        print(f"  Testing: {test_data.shape}")
        print(f"  Validation: {val_data.shape}")

        # Apply masking for multi-step forecasting
        test_data_masked = prepare_masked_data(test_data)
        val_data_masked = prepare_masked_data(val_data)

        # Calculate and print data distribution
        total_samples = len(train_data) + len(test_data) + len(val_data)
        train_fraction = len(train_data) / total_samples
        test_fraction = len(test_data) / total_samples
        val_fraction = len(val_data) / total_samples

        print("Data distribution:")
        print(f"  Training: {train_fraction:.1%} ({len(train_data)} samples)")
        print(f"  Testing: {test_fraction:.1%} ({len(test_data)} samples)")
        print(f"  Validation: {val_fraction:.1%} ({len(val_data)} samples)")

        # Ensure output directory exists
        os.makedirs('./postprocessed_dataset/', exist_ok=True)

        # Save processed data with descriptive filenames
        base_filename = f"{data_name}_{data_date}_private"
        np.savez(
            f'./postprocessed_dataset/train_data_{base_filename}.npz',
            train_data)
        np.savez(
            f'./postprocessed_dataset/test_data_{base_filename}.npz',
            test_data)
        np.savez(
            f'./postprocessed_dataset/val_data_{base_filename}.npz',
            val_data)
        np.savez(
            f'./postprocessed_dataset/test_data_{base_filename}_masked.npz',
            test_data_masked)
        np.savez(
            f'./postprocessed_dataset/val_data_{base_filename}_masked.npz',
            val_data_masked)

        print(f"Successfully saved processed data for {base_filename}")

    except FileNotFoundError:
        raise FileNotFoundError(f"Private dataset file not found: {filename}")
    except KeyError as e:
        raise KeyError(
            f"Pickle file {filename} doesn't contain expected key: {e}")
    except Exception as e:
        raise RuntimeError(f"Error processing private dataset {filename}: {e}")


def main() -> None:
    """
    Main function to process all datasets (public and private).

    This function orchestrates the processing of all available datasets:
    - Public datasets: affi, beignet
    - Private datasets: beignet_2022-06-02, beignet_2022-06-01, affi_2024-03-20

    The function processes datasets sequentially and prints progress.
    """
    print("Starting data preprocessing pipeline")

    try:
        # Process public datasets
        print("Processing public datasets...")
        process_public_dataset('affi')
        process_public_dataset('beignet')

        # Process private datasets
        print("Processing private datasets...")
        private_datasets = [
            'beignet_2022-06-02_5423_subset.pkl',
            'beignet_2022-06-01_5405_subset.pkl',
            'affi_2024-03-20_15499_subset.pkl'
        ]

        for dataset in private_datasets:
            process_private_dataset(dataset)

        print("Data preprocessing pipeline completed successfully!")

    except Exception as e:
        print(f"Data preprocessing pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
