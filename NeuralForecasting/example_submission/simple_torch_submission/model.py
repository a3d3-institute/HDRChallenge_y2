import torch
import numpy as np
from torch.utils.data import Dataset


def normalize(data, average=[], std=[]):
    # normalizing input to the range of [~mean - 4*std, ~mean + 4*std]
    # becasue in this dataset, the timestep is in first dimension.
    # adapt the normalization to fit this situation, compute mean and std in
    # axis 0
    if data.ndim == 4:
        n, t, c, f = data.shape
        data = data.reshape((n*t, -1))  # neuron input
    if len(average) == 0:
        average = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
    combine_max = average + 4 * std
    combine_min = average - 4 * std
    norm_data = 2 * (data - combine_min) / (combine_max - combine_min) - 1
    norm_data = norm_data.reshape((n, t, c, f))
    return norm_data, average, std


class NeuroForcastDataset(Dataset):
    def __init__(self, neural_data, use_graph=False, average=[], std=[]):
        """
        neural_data: N*T*C*F (sampe size * total time steps * channel 
        *feature dimension)
        f_window: T' the length of prediction window
        batch_size: batch size
        """
        self.data = neural_data
        self.use_graph = use_graph
        if len(average) == 0:
            self.data, self.average, self.std = normalize(self.data)
        else:
            self.data, self.average, self.std = normalize(
                self.data, average, std)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if not self.use_graph:
            data = data[:, :, 0]

        data = torch.tensor(data, dtype=torch.float32)
        return data


class Model(torch.nn.Module):
    def __init__(self, monkey_name='beignet'):
        super(Model, self).__init__()
        self.monkey_name = monkey_name
        if self.monkey_name == 'beignet':
            self.input_size = 89
            self.hidden_size = 256
        elif self.monkey_name == 'affi':
            self.input_size = 239
            self.hidden_size = 256
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')
        self.encoder = torch.nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.output_layer = torch.nn.Linear(self.hidden_size, self.input_size)
        # Load average and std from file during initialization
        try:
            if self.monkey_name == 'beignet':
                data = np.load('train_data_average_std_beignet.npz')
            elif self.monkey_name == 'affi':
                data = np.load('train_data_average_std_affi.npz')
            else:
                raise ValueError(f'No such a monkey: {self.monkey_name}')
            self.average = data['average']
            self.std = data['std']
        except FileNotFoundError:
            print("Warning: train_data_average_std.npz not found. "
                  "Will load during predict.")
            self.average = None
            self.std = None

    def forward(self, x):
        output, hidden = self.encoder(x)
        output = self.output_layer(output)
        return output

    def load(self):
        path = "model.pth"
        if self.monkey_name == 'beignet':
            path = "model_beignet.pth"
        elif self.monkey_name == 'affi':
            path = "model_affi.pth"
        else:
            raise ValueError(f'No such a monkey: {self.monkey_name}')
        self.load_state_dict(torch.load(path, weights_only=True))

    def predict(self, x):
        # Load average and std from file if not already loaded
        if self.average is None or self.std is None:
            data = np.load('train_data_average_std.npz')
            self.average = data['average']
            self.std = data['std']

        # Create dataset from input data
        dataset = NeuroForcastDataset(
            x, use_graph=False, average=self.average, std=self.std)

        # Process each sample in the dataset
        predictions = []
        self.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                # Add batch dimension if needed
                if sample.dim() == 2:
                    sample = sample.unsqueeze(0)  # Add batch dimension

                # Forward pass
                output = self.forward(sample)

                # Remove batch dimension and convert to numpy
                if output.dim() == 3:
                    output = output.squeeze(0)  # Remove batch dimension

                predictions.append(output.numpy())

        # Stack all predictions
        return np.array(predictions)
