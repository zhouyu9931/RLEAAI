import torch
import torch.nn.functional as F


def max_shape(data):
    """Gets the maximum length along all dimensions in a list of Tensors"""
    shapes = torch.Tensor([_.shape for _ in data])
    return torch.max(shapes.transpose(0, 1), dim=1)[0].int()


def pad_data_to_same_shape(tensor_list, pad_value=0):
    target_shape = max_shape(tensor_list)

    padded_dataset_shape = [len(tensor_list)] + list(target_shape)
    padded_dataset = torch.Tensor(*padded_dataset_shape).type_as(
        tensor_list[0])

    for i, data in enumerate(tensor_list):
        # Get how much padding is needed per dimension
        padding = reversed(target_shape - torch.Tensor(list(data.shape)).int())

        # Add 0 every other index to indicate only right padding
        padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(-1, 1)
        padding = padding.view(1, -1)[0].tolist()

        padded_data = F.pad(data, padding, value=pad_value)
        padded_dataset[i] = padded_data

    return padded_dataset

# def pad_data_to_same_shape(tensor_list, pad_value=0):
#     shapes = torch.tensor([_.shape for _ in tensor_list])
#     target_shape = torch.max(shapes, dim=0)[0]
#
#     padded_dataset = []
#     for data in tensor_list:
#         padding = [0, 0]  # No padding along batch dimension
#         for i in range(len(target_shape) - 1):
#             padding.append(target_shape[i + 1] - data.shape[i])
#             padding.append(0)
#         padded_data = F.pad(data, padding, value=pad_value)
#         padded_dataset.append(padded_data)
#
#     return torch.stack(padded_dataset)