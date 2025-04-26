import csv
import datetime
import os
import torch.optim as optim
from torch.utils.data import DataLoader

from new_loader import TorchTensorFolderDataset


def get_loader(data_size: str = "sample", denoised: bool = False, use_mel: bool = True, target_data: str = "train", batch_size: int = 16):
    path = os.path.join(os.getcwd(), "data", "preprocessed", data_size, "denoised" if denoised else "standard", "mel" if use_mel else "raw", target_data)

    dataset = TorchTensorFolderDataset(path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


def save_to_csv(filename, column_names, data):
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H.%M")
    filename = f"{filename}_{timestamp}.csv"

    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        writer.writerows(data)

    return


# def test_learning_rates(model, train_model, test_model, times=3, learning_rates=[0.1, 0.01, 0.001], data_size="sample", denoised=False, use_mel=True):
#     val_loader = get_loader(data_size, denoised, use_mel, "validation")
#     test_loader = get_loader(data_size, denoised, use_mel, "test")

#     result = []
#     column_names = ["learning_rate", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
#     print(*column_names)

#     for learning_rate in learning_rates:
#         for _ in range(times):
#             train_loader = get_loader(data_size, denoised, use_mel, "train")

#             optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#             train_model(model, optimizer, train_loader, val_loader, num_epochs=10)

#             train_loss, train_acc = test_model(model, train_loader)
#             val_loss, val_acc = test_model(model, val_loader)
#             test_loss, test_acc = test_model(model, test_loader)

#             result.append((learning_rate, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

#             print(*result[-1])

#     save_to_csv("results/learning_rates_test.csv", column_names, result)

#     return


# def test_batch_sizes(model, train_model, test_model, times=3, batch_sizes=[16, 32, 64], data_size="sample", denoised=False, use_mel=True):
#     val_loader = get_loader(data_size, denoised, use_mel, "validation")
#     test_loader = get_loader(data_size, denoised, use_mel, "test")

#     result = []
#     column_names = ["batch_size", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
#     print(*column_names)

#     for batch_size in batch_sizes:
#         for _ in range(times):
#             train_loader = get_loader(data_size, denoised, use_mel, "train", batch_size)

#             optimizer = optim.Adam(model.parameters(), lr=0.001)

#             train_model(model, optimizer, train_loader, val_loader, num_epochs=10)

#             train_loss, train_acc = test_model(model, train_loader)
#             val_loss, val_acc = test_model(model, val_loader)
#             test_loss, test_acc = test_model(model, test_loader)

#             result.append((batch_size, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

#             print(*result[-1])

#     save_to_csv("results/batch_sizes_test.csv", column_names, result)

#     return


# def test_weights_decays(model, train_model, test_model, times=3, weights_decays=[0.1, 0.5, 0.9], data_size="sample", denoised=False, use_mel=True):
#     val_loader = get_loader(data_size, denoised, use_mel, "validation")
#     test_loader = get_loader(data_size, denoised, use_mel, "test")

#     result = []
#     column_names = ["weights_decays", "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
#     print(*column_names)

#     for weights_decay in weights_decays:
#         for _ in range(times):
#             train_loader = get_loader(data_size, denoised, use_mel, "train")

#             optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=weights_decay)

#             train_model(model, optimizer, train_loader, val_loader, num_epochs=10)

#             train_loss, train_acc = test_model(model, train_loader)
#             val_loss, val_acc = test_model(model, val_loader)
#             test_loss, test_acc = test_model(model, test_loader)

#             result.append((weights_decay, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

#             print(*result[-1])

#     save_to_csv("results/weights_decays_test.csv", column_names, result)

#     return
