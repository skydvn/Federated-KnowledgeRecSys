import numpy as np
import os
import random
from utils.dataset_utils import split_data, save_file
from scipy.sparse import coo_matrix
from os import path
import json


# https://github.com/FengHZ/KD3A/blob/master/datasets/AmazonReview.py
def load_amazon(base_path):
    dimension = 5000
    amazon = np.load(path.join(base_path, "amazon.npz"))
    amazon_xx = coo_matrix((amazon['xx_data'], (amazon['xx_col'], amazon['xx_row'])),
                           shape=amazon['xx_shape'][::-1]).tocsc()
    amazon_xx = amazon_xx[:, :dimension]
    amazon_yy = amazon['yy']
    amazon_yy = (amazon_yy + 1) / 2
    amazon_offset = amazon['offset'].flatten()
    # Partition the data into four categories and for each category partition the data set into training and test set.
    data_name = ["books", "dvd", "electronics", "kitchen"]
    num_data_sets = 4
    data_insts, data_labels, num_insts = [], [], []
    for i in range(num_data_sets):
        data_insts.append(amazon_xx[amazon_offset[i]: amazon_offset[i + 1], :])
        data_labels.append(amazon_yy[amazon_offset[i]: amazon_offset[i + 1], :])
        num_insts.append(amazon_offset[i + 1] - amazon_offset[i])
        # Randomly shuffle.
        r_order = np.arange(num_insts[i])
        np.random.shuffle(r_order)
        data_insts[i] = data_insts[i][r_order, :]
        data_labels[i] = data_labels[i][r_order, :]
        data_insts[i] = data_insts[i].todense().astype(np.float32)
        data_labels[i] = data_labels[i].ravel().astype(np.int64)
    return data_insts, data_labels


random.seed(1)
np.random.seed(1)
data_path = "AmazonReview/"
dir_path = "AmazonReview/"


# Allocate data to users
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    global_test_path = dir_path + "global_test/"  # New global test directory

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(global_test_path):
        os.makedirs(global_test_path)

    rawdata_dir = data_path + "rawdata"

    # Get AmazonReview data
    if not os.path.exists(rawdata_dir):
        os.makedirs(rawdata_dir)
        os.system(
            f'wget https://drive.google.com/u/0/uc?id=1QbXFENNyqor1IlCpRRFtOluI2_hMEd1W&export=download -P {rawdata_dir}')
    if not os.path.exists(rawdata_dir + '/AmazonReview'):
        os.system(f'unzip {rawdata_dir}/AmazonReview.zip -d {rawdata_dir}')
    rawdata_dir = rawdata_dir + '/AmazonReview'

    X, y = load_amazon(rawdata_dir)

    labelss = []
    for yy in y:
        labelss.append(len(set(yy)))
    num_clients = len(y)
    print(f'Number of labels: {labelss}')
    print(f'Number of clients: {num_clients}')

    statistic = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    train_data, test_data = split_data(X, y)

    # Create global test dataset by combining all client test sets
    global_test_data = create_global_test_set(test_data, num_clients)

    # Save individual client data
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, max(labelss),
              statistic, None, None, None)

    # Save global test dataset
    save_global_test_set(global_test_path, global_test_data, num_clients, max(labelss))


def create_global_test_set(test_data, num_clients):
    """
    Combine all client test sets into a single global test set

    Args:
        test_data: Dictionary with client test data
        num_clients: Number of clients

    Returns:
        Dictionary containing combined test data
    """
    all_X = []
    all_y = []
    client_indices = []  # Track which samples belong to which client

    for client_id in range(num_clients):
        client_data = test_data[client_id]
        X_test = client_data['x']
        y_test = client_data['y']

        all_X.append(X_test)
        all_y.append(y_test)
        client_indices.extend([client_id] * len(y_test))

    # Concatenate all data
    global_X = np.vstack(all_X)
    global_y = np.concatenate(all_y)
    client_indices = np.array(client_indices)

    print("\n" + "=" * 50)
    print("Global Test Set Statistics:")
    print(f"Total samples: {len(global_y)}")
    print(f"Feature dimension: {global_X.shape[1]}")
    print(f"Label distribution: {np.unique(global_y, return_counts=True)}")
    print(f"Samples per client: {[sum(client_indices == i) for i in range(num_clients)]}")
    print("=" * 50 + "\n")

    global_test_data = {
        'x': global_X,
        'y': global_y,
        'client_indices': client_indices  # Track origin of each sample
    }

    return global_test_data


def save_global_test_set(global_test_path, global_test_data, num_clients, num_classes):
    """
    Save global test dataset and its statistics

    Args:
        global_test_path: Path to save global test data
        global_test_data: Dictionary containing global test data
        num_clients: Number of clients
        num_classes: Number of classes
    """
    # Save the data
    data_file = os.path.join(global_test_path, "global_test.npz")
    np.savez(data_file,
             x=global_test_data['x'],
             y=global_test_data['y'],
             client_indices=global_test_data['client_indices'])

    # Create statistics
    global_y = global_test_data['y']
    client_indices = global_test_data['client_indices']

    # Overall statistics
    statistic = []
    for label in np.unique(global_y):
        statistic.append((int(label), int(sum(global_y == label))))

    # Per-client statistics within global test set
    client_statistics = []
    for client_id in range(num_clients):
        client_mask = client_indices == client_id
        client_labels = global_y[client_mask]
        client_stat = []
        for label in np.unique(client_labels):
            client_stat.append((int(label), int(sum(client_labels == label))))
        client_statistics.append(client_stat)

    # Save metadata
    metadata = {
        'num_samples': int(len(global_y)),
        'num_features': int(global_test_data['x'].shape[1]),
        'num_clients': num_clients,
        'num_classes': num_classes,
        'label_distribution': statistic,
        'client_statistics': client_statistics,
        'samples_per_client': [int(sum(client_indices == i)) for i in range(num_clients)]
    }

    metadata_file = os.path.join(global_test_path, "global_test_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Global test set saved to {global_test_path}")
    print(f"- Data file: {data_file}")
    print(f"- Metadata file: {metadata_file}")


if __name__ == "__main__":
    generate_dataset(dir_path)