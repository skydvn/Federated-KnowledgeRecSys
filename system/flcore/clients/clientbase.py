import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        """
        Compute test metrics including accuracy and AUC
        """
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob_list = []
        y_true_list = []

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(testloaderfull):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                # Get predictions
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                # Get probabilities for AUC calculation
                probs = torch.nn.functional.softmax(output, dim=1)

                # Convert to numpy immediately and store
                batch_probs = probs.detach().cpu().numpy()
                batch_labels = y.detach().cpu().numpy()

                y_prob_list.append(batch_probs)
                y_true_list.append(batch_labels)

                # Debug first batch
                if batch_idx == 0:
                    print(f"First batch - probs shape: {batch_probs.shape}, labels shape: {batch_labels.shape}")

        # Concatenate all batches
        y_prob = np.vstack(y_prob_list)  # Stack vertically to preserve (samples, classes) shape
        y_true = np.concatenate(y_true_list)

        print(f"Final shapes - y_prob: {y_prob.shape}, y_true: {y_true.shape}")
        print(f"Test accuracy: {test_acc}/{test_num} = {test_acc / test_num:.4f}")

        # Calculate AUC
        try:
            # Check if we have both classes in y_true
            unique_labels = np.unique(y_true)
            print(f"Unique labels: {unique_labels}")

            if len(unique_labels) < 2:
                print(f"Warning: Only one class present in test data: {unique_labels}")
                auc = 0.0
            elif self.num_classes == 2:
                # Binary classification: use probability of positive class
                auc = metrics.roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Multi-class classification
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                auc = metrics.roc_auc_score(y_true_bin, y_prob, average='micro')

        except Exception as e:
            print(f"Error calculating AUC: {e}")
            print(f"y_true shape: {y_true.shape}, unique values: {np.unique(y_true)}")
            print(f"y_prob shape: {y_prob.shape}, range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
            auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
