import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset


class SwitchDataset:

    def __init__(self, batch_size=4, training_size=800):
        self.batch_size = batch_size
        self.training_size = training_size
        self.train_feature, self.test_feature = self.get_feature()
        self.train_response, self.test_response = self.get_response()
        self.mean, self.std = self.compute_train_statistics()
        self.train_loader, self.val_loader = self.get_dataloaders()

    def get_feature(self):
        data_feature = np.load("data/VOQ_samples5.npy")
        train_feature = data_feature[:self.training_size, :, :]
        # Should I also divide the elements of this matrix by 5?
        test_feature = data_feature[self.training_size:, :, :]
        return train_feature, test_feature
    
    def get_response(self):
        data_response = np.load("data/Matching_samples5.npy")
        train_response = data_response[:self.training_size, :, :]
        test_response = data_response[self.training_size:, :, :]
        return train_response, test_response

    def compute_train_statistics(self):
        # compute mean and std with respect to self.train_feature
        mean = np.mean(self.train_feature/5)  
        std = np.std(self.train_feature/5)  
        return mean, std

    def get_dataloaders(self):
        normalize = transforms.Normalize(self.mean, self.std)
        # train set
        train_set = TensorDataset(normalize(torch.tensor(self.train_feature)), torch.tensor(self.train_response))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)

        # validation set
        val_set = TensorDataset(normalize(torch.tensor(self.test_feature)), torch.tensor(self.test_response))
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    
if __name__ == '__main__':
    dataset = SwitchDataset()
    print(dataset.mean, dataset.std)