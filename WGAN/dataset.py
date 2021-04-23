import torch as th
from torch.utils.data import Dataset


class StrainDataset(Dataset):
    def __init__(self, path_to_data, max_length, min_length, overlapping, data_augmentation, initial_stage=0):
        """
        Write later
        """

        # Body of the constructor - overwrite pytorch Dataset class
        super(StrainDataset, self).__init__()
        self.strain_dict = th.load(path_to_data)
        # Detector sampling frequency - fixed!
        self.fs = 4096
        # Starting length (in num of samples) of the elements
        self.initial_length = min_length
        # Starting length (in num of samples) of the elements
        self.final_length = max_length
        # Data augmentation (bool)
        self.data_augmentation = data_augmentation
        # Number of overlapping time samples between elements (when they're at max length)
        self.overlapping_samples = int(max_length * overlapping)
        # Number of chunks in strain_dict
        self.n_chunks = self.strain_dict["num_of_chunks"]
        # Now get the number of dataset elements in each chunk in strain_dict
        self.n_elements_per_chunks = []
        for i in range(self.n_chunks):
            n = (len(self.strain_dict[int(i)]) - self.final_length) // \
                (self.final_length - self.overlapping_samples) + 1
            self.n_elements_per_chunks.append(n)
        # And the total number of elements
        # The actual number of elements is 2 times n_elements as we add a sign flip
        self.n_elements = sum(self.n_elements_per_chunks)
        # Set stage
        self.stage = initial_stage

    def __len__(self):
        """ The len method returns the length of the dataset, which is the number of chunks (elements) in it
        """
        return self.n_elements * (1 + int(self.data_augmentation))  # The factor 2 takes into account the sign flip

    def get_segment_in_chunk(self, item_idx, chunk_number):
        """
        Write later
        """
        strain = self.strain_dict[int(chunk_number)]
        start = (self.final_length - self.overlapping_samples) * item_idx
        end = start + self.final_length
        stride = (self.final_length // self.initial_length) // (2 ** self.stage)
        return strain[start:end:stride]
        
    def __getitem__(self, item_idx):
        """ Write later
        """
        try:
            idx = item_idx.item()
        except AttributeError:
            idx = item_idx

        if idx >= (1 + int(self.data_augmentation)) * self.n_elements:
            raise IndexError

        sign = 1
        if idx >= self.n_elements:
            idx -= self.n_elements
            sign = - 1
        n = 0
        element = None
        while n < self.n_chunks:
            if idx < self.n_elements_per_chunks[n]:
                element = self.get_segment_in_chunk(idx, n)
                break
            else:
                idx -= self.n_elements_per_chunks[n]
            n += 1
        return sign * element.float()
