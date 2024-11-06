import h5py
import torch
import random
from torch.utils.data import random_split, Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


class HDF5Dataset(Dataset):

    def __init__(self, hdf5_file_human, hdf5_file_mouse):
        """
        Custom PyTorch Dataset for reading an HDF5 file.
        
        :param hdf5_file: Path to the HDF5 file.
        :param dataset_name: Name of the dataset inside the HDF5 file to read.
        """
        self.hdf5_file_human = hdf5_file_human
        self.hdf5_file_mouse = hdf5_file_mouse
        # self.dataset_name = dataset_name

        # Open the HDF5 file and check the shape of the dataset
        with h5py.File(hdf5_file_human, 'r') as hdf:
            self.dataset_shape = hdf['sequence'].shape

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.dataset_shape[0]

    def __getitem__(self, idx):
        """Get one sample of data from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Open the HDF5 file and retrieve the data for the given index
        with h5py.File(self.hdf5_file_human, 'r') as hdf:
            sequence = hdf['sequence'][idx]#[196_608:-196_608]
            target = hdf['target'][idx]

        return {
            'sequence': torch.tensor(sequence).float(), 
            'target': torch.tensor(target)
        }


def get_dataloaders(batch_size, sampler_cfg, split_lengths=None, num_workers=0, pin_memory=True):
         
    dataset_train = HDF5Dataset(
        hdf5_file_human="/grand/TFXcan/imlab/data/enformer_training_data/larger_window/train_human.hdf5",
        hdf5_file_mouse=None
    )

    dataset_test = HDF5Dataset(
        hdf5_file_human="/grand/TFXcan/imlab/data/enformer_training_data/larger_window/test_human.hdf5",
        hdf5_file_mouse=None
    )

    if split_lengths is None:
        split_lengths = [32000, 1992]
    
    assert len(split_lengths) in {2, 3} and (split_lengths[0] + split_lengths[1]) <= len(dataset_train)

    total_train_samples = len(dataset_train)
    random_indices = random.sample(range(total_train_samples), split_lengths[0]+split_lengths[1])
    dataset_train = Subset(dataset_train, random_indices)

    dataset_train, dataset_val = random_split(dataset_train, split_lengths[:2])

    if len(split_lengths) == 2 or split_lengths[2] is None:
        pass
    else:
        assert ( 
            isinstance(split_lengths[2], int) and 
            split_lengths[2] < len(dataset_test)
        )        
        total_test_samples = len(dataset_test)
        random_indices = random.sample(range(total_test_samples), split_lengths[2])
        dataset_test = Subset(dataset_test, random_indices)

    if sampler_cfg is not None:
        sampler_train = DistributedSampler(dataset_train, **sampler_cfg)
        sampler_val   = DistributedSampler(dataset_val,   **sampler_cfg)
        sampler_test  = DistributedSampler(dataset_test,  **sampler_cfg)
    else:
        sampler_test = sampler_val = sampler_train = None

    common_kwargs = {
        "batch_size" : batch_size, 
        "num_workers": num_workers, 
        "pin_memory" : pin_memory,
    }

    train_dataloader = DataLoader(dataset_train, sampler=sampler_train , **common_kwargs)
    val_dataloader   = DataLoader(dataset_val,   sampler=sampler_val   , **common_kwargs)
    test_dataloader  = DataLoader(dataset_test,  sampler=sampler_test  , **common_kwargs)

    return train_dataloader, val_dataloader, test_dataloader