import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from collections import Counter
from collections import defaultdict
import math

class DataMonitor:
    def __init__(self):
        self.data = {}

    def __str__(self):
        display_str = "\n\nDataMonitor Contents:\n"
        for key, value in self.data.items():
            display_str += f"Key: {key},\n Value: {value}\n\n"
        return display_str


class CIFAR10Loader:
    def __init__(self, g_para):
        self.dataset_dir = g_para.path[g_para.data_name]
        self.ensure_directory(self.dataset_dir)
        #self.download_cifar10()

    def ensure_directory(self, path):
        os.makedirs(path, exist_ok=True)

    def download_cifar10(self):
        datasets.CIFAR10(self.dataset_dir, download=True, train=True)
        datasets.CIFAR10(self.dataset_dir, download=True, train=False)

    @staticmethod
    def create_transforms(g_para, image_size=32):
        """
        Create data transformations.
        """
        if g_para.g_iter_index == 0: print(f"Loading the {image_size} data_type {g_para.data_name} begins")
        mean, std = [0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]
        base_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(), 
            transforms.Normalize(mean, std),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  # Corrected typo here
        ]
        if image_size == 32:
            return transforms.Compose([transforms.RandomCrop(32, padding=4)] + base_transforms)
        elif image_size == 224:
            return transforms.Compose([transforms.Resize((224, 224)), transforms.RandomCrop(224)] + base_transforms)
        else:
            raise ValueError("Unsupported image size for transformations")


    # To see whether the directory has images
    def has_images(self, directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    return True
        return False

    def load_client_data(self, g_para, monitor, image_size=32):
        """
        Load the CIFAR10 dataset and split it into training, validation, and testing sets.
        """
        transformations = self.create_transforms(g_para, image_size=image_size)
        data_directory = g_para.data_dir[:2] if g_para.dataset['iid_data'] > 0 and len(g_para.data_dir) >= 3 else g_para.data_dir[:1]
        dataloaders = {}

        for data_type in ['train', 'valid', 'test']:
            image_datasets = {}
            datasets_list = []
            for directory in reversed(data_directory):
                client_name = os.path.basename(directory.rstrip('/'))
                dir_path = os.path.join(directory, data_type)
                if not os.path.exists(dir_path) or not self.has_images(dir_path):
                    print(f"{dir_path} does not exit or include no images")

                # Load CIFAR10 dataset
                full_dataset = datasets.ImageFolder(dir_path, transform=transformations)
                indices_per_class = defaultdict(list)

                # Gather indices for each class
                for idx, (_, class_index) in enumerate(full_dataset):
                    indices_per_class[class_index].append(idx)
               
                if client_name=="shared_data" and g_para.dataset['iid_data'] > 0:
                    entries_per_class_indices = [indices[0:g_para.dataset['iid_data']] for indices in indices_per_class.values()]
                    image_datasets[data_type] = Subset(full_dataset, entries_per_class_indices)
                    # Calculate number of entries per class for shared_data
                    num_per_class = {class_index: 1 for class_index in indices_per_class}  # Assuming one entry per class

                else:
                    entries_per_class_indices = []
                    num_per_class = {}
                
                    for class_index, class_indices in indices_per_class.items():
                        # If full_size is true, take all available images from the class
                        entries_per_class_indices.extend(class_indices)
                        num_per_class[class_index] = len(class_indices)

                    image_datasets[data_type] = Subset(full_dataset, entries_per_class_indices)


                    # Extract and save the key values from the datasets
                if g_para.Debug["data"]:
                    monitor.data[client_name+'-'+data_type] = {
                        "dataset_sizes": len(image_datasets[data_type]),
                        "class_names": tuple(full_dataset.classes),
                        "num_entries_per_class": num_per_class
                    }

                datasets_list.append(image_datasets[data_type])

            combined_dataset = ConcatDataset(datasets_list)

            # Extract and save the key values from the datasets
            class_names = set(label for _, label in combined_dataset)
            combin = 'Combined-'+client_name+'-'+data_type
            monitor.data[combin] = {
                "dataset_sizes": len(combined_dataset),
                "class_names": tuple(class_names)
            }

            dataloaders[data_type] = DataLoader(image_datasets[data_type], batch_size=g_para.h_param["batch_size"], shuffle=True, drop_last=False)
            
        if g_para.g_iter_index==0 and g_para.Debug["data"]: print("{}".format(monitor))
        return dataloaders


    def load_data_global(self, g_para, monitor, image_size=32):
        #self.prepare_dataset()
        transformations = self.create_transforms(g_para, image_size=image_size)
        dataloaders = {}

        # Determine the test data directory based on the parameters
        data_directory = g_para.data_dir[2:] if g_para.dataset['iid_data'] > 0 and len(g_para.data_dir) >= 3 else g_para.data_dir[1:]

        print(f"data_directory:({data_directory})")

        # Function to process a dataset directory
        def process_directory(directory, dataset_label, limit_images=None):
            dir_path = os.path.join(directory, 'test')			
            if not os.path.exists(dir_path) or not self.has_images(dir_path):
                print(f"{dir_path} does not exit or include no images")

            # Load dataset
            full_dataset = datasets.ImageFolder(dir_path, transform=transformations)
            indices_per_class = defaultdict(list)

            # Gather indices for each class
            for idx, (_, class_index) in enumerate(full_dataset):
                indices_per_class[class_index].append(idx)

            # Select indices based on limit
            selected_indices = []
            num_per_class = {}
            for class_index, class_indices in indices_per_class.items():
                if limit_images is not None:
                    selected_indices.extend(class_indices[:limit_images])
                    num_per_class[class_index] = min(limit_images, len(class_indices))
                else:
                    selected_indices.extend(class_indices)
                    num_per_class[class_index] = len(class_indices)

            # Create and store the dataset and dataloader
            image_dataset = Subset(full_dataset, selected_indices)
            dataloaders[dataset_label] = DataLoader(image_dataset, batch_size=g_para.h_param["batch_size"], shuffle=True, drop_last=False)
           
            # Store dataset information
            monitor.data[dataset_label] = {
                "dataset_sizes": len(image_dataset),
                "class_names": tuple(full_dataset.classes),
                "num_entries_per_class": num_per_class
            }

        # Process directories
        for directory in data_directory:
            process_directory(directory, 'test_global')  # Full dataset
            process_directory(directory, 'test_global_small', limit_images=30)  # Limited dataset

        if g_para.g_iter_index == 0 and g_para.Debug["data"]: print("{}".format(monitor))
        return dataloaders


    def load_data_central(self, g_para, monitor, image_size):
        data_directory = g_para.data_dir[0]
        transformations = self.create_transforms(g_para, image_size=image_size)
        dataloaders = {}        
        for data_type in ['train', 'valid', 'test']:
            dir_path = os.path.join(data_directory, data_type)
            client_name = os.path.basename(data_directory.rstrip('/'))
            if not os.path.exists(dir_path) or not self.has_images(dir_path):
                print(f"{dir_path} does not exit or include no images")

            full_dataset = datasets.ImageFolder(dir_path, transform=transformations)
            indices_per_class = defaultdict(list)

            # Counting images per class
            for idx, (_, class_index) in enumerate(full_dataset):
                indices_per_class[class_index].append(idx)

            # Extract and save the key values from the datasets
            if g_para.Debug["data"]:
                dataset_sizes = len(full_dataset)
                class_names = set(label for _, label in full_dataset)
                monitor.data[client_name+'-'+data_type] = {
                    "dataset_sizes": dataset_sizes,
                    "class_names": tuple(class_names)
                }

            dataloaders[data_type] = DataLoader(full_dataset, batch_size=g_para.h_param["batch_size"], shuffle=True, drop_last=False)

        if g_para.g_iter_index==0 and g_para.Debug["data"]: print("{}".format(monitor))
        return dataloaders


def load_cifar10(g_para):
    cifar10_loader = CIFAR10Loader(g_para)
    monitor     = DataMonitor()

    if g_para.data_name=="cifar10":
        return cifar10_loader.load_client_data(g_para, monitor, 32)
    elif g_para.data_name=="cifar10_big":
        return cifar10_loader.load_client_data(g_para, monitor, 224)


def load_cifar10_global(g_para):
    cifar10_loader = CIFAR10Loader(g_para)
    monitor     = DataMonitor()

    if g_para.data_name=="cifar10":
        return cifar10_loader.load_data_global(g_para, monitor, 32)
    elif g_para.data_name=="cifar10_big":
        return cifar10_loader.load_data_global(g_para, monitor, 224)
        

def load_cifar10_central(g_para):
    cifar10_loader = CIFAR10Loader(g_para)
    monitor     = DataMonitor()

    if g_para.data_name=="cifar10":
        return cifar10_loader.load_data_central(g_para, monitor, 32)
    elif g_para.data_name=="cifar10_big":
        return cifar10_loader.load_data_central(g_para, monitor, 224)