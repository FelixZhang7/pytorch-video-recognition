import mc
from torch.utils.data import Dataset
import linklink as link


class McDataset(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.rank = link.get_rank()
        self.meta_file = meta_file
        if self.rank == 0:
            print("building dataset from %s" % self.meta_file)

        self.initialized = False

    def __len__(self):
        return None

    def _parse_list(self):
        """This method should be implemented in child class.
        """
        raise NotImplementedError

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            self.initialized = True

        return

    def __getitem__(self, idx):
        """This method should be implemented in child class.
        """
        raise NotImplementedError
