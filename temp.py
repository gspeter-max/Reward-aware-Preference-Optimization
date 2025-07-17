import numpy as np
from torch.utils.data import DataLoader

temp_dataset = np.random.randn( 23, 3 )

class create_dataset(DataLoader):
    def __init__(self):
        super().__init__()
        self.dataset = np.random.randn( 12,4 ) 

    def __getitems__( self, idx ):
        return self.dataset[idx] 

    def __len__( self ):
        return len(self.dataset)

data = create_dataset( temp_dataset ) 
print(type(data)) 



