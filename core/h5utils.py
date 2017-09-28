import numpy as np
import h5py
import os

class H5Tensor(object):
    def __init__(self, store_path, maxshape, dtype=np.float32):
        self.store_path = store_path
        self.maxshape = maxshape
        self.h = None
        self.dtype = dtype

        if os.path.isfile(store_path):
            self.h = h5py.File(store_path,mode='r+')
            self._check_shape_compatible(self.data().shape,maxshape)

    def _check_shape_compatible(self, s1, s2):
        if s1[1:] != s2[1:]:
            raise RuntimeError("shape not compatible: ", s1,'!=',s2)

    def data(self):
        return self.h['data'][:]

    def flush(self):
        self.h.flush()

    def append(self, tensor):
        self._check_shape_compatible(tensor.shape,self.maxshape)

        if self.h == None:
            self.h = h5py.File(self.store_path,mode='w')
            self.h.create_dataset('data',shape=tensor.shape, data=tensor, maxshape=self.maxshape, dtype=self.dtype)
            self.h.flush()
        else:
            d = self.h['data']
            newsize = tensor.shape[0] + d.shape[0]
            s = d.shape[0]
            d.resize(newsize,axis=0)
            d[s:] = tensor


def test():
    aa = np.zeros(shape=(2,2,2), dtype=np.int32)
    bb = np.ones(shape=(3,2,2), dtype=np.int32)
    
    t = H5Tensor('tensor1.h5',maxshape=(1000000,2,2),dtype=np.int32)
    
    for i in range(1000):
        t.append(aa)

    print(t.data().shape)

    t.append(bb)
    print(t.data().shape)

    #print(t.data())

    t.flush()

if __name__ == "__main__":
    test()