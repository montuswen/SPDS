import numpy as np

if __name__ == "__main__":
    path = './data_index.npy'
    vector_positions = np.load(path, allow_pickle=True).tolist()
    print(vector_positions)
