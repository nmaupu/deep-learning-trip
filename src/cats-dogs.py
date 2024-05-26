from ml.utils import load_dataset

if __name__ == "__main__":
    X, y = load_dataset("datasets/trainset.hdf5", "X_train", "Y_train")
    print("X train dim=", X.shape)
    print("y train dim=", y.shape)

    X, y = load_dataset("datasets/testset.hdf5", "X_test", "Y_test")
    print("X test dim=", X.shape)
    print("y test dim=", y.shape)
