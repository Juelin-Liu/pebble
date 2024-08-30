from util import Dataset, Config, load_dataset, get_args

if __name__ == "__main__":
    config = get_args()
    data = load_dataset(config)
    print(data)