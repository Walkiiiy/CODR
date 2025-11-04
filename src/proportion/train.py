import dataloader

def main() -> None:
    initial_ds=dataloader.get_mixed_dataset(
        dataset_dir='data/KodCode_splited/BySubset'
        )
    
    pass

if __name__ == "__main__":
    main()