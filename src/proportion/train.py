import dataloader
import models
import trainer

def main() -> None:
    dataset_path=''
    domain_config=None
    #缺少tokenize
    ds=dataloader.get_mixed_dataset(
        dataset_dir=dataset_path,
        domain_config=domain_config
    )
    
    model=models.ProportionGPT2LMHeadModel()

    
    initial_ds=dataloader.get_mixed_dataset(
        dataset_dir='data/KodCode_splited/BySubset'
        )
    
    pass

if __name__ == "__main__":

    main()