import dataloader
import models
import trainer
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import json
from pathlib import Path
def main() -> None:
    model_path='models/Qwen2.5-0.5B-Instruct'
    reference_model_path='models/reference_model_SFT_5000'
    dataset_path=''
    domain_config=None
    domain_ouput_dir='data'



    train_domain_weights_dict = domain_config['train_domain_weights']
    # 将字典转换为数组时始终按键排序
    domain_list = list(sorted(train_domain_weights_dict.keys()))

    #缺少tokenize
    ds=dataloader.get_mixed_dataset(
        dataset_dir=dataset_path,
        domain_config=domain_config
    )
    
    reference_model=AutoModelForCausalLM.from_pretrained(reference_model_path)
    tokenizer=AutoTokenizer.from_pretrained(model_path)
    model=models.ProportionGPT2LMHeadModel(reference_model=reference_model)
    

    trainer=trainer.ProportionTrainer(
        model=model,
        train_dataset=ds,
        tokenizer=tokenizer,
        # eval_dataset=ds,
        data_collator=dataloader.get_data_collator(tokenizer,max_length=2048,do_padding=True),
    )
    train_res=trainer.train()
    trainer.save_model()
    
    avg_domain_weights_dict = {}
    for i in range(len(model.avg_domain_weights)):
        domain_name = domain_list[i]
        metrics[f'avg_domain_weight:{domain_name}'] = model.avg_domain_weights[i].item()
        avg_domain_weights_dict[domain_name] = model.avg_domain_weights[i].item()

    # 将平均领域权重保存为 JSON
    avg_domain_weights_file = Path(domain_ouput_dir) / 'avg_domain_weights.json'
    with open(avg_domain_weights_file, 'w') as f:
        json.dump(avg_domain_weights_dict, f, indent=2)

if __name__ == "__main__":

    main()