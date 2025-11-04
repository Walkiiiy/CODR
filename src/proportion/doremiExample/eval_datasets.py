"""
下游任务评估数据集。

该模块提供加载与处理评估数据集的工具，用于在少样本场景下评估
如 Trivia QA、Natural Questions 等下游任务。
"""
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Any, Optional, Callable, Union
import string


def substring_until(s: str, split_strs: List[str]) -> str:
    """
    提取直到遇到任一分隔字符串之前的子串。

    参数：
        s: 输入字符串。
        split_strs: 待查找的分隔字符串列表（返回首次匹配前的子串）。

    返回：
        str: s 中第一次出现任一分隔字符串之前的子串。
    """
    idx: int = len(s)
    for split_str in split_strs:
        try:
            new_idx = s.index(split_str)
            if new_idx < idx:
                idx = new_idx
        except Exception:
            pass
    return s[:idx]


def pred_postprocess_default(pred: str) -> str:
    """
    默认的预测后处理函数。

    去除换行符与标点，并转换为小写。

    参数：
        pred: 原始预测字符串。

    返回：
        str: 经过后处理的预测结果。
    """
    pred = pred.strip().lower()
    return substring_until(pred, ['\n']).strip().lower().translate(str.maketrans('', '', string.punctuation))


def eval_func_default(
    answer: Union[str, List[str]],
    pred: str,
    prompt: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    inputs: Optional[Dict[str, Any]] = None,
    trainer: Optional[Any] = None
) -> bool:
    """
    默认的评估函数，用于判断预测是否匹配答案。

    参数：
        answer: 正确答案（字符串或字符串列表）。
        pred: 模型预测的答案字符串。
        prompt: 输入的提示字符串。
        model: 可选的模型实例（未使用）。
        tokenizer: 可选的分词器实例（未使用）。
        inputs: 可选的输入字典（未使用）。
        trainer: 可选的训练器实例（未使用）。

    返回：
        bool: 若预测匹配任一答案则为 True，否则为 False。
    """
    if not isinstance(answer, list):
        answer = [answer.strip().lower().translate(str.maketrans('', '', string.punctuation))]
    else:
        answer = [a.strip().lower().translate(str.maketrans('', '', string.punctuation)) for a in answer]
    return pred in answer


def get_eval_dataset(
    dataset_name: str,
    num_shots: int,
    seed: int = 42
) -> Dict[str, Any]:
    """
    获取用于下游任务的评估数据集配置。

    该函数会加载并配置少样本评估所需的数据集。
    当前支持：trivia_qa、natural_questions、web_questions、lambada、squad_v2。

    参数：
        dataset_name: 评估数据集名称。
        num_shots: 使用的少样本数量。
        seed: 用于打乱数据集的随机种子。

    返回：
        Dict[str, Any]: 包含以下内容的字典：
            - dataset_train: 用于少样本上下文的训练集
            - dataset_val: 验证/测试集
            - top_k, top_p, temperature: 生成超参数
            - num_shots: 少样本数量
            - max_new_tokens: 最多生成的 token 数
            - prompt_transform: 将样本转换为提示的函数
            - eval_func: 评估预测结果的函数
            - pred_postprocess_func: 预测后处理函数
            - shuffle_train: 是否打乱训练样本

    异常：
        ValueError: dataset_name 不受支持时抛出。
    """

    # 默认配置
    top_k = 1
    top_p = 0
    temperature = 1
    num_shots = num_shots
    max_new_tokens = 20
    shuffle_train = True

    eval_func = eval_func_default
    pred_postprocess_func = pred_postprocess_default

    # 加载少样本数据集
    if dataset_name == 'trivia_qa':
        dataset = load_dataset(dataset_name, name='rc.nocontext')
        dataset_train = dataset['train']
        dataset_val = dataset['validation']
        input_key = 'question'
        output_key = 'answer'

        def prompt_transform(ex, context_exs):
            prompt = '\n\n'.join([f"Question: {c_ex[input_key]}\nAnswer: {c_ex[output_key]['aliases'][0]}" for c_ex in context_exs])
            prompt += f"\n\nQuestion: {ex[input_key]}\nAnswer:"

            answer_list = ex[output_key]['aliases']
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'natural_questions':
        dataset = load_dataset("lucadiliello/naturalquestionsshortqa")
        dataset_train = dataset['train']
        dataset_val = dataset['validation']

        def prompt_transform(ex, context_exs):
            prompt = '\n\n'.join([f"Q: {c_ex['question']}?\n\nA: {c_ex['answers'][0]}"
                                  for c_ex in context_exs])
            prompt += f"\n\nQ: {ex['question']}?\n\nA:"

            answer_list = ex['answers']
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'web_questions':
        dataset = load_dataset(dataset_name)
        dataset_train = dataset['train']
        dataset_val = dataset['test']

        def prompt_transform(ex, context_exs):
            prompt = '\n\n'.join([f"Question: {c_ex['question']}\nAnswer: {c_ex['answers'][0]}"
                                  for c_ex in context_exs])
            prompt += f"\n\nQuestion: {ex['question']}\nAnswer:"

            answer_list = ex['answers']
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'lambada':
        dataset = load_dataset(dataset_name)
        dataset_train = dataset['validation']
        dataset_val = dataset['test']

        def prompt_transform(ex, context_exs):
            words = ex['text'].split(' ')
            ex_input = ' '.join(words[:-1])
            ex_answer = words[-1]

            context_ex_toks = [c_ex['text'].split(' ') for c_ex in context_exs]
            prompt = '\n\n'.join([f"Input: {' '.join(c_ex_toks[:-1])}\nOutput: {c_ex_toks[-1]}"
                                  for c_ex_toks in context_ex_toks])
            prompt += f"\n\nInput: {ex_input}\nOutput:"
            prompt = "Complete the following sentences.\n\n" + prompt

            answer_list = [ex_answer]
            return {'prompt': prompt, 'answer': answer_list}

    elif dataset_name == 'squad_v2':
        dataset = load_dataset(dataset_name)
        # dataset_train = dataset['train']
        shuffle_train = False

        dataset_val = dataset['validation']

        # 为每个标题收集索引
        dataset_val_chunks = []
        dataset_train_chunks = []
        all_titles = set([ex['title'] for ex in dataset_val])
        for i, title in enumerate(all_titles):
            title_dataset_val = dataset_val.filter(lambda x: x['title'] == title).shuffle(seed + i)
            title_dataset_train = title_dataset_val.select(list(reversed(range(len(title_dataset_val)))))
            assert(len(title_dataset_train) == len(title_dataset_val))
            dataset_train_chunks.append(title_dataset_train)
            dataset_val_chunks.append(title_dataset_val)

        dataset_train = concatenate_datasets(dataset_train_chunks)
        dataset_val = concatenate_datasets(dataset_val_chunks)

        def prompt_transform(ex, context_exs):
            for c_ex in [ex] + context_exs:
                if len(c_ex['answers']['text']) == 0:
                    c_ex['answers']['text'] = ['unanswerable']

                assert(c_ex['title'] == ex['title'])

            prompt = f"Title: {ex['title']}\n\nBackground: {ex['context']}\n\n"
            prompt += '\n\n'.join([f"Question: {c_ex['question']}\n\nAnswer (use Background or answer \"unanswerable\"): {c_ex['answers']['text'][0]}"])
            prompt += f"\n\nQuestion: {ex['question']}\n\nAnswer (use Background or answer \"unanswerable\"):"

            answer_list = ex['answers']['text']
            return {'prompt': prompt, 'answer': answer_list}

        def eval_func(answer, pred, prompt, model, tokenizer, inputs, trainer):
            if not isinstance(answer, list):
                answer = [answer.strip().lower().translate(str.maketrans('', '', string.punctuation))]
            else:
                answer = [a.strip().lower().translate(str.maketrans('', '', string.punctuation)) for a in answer]
            return pred in answer

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

    return {
        'top_k': top_k,
        'top_p': top_p,
        'temperature': temperature,
        'num_shots': num_shots,
        'max_new_tokens': max_new_tokens,
        'prompt_transform': prompt_transform,
        'dataset_train': dataset_train,
        'shuffle_train': shuffle_train,
        'dataset_val': dataset_val,
        'eval_func': eval_func,
        'pred_postprocess_func': pred_postprocess_func, }
