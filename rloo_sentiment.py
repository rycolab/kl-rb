import argparse
import json
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    set_seed
)

from trl import RLOOConfig, RLOOTrainer

tokenizer = AutoTokenizer.from_pretrained(
    "lvwerra/gpt2-imdb", padding_side="left", trust_remote_code=True
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
ref_policy = AutoModelForCausalLM.from_pretrained(
    "lvwerra/gpt2-imdb", trust_remote_code=True
)
policy = AutoModelForCausalLM.from_pretrained(
    "lvwerra/gpt2-imdb", trust_remote_code=True
)

sentiment_fn = pipeline(
    "sentiment-analysis",
    "lvwerra/distilbert-imdb",
    truncation=True,
    batch_size=256,
    device="cuda",
)


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return scores[1]["score"]


def reward_fn_normal(samples, **kwargs):
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none"}
    sentiments = list(map(get_positive_score, sentiment_fn(samples, **sent_kwargs)))
    return sentiments


def load_prompts(filepath, filename, split="train"):
    prompts = []
    with open(f"{filepath}/{filename}-{split}.jsonl", "r") as f:
        for line in f:
            line_dict = json.loads(line)
            prompts.append(line_dict["prompt"])
    return list(set(prompts))


class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer=None, max_length=512):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        text = self.prompts[idx]

        if self.tokenizer:
            # Tokenize the text (for NLP tasks)
            encoded = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            return {key: val.squeeze(0) for key, val in encoded.items()}

        return text  # If no tokenizer, return raw text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_kl_coef", type=float, default=0.05, help="KL coefficient")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--method", type=str, default="MC", help="KL estimation method")
    parser.add_argument("--filename", type=str, default="imdb",
                        help="name of the prompt file")
    parser.add_argument("--filepath", type=str, default="/data", help="data path")
    args = parser.parse_args()

    stepwise = False

    if args.method == "RB":
        stepwise = True

    set_seed(args.seed)

    train_prompts = load_prompts(args.filepath, args.filename, split="train")
    test_prompts = load_prompts(args.filepath, args.filename, split="test")

    train_dataset = PromptDataset(train_prompts, tokenizer=tokenizer)
    eval_dataset = PromptDataset(test_prompts, tokenizer=tokenizer)

    config = RLOOConfig(learning_rate=3e-5, per_device_train_batch_size=16,
                        gradient_accumulation_steps=2, num_train_epochs=1,
                        kl_coef=args.init_kl_coef, stop_token='eos', temperature=1.,
                        stepwise=stepwise)
    trainer = RLOOTrainer(
        config=config,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_fn_normal,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()