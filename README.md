# Better Estimation of the KL Divergence Between Language Models
This repository contains code for the Rao--Blackwellized Monte Carlo estimator of the KL divergence between two language models. 

The code is based on the [`trl`](https://github.com/huggingface/trl) library. To run the code, you need to replace the the corresponding files in the `trl` library with the ones in this repository.

# Our Estimator
Our RB estimator is implemented in `trl/trainer/utils.py` in `compute_kl` function, with a few lines of code.

```python
def compute_kl(new_logprobs, ref_logprobs, logits_p=None, logits_q=None):
    if logits_p is not None:
        logp = torch.log_softmax(logits_p, dim=-1)
        logq = torch.log_softmax(logits_q, dim=-1)

        return torch.sum(torch.exp(logp) * (logp - logq), dim=-1)
    return new_logprobs - ref_logprobs
```

Note that if `logits_p` and `logits_q` are `None`, the KL divergence is computed using the MC estimator. Otherwise, the KL divergence is computed using the RB estimator using the full conditional distribution given by the logits.

# Integration with RLOO
We can use this estimator either to evaluate RLHFed models, or to use it in the RL loop. We modify the `rloo` trainer in the `trl` library to use this RB estimator. This includes the modified trainer in `trainer/rloo_trainer.py` and a new config in `trainer/rloo_config.py`, which add `stepwise` to the RLOO trainer config. 

We further provide an example of how to run RLHF with this modified version in `rloo_sentiment.py`. An example command to run the example script is
```bash
python rloo_sentiment.py \
	--init_kl_coef 0.05 \
	--method [MC, RB] \
	--seed 0 \
	--filepath /data
	--filename imdb
```
- `--init_kl_coef` argument specifies the initial KL coefficient 
- `--method` argument specifies the method to use, either `MC` for Monte Carlo or `RB` for Rao--Blackwellized
- `--seed` argument specifies the random seed for reproducibility
- `--filepath` argument specifies the path to the dataset
- `--filename` argument specifies the name of the dataset
