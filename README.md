# gpu-rl : early experiments with training LLMs with RL

## Experiment 1 : GRPO on Qwen2.5-1.5B-Instruct with GSM8K dataset

The most classic experiment one can do to get its hands into the subject.
You can reproduce this experiment using only one A/H100 GPU by running `./train_one_gpu.sh`. 
Depending on the batch size, it should take a few hours.

These are the different rewards during training :
<p align="center">
<img src="assets/rewards.png" alt="rewards" width="1200"/>
</p>

The total reward is the sum of these three rewards.
Accuracy reward goes from ≃1.1 to ≃1.5. Correct answer is 2, incorrect answer is 0, so accuracy actually goes from ≃55% to ≃75%.
For the two format rewards, it's either 0.5 or 0, so we see that the model learns to adapt almost perfectly the format of its response, after around 100 steps of RL.

*Note*: the accuracy reward, as its name suggests, tells if the model got the answer right. I quite changed its definition compared to what you can find in other scripts. For example, [here](https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb), the accuracy reward already assumes that the model correctly follows the output format requested. But this formulation make the accuracy reward and the reasoning format reward correlated. With this formulation, if the accuracy reward goes up, we don't know if it's because the model is actually better at GSM8K or is simply picking up the output format requested. That's why I changed it and choose to take the last int in the model's answer (just like [here](https://arxiv.org/abs/2402.10200)), hence independent of the formatting.

Here is the completion length through training :

<p align="center">
    <img src="assets/seqlen.png" alt="seqlen" width="500"/>
</p>
