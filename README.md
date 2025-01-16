<div align="center">

<img src="docs/infinite-games.jpeg" alt="Project Logo" width="200"/>

# **Infinite Games** 


[Discord](https://discord.gg/BWdj7SGQ) • [Dashboard](https://app.hex.tech/1644b22a-abe5-4113-9d5f-3ad05e4a8de7/app/5f1e0e62-6072-4440-9646-6d2b60cd1674/latest) •
[Website](https://www.infinitegam.es/) • [Twitter](https://twitter.com/Playinfgames) •  [Network](https://taostats.io/) 

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

</div>



<!-- update with discord invite link -->

</div>

  <!--
  <a href="">Twitter</a> -->
    
  <!-- <a href="">Bittensor</a> -->


# Forecasting of Future Events

##  Introduction


We incentivize the prediction of future events. The prediction space is based on binary future events such as the ones listed on [Polymarket](https://polymarket.com/). We are always actively expanding to new data providers. Most of this data is then processed by an LLM pipeline which handles the event generation. We are focused on *judgemental forecasting* rather than *statistical forecasting*. We hence expect the models used by miners to be LLMs. 

### High-level mechanism

Miners submit their predictions to validators. Each prediction has to be done early enough before the event underlying the prediction settles. Once the event settles, the validators that received the prediction score the miner.

## LLMs open new predictive capabilities

Making predictions is a hard task that requires cross-domain knowledge and intuition. It is often limited in explanatory reasoning and domain-specific (the expert in predicting election results will differ from the one predicting the progress in rocket-engine technology) ([1]). At the same time it is fundamental to human society, from geopolitics to economics. 

<!-- The COVID-19 measures for example were based on epidemiological forecasts. Science is another area where prediction is crucial ([2]) to determine new designs or to predict the outcome of experiments (executing one experiment is costly). Such predictions rely on the knowledge of thousands papers and on multidisciplinary and multidimensional analysis (*can a study replicate ? should one use a molecular or behavioral approach?*). -->

LLMs approach or surpass human forecasting abilities. They near on average the crowd prediction on prediction market events ([1]), and surpass humans in predicting neuroscience results ([2]). They are also shown to be calibrated with their predictions i.e confident when right. Through their generalization capabilities and unbounded information processing, LLMs have the potential to automate the prediction process or complement humans. 


### Real-world applications

The value of the subnet first relies in the improvement of the efficiency of prediction markets. This value can be extracted by validators through arbitrage. The validators may obtain a better knowledge of the probability of an event settling and communicate this information to a prediction market by opening a position. 

The first applications built on top of our subnet could be related to prediction markets. A trader could query our market to obtain the most up to date and relevant predictions to their portfolio based on the current news landscape (LLMs would be constantly aggregating the most up to date and relevant news articles). They could then readjust their positions accordingly or trade directly on this information. 

In the long term, a validator could provide paid economic forecasts or more generally the output of any forward-looking task addressed to an LLM ([2]). A customer might then provide a series of paid sub-queries related to the information they aim at retrieving.

<!-- It could also be used by scientists to design their experiment and frame their ideas. For example, the value of a paper often resides in the way the results are presented and cross-analysed. One way resulting in poor conclusions while the other giving good results. An LLM might help detect the adequate framework. -->


## Miners 

Miners compete by sending to the validators for each binary event $E$ their estimation of the probability $p$ that $E$ is realized. For example, $E$ could be *o3 is released to the public by January 15th 2025*.


### Miner strategy 

A reference providing a **baseline miner** strategy is the article ["Approaching Human Level Forecasting with Langage Models"](https://arxiv.org/html/2402.18563v1?s=35) ([1]). The authors fine-tune an LLM to generate predictions on binary events (including the ones listed on Polymarket) which nears the performance of human forecasters when submitting a forecast for each prediction, and which beats human forecasters in a setting where the LLM can choose to give a prediction or not based on its confidence.

According to the article, the performance of forecasting LLMs depends significantly on the amount of data one can retrieve for a given prediction or event (for example prediction market data). If our subnet is able to continually produce new synthetic data, miners should be able to beat the SoA.


## Validators

Validators record the miners' predictions and score them once the events settle. At each event settlement, a score is added to the moving average of the miner's score. This simple model ensures that all validators score the miners at roughly the same time. We implement a **cutoff** for the submission time of a prediction. The cutoff is set at 24 hours before the resolution date for most events.

## Scoring rule

Denote by $S(p_i, o_i)$ the Brier score of a prediction $p_i$ on the binary event $E_i$ and where $o_i$ is $1$ if $E_i$ is realized and $0$ otherwise. We have that $S(p_i, 1)= 1- (1-p_i)^2$ if $o_i$ is $1$ and $S(p_i,0)=1-p_i^2$ if $o_i$ is $0$. The Brier score is strictly proper i.e it strictly incentivizes miners to report their true prediction. 

The validator stores **the time series of the miner's predictions** and computes the Brier score of each element of the time series. It hence obtains a new time series of Brier scores. A number $n$ of intervals is set between the issues date and the resolution date. The validator then computes a **weighted average of the Brier scores**, where the weight is exponentially decreasing with time, in interval $k$ it has value $exp(-\frac{n}{k} +1)$ where $k$ starts at $n$ and decreases to $1$.

The final score is a linear combination of the weighted average and of a linear component that depends on how good is the miner compared to other miners.

This is described in details [here](https://hackmd.io/@nielsma/S1sB8xO_C). We give miners a score of $0$ on the events for which they did not submit a prediction.


<!--
### model 3
We implement a **sequentially shared quadratic scoring rule**. This allows us crucially to aggregate information as well as to score $0$ miners that do not bring new information to the market.
The scoring rule functions by scoring each miner relatively to the previous one. The score of the miner $j$ is then $S_j = S(p_j, o_i) - S(p_{j-1}, o_i)$ where $p_{j-1}$ is the submission of the previous miner. Importantly this payoff can be negative, therefore in practice when aggregating the scores of a miner we add a $\max(-,0)$ operation. 

The aggregated score of a miner that a validator sends to the blockchain is the following:

$$\frac{1}{N} \sum_j S_j$$

where $N$ is the number of events that the validator registered as settled during the tempo.

A simpler version of this model is, instead of paying the miner for their delta to the previous prediction, pay them for their delta to the Polymarket probability at the submission time i.e $S(p_j, o_i) - S(\text{price on polymarket at t}, o_i)$ where $p_j$ is submitted at $t$.
-->

<!--
## Incentive compability

See [here](docs/mechanism.md) for a discussion of our mechanism. -->

## Roadmap


- [x] Scoring with exponentially decreasing weights until settlement date and linear differentiation mechanism 
- [x] Synthetic event generation with central resolution using ACLED data 
- [x] Scoring with exponential differentiation mechanism 
- [x] Comprehensive and granular analytics 
- [x] Synthetic event generation with UMA resolution - human verifiers resolve our events through the OOv2 
- [x] Synthetic event generation from news data using an LLM 

- [ ] Validator v2 - modular and much higher throughput 
- [ ] Scoring v2 (batches, peer score)
- [ ] Exposing the silicon crowd predictions 
- [ ] Decentralisation of event generation and validator dynamic desirability (inspired from SN13)

- [ ] Trustless event resolution using UMA - leveraging the data asserter framework
- [ ] Advanced aggregation mechanism based on sequential scoring
- [ ] Commit-reveal on the miners' predictions
- [ ] Scoring a reasoning component
- [ ] Data generation for iterative fine-tuning of prediction focused LLMs


<!-- We first aim at adjusting the scoring rule by updating to a variation of the *model 2* described above. We will likely implement several other updates in order to make the mechanism more robust. One of them could be a commit-reveal step for the predictions submitted by miners. Some updates may be due to experimental data.

We would also possibly like to make the prediction framework more LLM specific and create mechanisms that explicitely generate data for the fine-tuning of prediction focused LLMs.

We plan to extend the set of predicted events to other prediction markets and event providers (Manifold, Metacalculus, Metacalculus). Our main goal is to obtain a continuous feed of organic events by using e.g WSJ headlines or Reuters' API. -->

<!-- In the limit, miners could update the weights of an entire LLM.-->

## Running a miner or validator

Regarding instructions and requirements, see [here](docs/validator.md) for validators and [here](docs/miner.md) for miners.

## Setting up a bittensor wallet
A detailed explanation of how to set up a wallet can be found [here](https://docs.bittensor.com/getting-started/wallets). 
We also provide some indications [here](docs/wallet-setup.md).


## References

| Reference ID | Author(s) | Year | Title |
|--------------|-----------|------|-------|
| 1 | Halawi and al. | 2024| [Approaching Human Level Forecasting with Langage Models](https://arxiv.org/html/2402.18563v1?s=35)|
| 2 | Luo and al. | 2024 | [LLM surpass human experts in predicting neuroscience results](https://arxiv.org/pdf/2403.03230.pdf)|


---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
