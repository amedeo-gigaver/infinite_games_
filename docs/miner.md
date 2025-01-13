
# Miner

You will be receiving data in the form defined in `infinite_games/events/base.py` and will only need to complete the `probability` entry. 

The current scoring mechanism is described [here](https://hackmd.io/@nielsma/S1sB8xO_C). 

You will be receiving requests from validators every minute. You do not have to recalculate your prediction every time. Our baseline miners are implementing a caching system that allows for a recalculation schedule. This is particularly important if you use a base model like GPT4 that might be expensive.

## Baseline Miner

We are constantly working on improving the baseline miners, especially with regards to new categories of synthetic events being added to the subnet.

We integrate the LLM prompting [pipeline](https://github.com/dannyallover/llm_forecasting.git) with news retrieval developed by the authors of the forecasting LLM paper quoted in the readme. In the current form it only uses [Google News](https://news.google.com/home?hl=en-US&gl=US&ceid=US:en) for news retrieval while the original model from the article used 4 other different sources (Newscatcher, Newsdata.io, Aylien, NewsAPI.org).
This pipeline can work with different base models.

**version 1**
GPT4 as base model. 

steps: 
- add an OpenAI key (OPENAI_KEY) to your local environment

**version 2**
GPT-3.5 and GPT4 Mini as base models.

steps:
- add an OpenAI key to your local environment
- add the parameter 1 to the `get_prediction` function in `miner.py`

```llm_prediction = (await self.llm.get_prediction(market, 1))```

**version 3**

Gemini as base model

steps:
- add a Google Gemini key (GOOGLE_AI_KEY) to your local environment
- add the parameter 2 to the `get_prediction` function in `miner.py`

```llm_prediction = (await self.llm.get_prediction(market, 2))```

**version 4**
GPT-3.5 (used for reasoning) and Gemini as base models

steps:
- add an OpenAI key and a Google Gemini key (GOOGLE_AI_KEY) to your local environment
- add the parameter 3 to the `get_prediction` function in `miner.py`

```llm_prediction = (await self.llm.get_prediction(market, 3))```

You can also set up your own configurations.


## Sample data for generated events from ACLED

We join [here](docs/ACLED-data/protest-data-3days.json) and [here](docs/ACLED-data/Protests-events-7-days.json) two dataset of events that stretches over 6 months (from January 2024 to June 2024) which represents the type of events that will be sent to miners from [ACLED](https://acleddata.com/) data.

This first set of events will have the following structure:
"Will the amount of protests in [country] during [3 days window or 1 week window] be above [30 day moving average] ?"

**Key details**:
- the countries will initially be EU countries (with non degenerate protest rates) and the US. This is the exact list: ['Germany', 'United Kingdom', 'France', 'Italy', 'Spain', 'Poland', 'Romania', 'Netherlands', 'Belgium', 'Ireland', 'Sweden', 'Czech Republic', 'Greece', 'Portugal', 'Hungary', 'Austria', 'Serbia', 'Bulgaria', 'Denmark', 'United States']
- the cutoff will be set at the start of the time window
- events will be generated daily but they will only resolve on Tuesday-Wednesday due to the schedule ACLED follows to update its data.

Miners will have to retrieve protest data relevant to each of the countries listed above in order to improve their predictions. 

# Miner strategy 

A reference providing a **baseline miner** strategy is the article ["Approaching Human Level Forecasting with Language Models"](https://arxiv.org/html/2402.18563v1?s=35) ([1]). The authors fine-tune an LLM to generate predictions on binary events (including the ones listed on Polymarket) which nears the performance of human forecasters when submitting a forecast for each prediction, and which beats human forecasters in a setting where the LLM can choose to give a prediction or not based on its confidence.

The authors released a version of the code they used to build the LLM they describe in the paper. You can find it [here](https://github.com/dannyallover/llm_forecasting.git).


# System requirements

The computational requirements for a miner will depend significantly on the frequency at which they need to send predictions to be competitive since the base model we consider consists of a set of LLM sub-modules that each need to perform computations. Miners will also need to eventually continually provide new data to their models.

A significant cost in the case of the repo referenced above is the cost of using e.g the OpenAI API as well as the cost of retrieving news data. Every time a prediction is produced, a base model is used to generate various prompts. A miner could circumvent that by using a local model or by using the output of the [subnet 18](https://github.com/corcel-api/cortex.t.git). 
News retrieval other than Google News is done through news provider like https://www.newscatcherapi.com/. They are most often behind a paywall.


# Getting Started



First clone this repo by running `git clone `. Then to run a miner:


Clone repository

```bash
git clone https://github.com/amedeo-gigaver/infinite_games.git
```

Change directory

```bash
cd infinite_games
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate a Virtual Environment

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```
Enable `torch`:

```bash
export USE_TORCH=1
```

The venv should be active whenever the neurons are run.

## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your miner.

The miner will be registered to the subnet specified. This ensures that the miner can run the respective miner scripts.

Create a coldkey and hotkey for your miner wallet.

```bash
btcli wallet new_coldkey --wallet.name miner
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default
```

## 2a. (Optional) Getting faucet tokens

Faucet is disabled on the testnet. Hence, if you don't have sufficient faucet tokens, ask the Bittensor Discord community for faucet tokens. Go to the help-forum on the Discord and post your request in the "Requests for Testnet TAO" thread.

## 3. Register keys

This step registers your subnet miner keys to the subnet.

```bash
btcli subnet register --wallet.name miner --wallet.hotkey default
```

To register your miner on the testnet add the `--subtensor.network test` flag.

Follow the below prompts:

```bash
>> Enter netuid (0): # Enter the appropriate netuid for your environment
Your balance is: # Your wallet balance will be shown
The cost to register by recycle is τ0.000000001 # Current registration costs
>> Do you want to continue? [y/n] (n): # Enter y to continue
>> Enter password to unlock key: # Enter your wallet password
>> Recycle τ0.000000001 to register on subnet:8? [y/n]: # Enter y to register
📡 Checking Balance...
Balance:
  τ5.000000000 ➡ τ4.999999999
✅ Registered
```

## 4. Check that your keys have been registered

This step returns information about your registered keys.

Check that your miner key has been registered:

```bash
btcli wallet overview --wallet.name miner
```

To check your miner on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: TBD # or 155 on testnet
COLDKEY    HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
miner  default  197    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                56  none  5GKkQKmDLfsKaumnkD479RBoD5CsbN2yRbMpY88J8YeC5DT4
1          1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                                Wallet balance: τ0.000999999
```

## Running a Miner

### Direct Run 

Run the following command inside the `infinite_games` directory:

`python neurons/miner.py --netuid 155 --subtensor.network test --wallet.name miner --wallet.hotkey default`

### PM2 Installation

Install and run pm2 commands to keep your miner online at all times.


`sudo apt update`

`sudo apt install npm` 

`sudo npm install pm2 -g`

`Confirm pm2 is installed and running correctly`

`pm2 ls`

Use the following example command to run the miner:

`pm2 start neurons/miner.py --interpreter python3  --name miner -- --wallet.name miner --netuid 155 --wallet.hotkey default --subtensor.network test --logging.debug`


#### Variables Explanation

--wallet.name: Provide the name of your wallet.
--wallet.hotkey: Enter your wallet's hotkey.
--netuid: Use 155 for testnet.
--subtensor.network: Specify the network you want to use (finney, test, local, etc).
--logging.debug: Adjust the logging level according to your preference.
--axon.port: Specify the port number you want to use.


Use the following commands to monitor the status and logs:

`pm2 status`
`pm2 logs 0`

### Useful PM2 Commands

The following commands will be useful for the management of your miner:

`pm2 list` # lists all pm2 processes

`pm2 logs <pid>` # replace pid with your process ID to view logs

`pm2 restart <pid>` # restart this pic

`pm2 reload <pid>` # reload this pic

`pm2 stop <pid>` # stops your pid

`pm2 del <pid>` # deletes your pid

`pm2 describe <pid>` # prints out metadata on the process

## 5. Stopping your miner

To stop your miner, press CTRL + C in the terminal where the miner is running.

