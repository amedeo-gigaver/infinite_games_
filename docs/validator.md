

# Validator

The validator is streaming binary outcomes events to miners. The events can come directly from external APIs or be LLM generated. Every time an event settles, the validator will score the miners which submitted a prediction for that given event.

The main computational prerequisite is related to storage. Validators are now storing the entire time-series of a miner's predictions for a given event, cf. the [scoring doc](https://hackmd.io/@nielsma/S1sB8xO_C).

It is essential that validators do not wipe out their local database and persist the data. We automatically implement a discard functionality for data which is more than 1 month old.

There are two auto-update scripts in the repo: [auto_update](/auto_update.sh) and [update_script](/update_script.py).

**IMPORTANT**

Before attempting to register on mainnet, we strongly recommend that you run a validator on the testnet. For that matter ensure you add the appropriate testnet flags.

`--netuid 155 --subtensor.network test`

| Environment | Netuid |
| ----------- | -----: |
| Mainnet     |      6 |
| Testnet     |    155 |

**DANGER**

- Do not expose your private keys.
- Only use your testnet wallet.
- Do not reuse the password of your mainnet wallet.

<br />

# System Requirements

- Requires at least **Python 3.10.**
- Requires at least **SQLite 3.37.**

Below are the computational prerequisites for validators.

- Validators are recommended to have at least 16GB of RAM and 2CPUs. We expect daily requirements to be lower but this would provide sufficient margin.

As we optimize the fetching and processing of events these requirements may evolve.

# Getting Started


First clone this repo by running `git clone `. Then to run a validator:


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

The venv should be active whenever the neurons are run.


## 2. Create Wallets

This step creates local coldkey and hotkey pairs for your validator.

The validator will be registered to the subnet specified. This ensures that the validator can run the respective validator scripts.

Create a coldkey and hotkey for your validator wallet.

```bash
btcli wallet new_coldkey --wallet.name validator
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

## 2a. (Optional) Getting faucet tokens

Faucet is disabled on the testnet. Hence, if you don't have sufficient faucet tokens, ask the Bittensor Discord community for faucet tokens. Go to the help-forum on the Discord and post your request in the "Requests for Testnet TAO" thread.

## 3. Register keys

This step registers your subnet validator keys to the subnet.

```bash
btcli subnet register --wallet.name validator --wallet.hotkey default
```

To register your validator on the testnet add the `--subtensor.network test` flag.

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

Check that your validator key has been registered:

```bash
btcli wallet overview --wallet.name validator
```

To check your validator on the testnet add the `--subtensor.network test` flag

The above command will display the below:

```bash
Subnet: 6 # or 155 on testnet
COLDKEY    HOTKEY   UID  ACTIVE  STAKE(τ)     RANK    TRUST  CONSENSUS  INCENTIVE  DIVIDENDS  EMISSION(ρ)   VTRUST  VPERMIT  UPDATED  AXON  HOTKEY_SS58
validator  default  197    True   0.00000  0.00000  0.00000    0.00000    0.00000    0.00000            0  0.00000                56  none  5GKkQKmDLfsKaumnkD479RBoD5CsbN2yRbMpY88J8YeC5DT4
1          1        1            τ0.00000  0.00000  0.00000    0.00000    0.00000    0.00000           ρ0  0.00000
                                                                                Wallet balance: τ0.000999999
```

## 6. Running a Validator

### Command Options

The following options are available to set the environment on start.
<br />
If wrong options / combinations are provided, the validator will error and a message displayed.

`--netuid`

Provide the subnet UID.
<br/>
Possible values:

- **6** for mainnet
- **155** for testnet

`--subtensor.network`

Provide the subtensor network to connect to.
<br/>
Possible values:

- **finney** for mainnet
- **test** for testnet
- **local** for local node

`--ifgames.env`

Provide the InfiniteGames environment.
**Only required** when subtensor network is **local**.
<br/>
Possible values:

- **prod** when running on mainnet
- **test** when running on testnet

`--wallet.name`

Provide the name of your wallet.

`--wallet.hotkey`

Provide the wallet's hotkey.

### Direct Run

Run the following command inside the `infinite_games` directory:

```bash
python neurons/validator.py --netuid 155 --subtensor.network test --wallet.name validator --wallet.hotkey default
```

Example for local subtensor network

```bash
python neurons/validator.py --netuid 155 --subtensor.network local --ifgames.env test --wallet.name validator --wallet.hotkey default
```

### PM2 Installation

Install and run pm2 commands to keep your validator online at all times.

```bash
sudo apt update

sudo apt install npm

sudo npm install pm2 -g
```

Confirm pm2 is installed and running correctly

```bash
pm2 ls
```

Command to run the validator:

```bash
pm2 start neurons/validator.py --interpreter python3  --name validator -- --netuid 155 --subtensor.network test --wallet.name validator  --wallet.hotkey hotkey
```

You can monitor the status and logs using these commands:

```bash
pm2 status

pm2 logs 0
```

Useful PM2 Commands

The following Commands will be useful for management:

```bash
pm2 list # lists all pm2 processes

pm2 logs <pid> # replace pid with your process ID to view logs

pm2 restart <pid> # restart this pic

pm2 stop <pid> # stops your pid

pm2 del <pid> # deletes your pid

pm2 describe <pid> # prints out metadata on the process
```

## 7. Get emissions flowing

Register to the root network using the `btcli`:

```bash
btcli root register
```

To register your validator to the root network on testnet use the `--subtensor.network test` flag.

Then set your weights for the subnet:

```bash
btcli root weights
```

To set your weights on testnet `--subtensor.network test` flag.

## 8. Stopping your validator

To stop your validator, press CTRL + C in the terminal where the validator is running.

