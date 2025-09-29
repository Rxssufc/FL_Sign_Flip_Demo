# Federated Audio Learning with Flower

This repository contains a Flower-based federated learning setup for training Wav2Vec2 on audio data, including an example adversarial client attack (sign flip attack on FedAvg).

---

##  Data

- The `data/` folder is **not included** here.  
- Download the prepared `data` folder from Google Drive (link provided by supervisor) and place it at the project root so the structure is:
- data/Client1
- data/client_1.csv
- follow this for x amount of clients

## CSV Format

- filename,text
- sample_0001.wav,HELLO
- sample_0002.wav,WORLD

Each sample client has 1000 samples from Mozilla Common Voice

To add more clients, download a new Common Voice subset, split into client-specific CSVs with the same format, and place them in data/.

If results are to be published please reference this dataset.

## Default Settings

- Adversarial Attack Strength: 100.0
- Adversarial Clients: [1] - set in client.py
- Num Client: 10
- Num Rounds: 5

## Changing Settings

- Edit inversion_strength or ADVERSARIAL_CLIENTS in client.py.

- Adjust NUM_CLIENTS and NUM_ROUNDS in simulate.py (simulation mode).

- Or adjust ServerConfig in server.py if running server/clients manually.

## Running the Code
### Method 1
- Run python simulate.py
- Server and clients are ran together from a central point.
- adjust name of saved model in simulate.py

### Method 2
- Run server.py in a terminal
- Run "Client.py 1" in another terminal
- Run "Client.py 2" in another terminal
- follow this until you have successfully launched all x amount of clients

## Evaluating a Trained Models Performance

- Adjust Output CSV and Model_Dir in "evaluate_wer.py"
- Then run "evaluate_wer.py" in a terminal this will save a csv file and print an average WER at the end of evaluation

## Notes

- Training uses batch size = 1 for VRAM safety with Wav2Vec2. Increase if your GPU allows.

- The adversarial client is enabled by default for demonstration.

- simulate.py includes resource controls for Colab and low-VRAM setups.

- Each client seeds randomness for reproducibility.


- Dataset used: Mozilla Common Voice: Please cite appropriately if publishing results.


## Credit

This work was developed by Ross Whitfield with guidance and help from Prof. S Nagaraja


