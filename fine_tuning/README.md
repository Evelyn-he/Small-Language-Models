# Instructions on Fine-tuning a model

** See the end of the README for an overall structure of where all the files are placed.

## Set up CCDB account and remote ssh

1. Login to `https://ccdb.alliancecan.ca/`

2. In Powershell on your machine, create a ssh key and copy the public key to `https://ccdb.alliancecan.ca/ssh_authorized_keys`

3. Go to `https://ccdb.alliancecan.ca/multi_factor_authentications` and enable mfa on your phone (duo)

4. For whatever clusters you want to login to, agree to their terms of use here: `https://ccdb.alliancecan.ca/me/access_systems`

5. In Powershell, `ssh username@cluster.computecanada.ca` to login (i.e. `ssh ehe@narval.computecanada.ca`). Enter `1` when prompted for two-factor authentication, then check the Duo app to accept.

## Set up Environment on CCDB

1. Create a virtual environment using `venv`. 

    ```
    python -m venv ~/<environment_name>
    ```
    
    You can then open the environment in the future using
    ```
    source ~/<environment_name>/bin/activate
    ```

    And when in the environment, you can leave using
    ```
    deactivate
    ```

2. Open the environment and install necessary packages.

    ```
    module load gcc arrow
    source ~/<environment_name>/bin/activate
    pip install --upgrade pip
    pip install torch transformers protobuf sentencepiece safetensors accelerate fastapi uvicorn datasets peft trl pandas numpy
    deactivate
    ```

    Note: The line `module load gcc arrow` is added because Canada Compute blocks `pip` from building `pyarrow` and requires you to use their prebuilt system module. Several modules are dependent on `pyarrow`.


3. To run and train the model, you will need a SLURM `.slurm` script to request jobs and a python `.py` script to actually run training and inference on the model. You'll also need a directory to write the output to. I have created example scripts in this repository called `queue_job.slurm` and `train.py`. You can create the scripts in CCDB like so:

    ```
    mkdir ~/projects/def-khisti/<user>/scripts
    mkdir ~/projects/def-khisti/<user>/outputs
    cd ~/projects/def-khisti/<user>/scripts
    cat queue_job.slurm
    cat train.py
    ```

## Download Model & Dataset

1. On your local machine (i.e. not CCDB), download the huggingface model. Apparently, it is common that CCDB nodes cannot access huggingface directly so we can circumvent this issue by downloading the model onto our local machine and then copying to CCDB. The model I have listed is the huggingface equivalent of ollama's phi3:3.8b. They are the same model but in different formats. It is easiest to train using the huggingface model and then convert to ollama format afterwards.

    ```
    huggingface-cli download microsoft/Phi-3-mini-4k-instruct \
        --local-dir phi3 \
        --local-dir-use-symlinks False
    ```

2. Copy the model to the directory you want it in CCDB. Then, you can change the model path in the training script to point to that directory.

    ```
    scp -r <model-path>/phi3 <username>@<network>.computecanada.ca:<model-path>/phi3
    ```

3. Copy the dataset file in this directory, or create a new dataset file in CCDB if you want to fine-tune the model differently. Copy the format I have used in `dataset.json`.
    
    ```
    scp -r fine_tuning/dataset.json <username>@<network>.computecanada.ca:<dataset-path>
    ```

## Queuing a Job & Training

To actually queue a job (i.e. run the training script), you can see `https://docs.alliancecan.ca/wiki/Running_jobs`. Here are the basics:

```
# To submit a job
sbatch ~/projects/def-khisti/<user>/scripts/queue_job.slurm
```
You can check the job's status by using the `sq` command and looking at the status column. `R` means the job is currently running and `PD` means that it is still pending. Check the output folder for anything the job writes.

## Format Model For Ollama & Insert Into Our Script

As mentioned before, the model is fine-tuned in hugging-face format. Now we need to transform it into a GGUF format so it can be loaded and run by ollama. 

1. Before proceeding, we need to merge LoRA adapters together using the `merge_lora.py` script. In the `queue_job.scrum` file, change the path to `merge_lora.py`. Then,
    ```
    sbatch queue_job.scrum
    ```
    The converter we will be using cannot detect LoRA adapters which we've used to fine-tune the model. Therefore, we must merge them before calling it.


2. Clone the `llama.cpp` repository. This repository includes scripts to convert a model to GGUF. Run the conversion script in it.

    ```
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    make
    source ~/<environment_name>/bin/activate
    python convert_hf_to_gguf.py \
        ~/scratch/phi3-merged \
        --outfile ~/scratch/phi3-q8_0.gguf \
        --outtype q8_0
    deactivate
    ```
    You can change the `--outtype` perameter to the quant format you want:
    - Full precision: `f16`, `f32`
    - Reduced precision: `q8_0`, `q4_K_M`       <-- More speed up & smaller model size

3. Save the `.gguf` file to your local computer by running the following command locally:
    ```
    scp <username>@<network>.computecanada.ca:/home/ehe/scratch/phi3-q8_0.gguf .
    ```

4. Create the ollama model using a `Modelfile`. An example file is included in this directory.

    ```
    ollama create phi3-rag -f fine_tuning/Modelfile
    ```
    Now, Ollama has a new model called `phi3-rag`. Update the `MODEL = "phi3:3.8b"` to `MODEL = "phi3-rag"` in `slm.py` to use this model instead.


## Directory Layout
```
/home/<user>/
    |----- <environment>/
    |----- scratch/
    |           |----- llama.cpp/
    |           |           |----- convert_hf_to_gguf.py
    |           |----- phi3/
    |           |----- phi3-finetuned/
    |           |----- phi3-merged/
    |           |----- dataset.json
    |----- projects/def-khisti/<user>/
    |           |----- outputs/
    |           |           |----- <output>.err
    |           |           |----- <output>.out
    |           |----- scripts/
    |           |           |----- queue_job.slurm
    |           |           |----- train.py
    |           |           |----- merge_lora.py
```