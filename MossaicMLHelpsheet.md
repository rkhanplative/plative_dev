#Getting Started

##Nvidia NGC Container Used

##Connecting to the virtual machine
    ssh root@66.135.16.47
    P]5yBWMQP.8j+oC$






##Installing Cuda 11.7 Toolkit (11.7 compatible w/ pytorch 1.13.1)


    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

    #add public keys

    # Old key
    #sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

    # new key, added 2022-04-25 22:52
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

    sudo apt update
    sudo apt install cuda-toolkit-11-7


    


##Environment Setup
    git clone https://github.com/mosaicml/llm-foundry.git
    cd llm-foundry

    apt install python3.10-venv

    python3 -m venv llmfoundry-venv
    source llmfoundry-venv/bin/activate
    
    pip install -e .
    pip install -e ".[gpu]"  



##Download model into scripts/inference directory
    cd scripts/inference
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    
    sudo apt-get install git-lfs

    git lfs install
    git clone https://huggingface.co/mosaicml/mpt-7b

Directory should resemble...

    scripts/inference/my_hf_model/
        config.json
        merges.txt
        pytorch_model.bin
        special_tokens_map.json
        tokenizer.json
        tokenizer_config.json
        vocab.json
        modeling_code.py

To generate response to prompt...

    python3 hf_generate_cont.py \
        --name_or_path fastchat-t5-3b-v1.0 \
        --temperature 1.0 \
        --top_p 0.95 \
        --top_k 50 \
        --seed 1 \
        --max_new_tokens 256

Specify model under name_or_path category



##Training models

    # Convert C4 dataset to StreamingDataset format
    python3 data_prep/convert_dataset_hf.py \
    --dataset c4 --data_subset en \
    --out_root my-copy-c4 --splits train_small val_small \
    --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'

    # Train an MPT-125m model for 10 batches
    composer train/train.py \
    train/yamls/pretrain/mpt-125m.yaml \
    data_local=my-copy-c4 \
    train_loader.dataset.split=train_small \
    eval_loader.dataset.split=val_small \
    max_duration=10ba \
    eval_interval=0 \
    save_folder=mpt-125m

    # Convert the model to HuggingFace format
    python3 inference/convert_composer_to_hf.py \
    --composer_path mpt-125m/ep0-ba10-rank0.pt \
    --hf_output_path mpt-125m-hf \
    --output_precision bf16 \
    # --hf_repo_for_upload user-org/repo-name

    # Evaluate the model on Winograd
    python3 eval/eval.py \
    eval/yamls/hf_eval.yaml \
    icl_tasks=eval/yamls/winograd.yaml \
    model_name_or_path=mpt-125m-hf

    # Generate responses to prompts
    python3 inference/hf_generate.py \
    --name_or_path mpt-125m-hf \
    --max_new_tokens 256 \
    --prompts \
        "The answer to life, the universe, and happiness is" \
        "Here's a quick recipe for baking chocolate chip cookies: Start by"


#Error Handling


Error: “ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:997)”

    sh {path to python installation}/Install Certificates.command

OSError: /anaconda/envs/py38_default/lib/python3.8/site-packages/nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11
    
    rm anaconda/envs/py38_default/lib/python3.8/site-packages/nvidia/cublas/lib/libcublas.so.11

ValueError: --out_root=my-copy-c4 contains ['train_small'] which cannot overlap with the requested splits ['train_small', 'val_small'].

    rm -r my-copy-c4/train_small/

For issues with gcc and glibcxx
    https://stackoverflow.com/questions/65349875/where-can-i-find-glibcxx-3-4-29


make sure cuda-tool-kit and cuda-driver versions match

on vultr, ctk not natiely installed
install ctk 11.7 (bc of pytorch dependencies)

test for drivers using: nvidia-smi
test for cuda using: nvcc -V

