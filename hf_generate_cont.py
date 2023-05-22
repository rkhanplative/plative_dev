# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import random
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from contextlib import nullcontext

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM


def get_dtype(dtype):
    if dtype == 'fp32':
        return torch.float32
    elif dtype == 'fp16':
        return torch.float16
    elif dtype == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError(
            f'dtype {dtype} is not supported. ' +\
            f'We only support fp32, fp16, and bf16 currently')


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str_or_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description='Load a HF CausalLM Model and use it to generate text.')
    parser.add_argument('-n', '--name_or_path', type=str, required=True)
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--max_batch_size', type=int, default=None)
    #####
    # Note: Generation config defaults are set to match Hugging Face defaults
    parser.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    parser.add_argument('--top_k', type=int, nargs='+', default=[50])
    parser.add_argument('--top_p', type=float, nargs='+', default=[1.0])
    parser.add_argument('--repetition_penalty',
                        type=float,
                        nargs='+',
                        default=[1.0])
    parser.add_argument('--no_repeat_ngram_size',
                        type=int,
                        nargs='+',
                        default=[0])
    #####
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--do_sample',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_cache',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--eos_token_id', type=int, default=None)
    parser.add_argument('--pad_token_id', type=int, default=None)
    parser.add_argument('--model_dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default=None)
    parser.add_argument('--autocast_dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default=None)
    parser.add_argument('--warmup',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--trust_remote_code',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_auth_token',
                        type=str_or_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--attn_impl', type=str, default=None)
    # TODO
    # parser.add_argument('--fsdp',
    #                     type=str2bool,
    #                     nargs='?',
    #                     const=True,
    #                     default=False)
    return parser.parse_args()




def maybe_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args: Namespace) -> None:
    # TODO
    # if args.fsdp and (not torch.cuda.is_available()):
    #     raise ValueError(
    #         'Cannot use FSDP because no cuda devices are available.')


    # Grab config first
    print(f'Loading HF Config...')
    from_pretrained_kwargs = {
        'use_auth_token': args.use_auth_token,
        'trust_remote_code': args.trust_remote_code,
        'revision': args.revision,
    }
    try:
        config = AutoConfig.from_pretrained(args.name_or_path,
                                            **from_pretrained_kwargs)
        
            
        
        if args.attn_impl is not None and hasattr(config, 'attn_config'):
            config.attn_config['attn_impl'] = args.attn_impl
        if args.max_seq_len is not None and hasattr(config, 'max_seq_len'):
            config.max_seq_len = args.max_seq_len

    except Exception as e:
        raise RuntimeError(
            'If you are having auth problems, try logging in via `huggingface-cli login` ' +\
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... ' +\
            'using your access token from https://huggingface.co/settings/tokens.'
        ) from e

    # Set model_dtype
    if args.model_dtype is not None:
        model_dtype = get_dtype(args.model_dtype)
    else:
        model_dtype = config.torch_dtype or torch.float32

    # Load HF Model
    print(f'Loading HF model with dtype={model_dtype}...')
    try:
        if args.name_or_path == "fastchat-t5-3b-v1.0":
            model = AutoModelForSeq2SeqLM.from_pretrained(args.name_or_path,
                                                     config=config,
                                                     torch_dtype=model_dtype,
                                                     **from_pretrained_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.name_or_path,
                                                     config=config,
                                                     torch_dtype=model_dtype,
                                                     **from_pretrained_kwargs)
    except Exception as e:
        raise RuntimeError(
            'If you are having auth problems, try logging in via `huggingface-cli login` ' +\
            'or by setting the environment variable `export HUGGING_FACE_HUB_TOKEN=... ' +\
            'using your access token from https://huggingface.co/settings/tokens.'
        ) from e
    model.eval()
    print(f'n_params={sum(p.numel() for p in model.parameters())}')

    # TODO
    # if args.fsdp:
    #     ...
    # else:
    # Set device
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Placing model on {device=}...')
    model.to(device)

    print('\nLoading HF tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path,
                                              **from_pretrained_kwargs)
    if tokenizer.pad_token_id is None:
        warnings.warn(
            'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
        )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    # Autocast
    if args.autocast_dtype is not None:
        autocast_dtype = get_dtype(args.autocast_dtype)
        autocast_context = torch.autocast(device, autocast_dtype)
        print(f'Using autocast with dtype={autocast_dtype}...')
    else:
        autocast_context = nullcontext()
        print('NOT using autocast...')

    done_warmup = False

    for temp, topp, topk, repp, nrnz, seed in itertools.product(
            args.temperature, args.top_p, args.top_k, args.repetition_penalty,
            args.no_repeat_ngram_size, args.seed):

        # Seed randomness
        random.seed(seed)
        torch.manual_seed(seed)

        generate_kwargs = {
            'max_new_tokens': args.max_new_tokens,
            'temperature': temp,
            'top_p': topp,
            'top_k': topk,
            'repetition_penalty': repp,
            'no_repeat_ngram_size': nrnz,
            'use_cache': args.use_cache,
            'do_sample': False if temp == 0 else args.do_sample,
            'eos_token_id': args.eos_token_id or tokenizer.eos_token_id,
            'pad_token_id': args.pad_token_id or tokenizer.pad_token_id,
            # TODO
            # 'synced_gpus': args.fsdp,
        }

        # Generate function with correct context managers
        def _generate(encoded_inp):
            with torch.no_grad():
                with autocast_context:
                    return model.generate(
                        input_ids=encoded_inp['input_ids'],
                        attention_mask=encoded_inp['attention_mask'],
                        **generate_kwargs,
                    )
        while True:
            temp = input("Enter temperature: ")
            if temp != "": generate_kwargs['temperature'] = float(temp)
            print(f'\nGenerate kwargs:\n{generate_kwargs}')
                    
            prompt = input("Enter your prompt: ")
            prompt_strings = [prompt]
            if prompt == 'quit':
                break
            start = time.time()
            
            # Split into prompt batches
            batches = []
            if args.max_batch_size:
                bs = args.max_batch_size
                batches = [
                    prompt_strings[i:i + bs]
                    for i in range(0, len(prompt_strings), bs)
                ]

            else:
                batches = [prompt_strings]

            for batch in batches:
                maybe_synchronize()
                encoded_inp = tokenizer(batch, return_tensors='pt', padding=True)
                for key, value in encoded_inp.items():
                    encoded_inp[key] = value.to(device)
                maybe_synchronize()
                input_tokens = torch.sum(
                    encoded_inp['input_ids'] != tokenizer.pad_token_id,
                    axis=1).numpy(force=True)  # type: ignore

                # Warmup
                if args.warmup and (not done_warmup):
                    _ = _generate(encoded_inp)
                    done_warmup = True

                # Run HF generate
                maybe_synchronize()
                encoded_gen = _generate(encoded_inp)
                maybe_synchronize()

                decoded_gen = tokenizer.batch_decode(encoded_gen,
                                                    skip_special_tokens=True)
                maybe_synchronize()
                gen_tokens = torch.sum(encoded_gen != tokenizer.pad_token_id,
                                    axis=1).numpy(force=True)  # type: ignore
                
            end = time.time()
            
            delimiter = '#' * 100
            for prompt, gen in zip(batch, decoded_gen):
                continuation = gen[len(prompt):]
                print(delimiter)
                print('\033[92m' + prompt + '\033[0m' + continuation)
            print(delimiter)
            print(f"Elapsed time = {end - start} seconds")            
            print(delimiter)
        
        


        

if __name__ == '__main__':
    main(parse_args())
