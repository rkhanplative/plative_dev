import guidance
import itertools
import os
import random
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from contextlib import nullcontext

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer



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
    parser.add_argument('--max_batch_size', type=int, default=128)
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

def main():
    
    
    # we use StableLM as an open example, but these issues impact all models to varying degrees
    guidance.llm = guidance.llms.Transformers("stabilityai/stablelm-base-alpha-3b", device=0)

    # we turn token healing off so that guidance acts like a normal prompting library
    program = guidance('''The link is <a href="http:{{gen max_tokens=10 token_healing=False}}''')
    program()

if __name__ == '__main__':
    main(parse_args())