import sys
sys.executable = 'miniconda3/envs/VLM/bin/python3.10'
sys.path += ['/home/hice1/jwessell6/VLM/hw1', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10', 
    '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/lib-dynload']
import pickle
import argparse
import os
from pathlib import Path

def main(path):
    dirs = os.listdir(path)
    correct = 0
    count = 0
    for video in dirs:
        res = f'{path}/{video}/results_blind.pkl'
        #ran into issues with some videos
        if not Path(res).exists():
            continue
        with open(res, "rb") as file:
            results = pickle.load(file)
        for pair in results:
            response = pair[1]
            if response[-1] == 'E':
                correct += 1
            count += 1
    print(correct/count)
    print(count)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for step3")

    # Define expected arguments
    parser.add_argument("--path", type=str, default = "~/scratch/VLM_Data/eval_100/training", help="file path containing information to measure.")
    args = parser.parse_args()
    main(args.path)