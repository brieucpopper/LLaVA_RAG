import sys
# sys.executable = 'miniconda3/envs/VLM/bin/python3.10'
# sys.path += ['/home/hice1/jwessell6/VLM/hw1', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10', 
#     '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/VLM/lib/python3.10/lib-dynload']
import pickle
import argparse
import os
from pathlib import Path
def main(path, blind):
    dirs = os.listdir(path)
    correct = 0
    count = 0
    example = True
    for video in dirs:
        res = f'{path}/{video}/results.pkl'
        #ran into issues with some videos
        if not Path(res).exists():
            continue
        with open(res, "rb") as file:
            results = pickle.load(file)
        for pair in results:
            if example:
                print(pair)
                print(results)
            true = pair[0]
            response = pair[1][pair[1].find("[/INST]"):]
            if example:
                print(response)
                print(path, video)
                example = False
            if true in response:
                correct += 1
            count += 1
    print(correct/count)
    print(count)
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for step3")

    # Define expected arguments
    parser.add_argument("--path", type=str, default = "~/scratch/VLM_Data/eval_20/training", help="file path containing information to measure.")
    parser.add_argument('-b', action = 'store_true', help='Whether to use image-blind evaluation')
    args = parser.parse_args()
    main(args.path, args.b)