import argparse
import sys
import torch

import mlflow_logger


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # print(args, flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--ds', default="complex9", type=str, help='Used stream data set')
    parser.add_argument('--offline', default=10000, type=int, help='Timesteps for offline phase')
    parser.add_argument('--method', default="clustream", type=str, help='Stream Clustering Method')
    parser.add_argument('--seed', default=0, type=int, help='Seed')
    args = parser.parse_args()

    flow_logger = mlflow_logger.MLFlowLogger(experiment_name="SCOPE_Competitors")
    tracking_uri = flow_logger.get_tracking_uri()



    flow_logger.finalise_experiment()



if __name__ == '__main__':
    main(sys.argv)