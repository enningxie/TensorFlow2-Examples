# coding=utf-8
import json
from src.esim_keras import ESIM
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Runner(object):
    def __init__(self, config_path):
        with open(config_path, "r") as fr:
            self.config = json.load(fr)

    def run(self):
        esim_obj = ESIM(self.config)
        esim_obj.train_()


if __name__ == '__main__':
    tmp_runner = Runner('data/config.json')
    tmp_runner.run()
