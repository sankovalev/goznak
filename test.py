"""Считаем метрики на val"""
import argparse
import pprint
import pandas as pd
import numpy as np

from base import BaselinePredictor
from helper_functions import load_cfg


def main(args) -> None:
    """
    Функция запуска предсказателя.
    """
    config = load_cfg(args.config)
    pretty_printer = pprint.PrettyPrinter(indent=2)
    pretty_printer.pprint(config)

    model = BaselinePredictor(config)
    metrics = model.calculate_metrics()

    for key, value in metrics.items():
        print(key, "===>", np.mean(value))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    # путь к конфигу
    PARSER.add_argument('--config', '-c', type=str, default="config.yaml",
                        help="path to the config")

    ARGUMENTS = PARSER.parse_args()
    print(ARGUMENTS)
    main(ARGUMENTS)
