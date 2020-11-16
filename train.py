"""Тренируем модели в соответствии с конфигом обучения"""
import argparse
import pprint

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from base import BaselineLearner
from helper_functions import load_cfg


def main(args) -> None:
    """
    Функция запуска обучения.
    """
    config = load_cfg(args.config)
    pretty_printer = pprint.PrettyPrinter(indent=2)
    pretty_printer.pprint(config)

    model = BaselineLearner(config)

    logger = False
    if args.use_logger:
        logger = WandbLogger(name=config.name)
        logger.watch(model.net)

    trainer = pl.Trainer(
        gpus=args.gpus,
        logger=logger,
        callbacks=[
            ModelCheckpoint(monitor='valid_loss',
                            dirpath=config.sources.ckpt_path,
                            filename=config.name)
        ],
        max_epochs=config.training.epochs,
        distributed_backend=args.distributed_backend,
        precision=16 if args.use_amp else 32,
    )

    trainer.fit(model)
    print('Model training completed!')


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    # путь к конфигу
    PARSER.add_argument('--config', '-c', type=str, default="config.yaml",
                        help="path to the config")
    PARSER.add_argument("--gpus", type=int, default=-1, help="number of available GPUs")
    PARSER.add_argument('--distributed-backend', type=str, default='ddp', choices=('dp', 'ddp', 'ddp2'),
                        help='supports three options dp, ddp, ddp2')
    PARSER.add_argument('--use_amp', action='store_true', help='if true uses 16 bit precision')
    PARSER.add_argument('--use_logger', action='store_true', help='use wandb logger')

    ARGUMENTS = PARSER.parse_args()
    print(ARGUMENTS)
    main(ARGUMENTS)
