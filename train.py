#!/usr/bin/env python3
import datetime
import os

from cyy_naive_lib.log import set_file_handler
from cyy_naive_pytorch_lib.default_config import DefaultConfig

if __name__ == "__main__":
    config = DefaultConfig()
    config.load_args()
    trainer = config.create_trainer()

    set_file_handler(
        os.path.join(
            "log",
            "train",
            config.dc_config.dataset_name,
            config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )

    trainer.train()
