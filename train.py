import datetime
import os

import hydra
from cyy_naive_lib.log import set_file_handler
from cyy_torch_toolbox.default_config import DefaultConfig

config = DefaultConfig()

remain_config = {}


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    global remain_config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    remain_config = DefaultConfig.load_config(config, conf, check_config=False)


if __name__ == "__main__":
    load_config()
    trainer = config.create_trainer()

    set_file_handler(
        os.path.join(
            "log",
            "train",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )

    if "use_amp" in remain_config:
        print("use AMP")
        trainer.set_amp()
    trainer.train()
