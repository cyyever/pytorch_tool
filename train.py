import datetime
import os
import muon_optimizer

try:
    import cyy_torch_vision  # noqa: F401
except BaseException:
    pass
try:
    import cyy_torch_text  # noqa: F401
except BaseException:
    pass
import hydra
from cyy_naive_lib.log import add_file_handler
from cyy_torch_toolbox import Config
from cyy_torch_toolbox.hyper_parameter import global_optimizer_factory
from muon import MuonWithAuxAdam

config = Config("", "")


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    config.load_config(conf, check_config=True)


if __name__ == "__main__":
    global_optimizer_factory.register("moun", MuonWithAuxAdam)
    load_config()
    trainer = config.create_trainer()

    task_time = datetime.datetime.now()
    filename = f"{task_time:%Y-%m-%d_%H_%M_%S}.log"
    add_file_handler(
        os.path.join(
            "log",
            "train",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            filename,
        )
    )

    print(trainer.model)
    trainer.train()
