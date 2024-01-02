import datetime
import os

import cyy_torch_graph  # noqa: F401
import hydra
import torch
from cyy_naive_lib.log import add_file_handler
from cyy_torch_graph import GraphDatasetUtil
from cyy_torch_toolbox import Config

config = Config("", "")


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    config.load_config(conf)


if __name__ == "__main__":
    load_config()
    trainer = config.create_trainer()
    trainer.hook_config.use_slow_performance_metrics = True
    trainer.hook_config.save_performance_metric = True

    add_file_handler(
        os.path.join(
            "log",
            "train",
            config.dc_config.dataset_name,
            config.model_config.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
        )
    )
    edge_drop_rate = 0.99
    assert isinstance(trainer.dataset_util, GraphDatasetUtil)
    edge_mask = trainer.dataset_util.get_edge_masks()[0]
    edge_index = trainer.dataset_util.get_edge_index(graph_index=0)

    dropout_mask = torch.bernoulli(torch.full(edge_mask.size(), 1 - edge_drop_rate)).to(
        dtype=torch.bool
    )
    edge_mask &= dropout_mask

    trainer.dataset_collection.transform_dataset(
        trainer.phase,
        lambda _, dataset_util, __: dataset_util.get_edge_subset(
            graph_index=0, edge_index=edge_index
        ),
    )
    trainer.set_save_dir(os.path.join("./session", config.dc_config.dataset_name))
    trainer.train()
