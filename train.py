import datetime
import os

import torch
import torch_geometric.utils
from cyy_torch_toolbox import MachineLearningPhase

try:
    import cyy_torch_vision  # noqa: F401
except BaseException:
    pass
try:
    import cyy_torch_text  # noqa: F401
except BaseException:
    pass
try:
    import cyy_torch_code  # noqa: F401
except BaseException:
    pass
try:
    import cyy_torch_graph  # noqa: F401
except BaseException:
    pass

import hydra
from cyy_naive_lib.log import add_file_handler
from cyy_torch_toolbox import Config

config = Config("", "")
remaining_config = {}


@hydra.main(config_path="conf", version_base=None)
def load_config(conf) -> None:
    global remaining_config
    if len(conf) == 1:
        conf = next(iter(conf.values()))
    remaining_config = config.load_config(conf, check_config=False)


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

    edge_index = trainer.dataset_collection.get_dataset_util(
        phase=MachineLearningPhase.Training
    ).get_edge_index(graph_index=0)

    training_node_mask = trainer.dataset_util.get_mask()[0]
    training_edge_mask = (
        training_node_mask[edge_index[0]] & training_node_mask[edge_index[1]]
    )
    dropout_mask = torch.bernoulli(
        torch.full(training_edge_mask.size(), 1 - edge_drop_rate)
    ).to(dtype=torch.bool)
    print(training_edge_mask.shape)
    training_edge_mask &= dropout_mask

    edge_index = torch_geometric.utils.coalesce(edge_index[:, training_edge_mask])
    trainer.dataset_collection.transform_dataset(
        trainer.phase,
        lambda _, dataset_util, __: dataset_util.get_edge_subset(
            graph_index=0, edge_index=edge_index
        ),
    )
    trainer.update_dataloader_kwargs(**remaining_config.pop("dataloader_kwargs"))
    assert not remaining_config
    trainer.set_save_dir(os.path.join("./session", config.dc_config.dataset_name))
    trainer.train()
