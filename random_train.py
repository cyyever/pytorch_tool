#!/usr/bin/env python3
import os
import json
import datetime

from cyy_naive_lib.log import set_file_handler
from tools.dataset import (
    replace_dataset_labels,
    randomize_subset_label,
)
from tools.arg_parse import (
    get_parsed_args,
    __create_unique_save_dir,
    create_trainer_from_args,
    get_arg_parser,
)


if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--random_percentage", type=float, required=True)
    args = get_parsed_args(parser)
    args.save_dir = __create_unique_save_dir(
        os.path.join("randomized_models", args.task_name)
    )

    set_file_handler(
        os.path.join(
            "log",
            "randomized_train",
            args.dataset_name,
            args.model_name,
            "{date:%Y-%m-%d_%H:%M:%S}.log".format(
                date=datetime.datetime.now()),
        )
    )

    trainer = create_trainer_from_args(args)
    training_dataset = trainer.training_dataset

    randomized_label_map = randomize_subset_label(
        training_dataset, args.random_percentage
    )

    os.makedirs(args.save_dir, exist_ok=True)
    with open(
        os.path.join(
            args.save_dir,
            "randomized_label_map.json",
        ),
        mode="wt",
    ) as f:
        json.dump(randomized_label_map, f)
    trainer.training_dataset = replace_dataset_labels(
        training_dataset, randomized_label_map
    )

    trainer.train()
