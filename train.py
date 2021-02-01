#!/usr/bin/env python3
import datetime
import os
import sys

from cyy_naive_lib.log import get_logger, set_file_handler
from cyy_naive_pytorch_lib.arg_parse import (affect_global_process_from_args,
                                             create_trainer_from_args,
                                             get_arg_parser, get_parsed_args)

if __name__ == "__main__":
    parser = get_arg_parser()
    parser.add_argument("--repeated_num", type=int, default=None)
    args = get_parsed_args(parser=parser)
    affect_global_process_from_args(args)

    if args.repeated_num is None:
        set_file_handler(
            os.path.join(
                "log",
                "train",
                args.dataset_name,
                args.model_name,
                "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
            )
        )
    else:
        set_file_handler(
            os.path.join(
                "log",
                "repeated_train",
                args.dataset_name,
                args.model_name,
                "{date:%Y-%m-%d_%H:%M:%S}.log".format(date=datetime.datetime.now()),
            )
        )
    trainer = create_trainer_from_args(args)

    if args.repeated_num is None:
        trainer.train(plot_class_accuracy=True)
        trainer.save_model(args.save_dir)
        sys.exit(0)

    results = trainer.repeated_train(
        args.repeated_num, args.save_dir, plot_class_accuracy=True
    )
    get_logger().info("training_loss is %s", results["training_loss"])
    get_logger().info("validation_loss is %s", results["validation_loss"])
    get_logger().info("validation_accuracy is %s", results["validation_accuracy"])
