import logging
import argparse
logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--pytorch_home", type=str, default="/media/dl-box/f12286fd-f13c-4fe0-a92d-9f935d6a7dbd/pretrained")
parser.add_argument("--dataset_root", type=str, default="/media/dl-box/f12286fd-f13c-4fe0-a92d-9f935d6a7dbd/CVPR/ptraniking") #
parser.add_argument("--root_experiment_folder", type=str, default="/media/dl-box/f12286fd-f13c-4fe0-a92d-9f935d6a7dbd/Project_output/Out_img_metric/ptranking_dml")
parser.add_argument("--global_db_path", type=str, default=None)
parser.add_argument("--merge_argparse_when_resuming", default=False, action='store_true')
parser.add_argument("--root_config_folder", type=str, default=None)
parser.add_argument("--bayes_opt_iters", type=int, default=0)
parser.add_argument("--reproductions", type=str, default="0")
args, _ = parser.parse_known_args()

if args.bayes_opt_iters > 0:
    from powerful_benchmarker.runners.bayes_opt_runner import BayesOptRunner
    args.reproductions = [int(x) for x in args.reproductions.split(",")]
    runner = BayesOptRunner
else:
    # from powerful_benchmarker.runners.single_experiment_runner import SingleExperimentRunner
    from eval.single_experiment_runner import SingleExperimentRunner
    runner = SingleExperimentRunner
    del args.bayes_opt_iters
    del args.reproductions


if __name__ == '__main__':
    r = runner(**(args.__dict__))
    # register loss functions
    # r.register("loss", TopKPreLoss)
    # r.register("loss", RSTopKPreLoss)
    models_to_run = ["TopKPre", "RSTopKPre"]
    for model_id in models_to_run:
        r.run(model_id=model_id)