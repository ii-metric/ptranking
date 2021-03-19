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
parser.add_argument("--model_id", type=str, default=None)
args, _ = parser.parse_known_args()


if __name__ == '__main__':
    models_to_run = ["TopKPre", "RSTopKPre"]
    for model_id in models_to_run:
        args.model_id = model_id
        import json
        json_file = '/home/dl-box/MT2020/ptranking/testing/ltr_dml/json/{}Parameter.json'.format(model_id)
        json_open = open(json_file, 'r')
        json_load = json.load(json_open)
        Bayse_opt_iter = json_load["bayes_opt_iters"]
        if Bayse_opt_iter>0:
            args.bayes_opt_iters = Bayse_opt_iter
            from eval.bayes_opt_runner import BayesOptRunner
            args.reproductions = [int(x) for x in args.reproductions.split(",")]
            runner = BayesOptRunner
            r = runner(**(args.__dict__))
            r.run()
        else:
            from eval.single_experiment_runner import SingleExperimentRunner
            runner = SingleExperimentRunner
            del args.bayes_opt_iters
            del args.reproductions
            r = runner(**(args.__dict__))
            r.run()