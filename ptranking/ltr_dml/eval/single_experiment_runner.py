#! /usr/bin/env python3
import logging
logging.info("Importing packages in single_experiment_runner")
from powerful_benchmarker.utils import common_functions as c_f, dataset_utils as d_u
from powerful_benchmarker.runners.base_runner import BaseRunner
import glob
import os
logging.info("Done importing packages in single_experiment_runner")
from losses.TopKPre import TopKPreLoss
from losses.RSTopKPre import RSTopKPreLoss
from easy_module_attribute_getter import YamlReader, PytorchGetter, utils as emag_utils
import argparse

id_to_model = {"TopKPre": TopKPreLoss,
               "RSTopKPre": RSTopKPreLoss}

class SingleExperimentRunner(BaseRunner):
    def __init__(self, model_id=None, **kwargs):
        super().__init__(**kwargs)
        self.model_id=model_id

    def run(self):
        self.register("loss", id_to_model[self.model_id])
        YR =self.set_YR_from_json(self.model_id)
        if YR.args.reproduce_results:
            return self.reproduce_results(YR)
        else:
            return self.run_new_experiment_or_resume(YR)

    def set_YR_from_json(self, model_id):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        # experiment name
        e_name = ""
        import json
        json_file = '/home/dl-box/MT2020/ptranking/testing/ltr_dml/json/{}Parameter.json'.format(model_id)
        json_open = open(json_file, 'r')
        json_load = json.load(json_open)
        e_name+=model_id+"_"
        json_para = json_load["Parameters"]
        for loss in json_para:
            for para in json_para[loss]:
                e_name+=para
                e_name+=str(json_para[loss][para])+"_"
        e_name+=list(json_load["EvalSettings"]["dataset~OVERRIDE~"].keys())[0]+"_"
        e_name+="B"
        e_name+=str(json_load["EvalSettings"]["trainer"]["MetricLossOnly"]["batch_size"])+"_"
        e_name+="m"
        e_name+=str(json_load["EvalSettings"]["sampler"]["MPerClassSampler"]["m"])


        parser.add_argument("--experiment_name", type=str, required=False, default=e_name)
        parser.add_argument("--resume_training", type=str, default=None, choices=["latest", "best"])
        parser.add_argument("--evaluate", action="store_true")
        parser.add_argument("--evaluate_ensemble", action="store_true")
        parser.add_argument("--reproduce_results", type=str, default=None)

        YR = YamlReader(argparser=parser)
        YR.args.dataset_root = self.dataset_root
        YR.args.experiment_folder = os.path.join(self.root_experiment_folder, YR.args.experiment_name)
        YR.args.place_to_save_configs = os.path.join(YR.args.experiment_folder, "configs")
        config_foldernames_yaml = "{}.yaml".format(self.config_foldernames_base)
        foldername_info = None
        if not hasattr(YR.args, self.config_foldernames_base):
            # first try loading config_foldernames from "place_to_save_configs", in case we're resuming
            already_saved_config_foldernames = os.path.join(YR.args.place_to_save_configs, config_foldernames_yaml)
            if os.path.isfile(already_saved_config_foldernames):
                foldername_info = c_f.load_yaml(already_saved_config_foldernames)
            else:
                foldername_info = c_f.load_yaml(os.path.join(self.root_config_folder, config_foldernames_yaml))
            YR.args.config_foldernames = foldername_info[self.config_foldernames_base]

        for subfolder in YR.args.config_foldernames:
            if not hasattr(YR.args, subfolder):
                yaml_names = ["default"] if foldername_info is None else foldername_info[subfolder]
                setattr(YR.args, subfolder, yaml_names)

        #### loss parameters
        json_file = '/home/dl-box/MT2020/ptranking/testing/ltr_dml/json/{}Parameter.json'.format(model_id)
        json_open = open(json_file, 'r')
        json_load = json.load(json_open)
        json_para = json_load["Parameters"]
        for loss in json_para:
            for para in json_para[loss]:
                YR.args.__setattr__('loss_funcs~OVERRIDE~', {'metric_loss': {loss: {para: json_para[loss][para]}}})

        #### Eval Settings
        json_eval = json_load["EvalSettings"]
        for para in json_eval:
            YR.args.__setattr__(para, json_eval[para])
        return YR

    def start_experiment(self, args):
        api_parser = self.get_api_parser(args)
        run_output = api_parser.run()
        del api_parser
        return run_output

    def run_new_experiment_or_resume(self, YR):
        # merge_argparse at the beginning of training, or when evaluating
        merge_argparse = self.merge_argparse_when_resuming if YR.args.resume_training else True
        args, _, args.dict_of_yamls = YR.load_yamls(self.determine_where_to_get_yamls(YR.args),
                                                    max_merge_depth=float('inf'),
                                                    max_argparse_merge_depth=float('inf'),
                                                    merge_argparse=merge_argparse)
        return self.start_experiment(args)

    def reproduce_results(self, YR, starting_fresh_hook=None, max_merge_depth=float('inf'),
                          max_argparse_merge_depth=float('inf')):
        configs_folder = os.path.join(YR.args.reproduce_results, 'configs')
        default_configs = self.get_root_config_paths(YR.args)  # default configs
        experiment_config_paths = self.get_saved_config_paths(YR.args,
                                                              config_folder=configs_folder)  # reproduction configs
        for k, v in experiment_config_paths.items():
            if any(not os.path.isfile(filename) for filename in v):
                logging.warning("{} does not exist. Will use default config for {}".format(v, k))
                experiment_config_paths[k] = default_configs[k]
        args, _, args.dict_of_yamls = YR.load_yamls(config_paths=experiment_config_paths,
                                                    max_merge_depth=max_merge_depth,
                                                    max_argparse_merge_depth=max_argparse_merge_depth,
                                                    merge_argparse=self.merge_argparse_when_resuming)

        # check if there were config diffs if training was resumed
        temp_split_manager = self.pytorch_getter.get("split_manager", yaml_dict=args.split_manager)
        resume_training_dict = c_f.get_all_resume_training_config_diffs(configs_folder, temp_split_manager)

        if len(resume_training_dict) > 0:
            for sub_folder, num_epochs_dict in resume_training_dict.items():
                # train until the next config diff was made
                args.num_epochs_train = num_epochs_dict
                self.start_experiment(args)
                # Start fresh
                YR = self.setup_yaml_reader()
                if starting_fresh_hook: starting_fresh_hook(YR)
                # load the experiment configs, plus the config diffs
                for k in glob.glob(os.path.join(sub_folder, "*")):
                    config_name = os.path.splitext(os.path.basename(k))[0]
                    experiment_config_paths[config_name].append(k)
                args, _, args.dict_of_yamls = YR.load_yamls(config_paths=experiment_config_paths,
                                                            max_merge_depth=0,
                                                            max_argparse_merge_depth=max_argparse_merge_depth,
                                                            merge_argparse=self.merge_argparse_when_resuming)
                args.resume_training = "latest"
        return self.start_experiment(args)