from pathlib import Path
def parse_args(argparse):
  parser = argparse.ArgumentParser(description="Sentinel 1 to Sentinel 2 images translator")
  parser.add_argument(
      "-c",
      "--config",
      type = Path,
      default= "configs/default_config.yaml",
      help="YAML cofiguration file",
      )

  parser.add_argument(
      "-a",
      "--all_steps",
      type = bool,
      default= False,
      help="Set to True if you want to run all the scripts preceeding the selected step.",
      )

  parser.add_argument(
      "-s",
      "--step",
      choices=["preprocessing", "training", "evaluation", "inference", "performance"],
      required = True,
      help="The step to run: training, evaluation, inference or performance.",
      )

  return parser.parse_args()
