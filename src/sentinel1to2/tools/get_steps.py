def get_steps(args):
  step_choices = ["preprocessing", "training", "evaluation", "inference", "performance"]
  steps = {}
  if args.step not in step_choices:
    return {}
  if args.step == "preprocessing":
    steps["preprocessing"] = 0
    return steps

  if args.all_steps:
    steps["training"] = 1
    if args.step == "training":
      return steps
    else:
      steps["evaluation"] = 2
      if args.step == "evaluation":
        return steps
      else:
        steps["inference"] = 3
        if args.step == "inference":
          return steps
        else:
          steps["performance"] = 4
          return steps
  elif args.step == "training":
    steps["training"] = 1
    return steps
  elif args.step == "evaluation":
    steps["evaluation"] = 1
    return steps
  elif args.step == "inference":
    steps["inference"] = 1
    return steps
  elif args.step == "performance":
    steps["performance"] = 1
    return steps
  return {}
