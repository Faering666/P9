# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import argparse
from validate import Evaluator, MetricCalculator
from data_loaders import get_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation of predictions.")
    parser.add_argument("--pred_path", type=str, help="Path to directory with prediction data")
    parser.add_argument("--truth_path", type=str, help="Path to directory with ground truth data")
    parser.add_argument("--type", type=str, choices=["Sri-Lanka", "Kazakhstan", "Weedy-Rice"], help="The type of the loader to use for predictions data")
    parser.add_argument("--red_index", type=int, help="The index of the red band", default=1)
    parser.add_argument("--re_index", type=int, help="The index of the red edge band", default=2)
    parser.add_argument("--nir_index", type=int, help="The index of the nir band", default=3)
    parser.add_argument("--result_path", type=str, help="Path to the output directory with filename")
    parser.add_argument("--only_compute", type=bool, help="Only compute from a precomputed result json file - use with '--result_path'.", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    if not args.only_compute:
        metric_calc = MetricCalculator(
            data_range=1.0,
            nir_index=args.nir_index,
            red_index=args.red_index,
            rededge_index=args.re_index
        )

        pred_loader, gt_loader = get_loader(args.type, args.pred_path, args.truth_path)

        evaluator = Evaluator(
            pred_loader=pred_loader,
            gt_loader=gt_loader,
            metric_calculator=metric_calc
        )

        results = evaluator.evaluate()
        evaluator.save_results(args.result_path)

    else:
        results = Evaluator.load_results(args.result_path)

    for i, r in enumerate(results):
        print(f"{i} -- {r['id']}")
        
        for name, value in r.items():
            print(f"{name}: {value}")