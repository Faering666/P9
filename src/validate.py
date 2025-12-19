import json
from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np
from metric_calculator import MetricCalculator

class Evaluator:
    """
    Evaluates predictions vs ground truths using a MetricCalculator.

    pred_loader, gt_loader:
        Zero-argument callables returning dicts:
            {
              "<id>": {
                  "cube": <array-or-tensor>,
                  "path": <optional-path>,
                  ... (other metadata)
              },
              ...
            }

    metric_calculator:
        Object with a method:
            compute(pred_cube, gt_cube) -> dict[str, float]
        (the MetricCalculator we built earlier).
    """

    def __init__(
        self,
        pred_loader: Callable[[], dict[str, dict[str, Any]]],
        gt_loader: Callable[[], dict[str, dict[str, Any]]],
        metric_calculator: MetricCalculator,
    ):
        self.pred_loader = pred_loader
        self.gt_loader = gt_loader
        self.metric_calculator = metric_calculator
        
        self._last_results: list[dict[str, Any]] | None = None

    def _index_by_name(self, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Turn a list of dicts into a dict keyed by 'name'."""
        indexed = {}
        for item in items:
            name = item["name"]
            if name in indexed:
                raise ValueError(f"Duplicate name '{name}' detected in loader output")
            indexed[name] = item
        return indexed

    def evaluate(self) -> list[dict[str, Any]]:
        """
        Run evaluation over all matching prediction/GT pairs.

        Returns a list of dicts, each like:
          {
            "name": <image name>,
            "path_pred": <pred path or None>,
            "path_gt": <gt path or None>,
            **metrics...
          }
        """
        # Load prediction and GT descriptors
        pred_map = self.pred_loader()
        gt_map = self.gt_loader()

        pred_ids = set(pred_map.keys())
        gt_ids = set(gt_map.keys())
        common_ids = sorted(pred_ids & gt_ids)

        if not common_ids:
            raise ValueError(
                "No overlapping IDs between prediction and ground truth loaders.\n"
                f"pred IDs (sample): {list(sorted(pred_ids))[:5]}\n"
                f"gt IDs   (sample): {list(sorted(gt_ids))[:5]}"
            )

        results: list[dict[str, Any]] = []

        for i, sid in enumerate(common_ids):
            print(f"Computing metrics for '{sid}' :: {i + 1}/{len(common_ids)}")
            p_info = pred_map[sid]
            g_info = gt_map[sid]

            pred_cube = p_info["cube"]
            gt_cube = g_info["cube"]
            breakpoint()
            gt_cube = cv2.resize(gt_cube, (256, 256), interpolation=cv2.INTER_AREA)

            metrics = self.metric_calculator.compute(pred_cube, gt_cube)

            rec: dict[str, Any] = {
                "id": sid,
                "path_pred": p_info.get("path"),
                "path_gt": g_info.get("path"),
            }
            rec.update(metrics)
            results.append(rec)
        
        self._last_results = results
        return results

    def save_results(
        self,
        path: str | Path,
        results: list[dict[str, Any]] | None = None
    ) -> None:
        """
        Save results to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        results : list of dict, optional
            If provided, this list is saved. If None, the last results
            produced by evaluate() are saved.
        """
        if results is None:
            if self._last_results is None:
                raise ValueError(
                    "No results to save: call evaluate() first or "
                    "pass a results list explicitly."
                )
            results_to_save = self._last_results
        else:
            results_to_save = results
            
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(results_to_save, f, indent=4)
        
    @staticmethod
    def load_results(path: str | Path) -> list[dict[str, Any]]:
        """
        Load previously saved results from a JSON file.

        Parameters
        ----------
        path : str or Path
            The file that was written by save_results().

        Returns
        -------
        list of dict
            The list of result records.
        """
        
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            
        return data