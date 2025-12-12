from validate import Evaluator, MetricCalculator
from data_loaders import make_weedy_rice_tif_loader


metric_calc = MetricCalculator(
    data_range=1.0,
    nir_index=3,
    red_index=1,
    rededge_index=2
)

weedy_rice = "C:/Users/tobia/Downloads/A Dataset of Aligned RGB and Multispectral UAV Ima/WeedyRice-RGBMS-DB/WeedyRice-RGBMS-DB/Multispectral"
pred_loader = make_weedy_rice_tif_loader(weedy_rice)
gt_loader = make_weedy_rice_tif_loader(weedy_rice)

evaluator = Evaluator(
    pred_loader=pred_loader,
    gt_loader=gt_loader,
    metric_calculator=metric_calc
)

# results = evaluator.evaluate()
save_results = "results/test_results.json"
# evaluator.save_results(save_results)

results = Evaluator.load_results(save_results)

for i, r in enumerate(results):
    print(f"{i} -- {r['id']}")
    
# for name, value in results[200].items():
#     print(f"{name}: {value}")