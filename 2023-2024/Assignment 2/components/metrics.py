from typing import Any

import numpy as np

from cinnamon_generic.components.metrics import LambdaMetric


class MultiLabelMetric(LambdaMetric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        scores = {}
        for label_name, label_pred in y_pred.items():
            label_pred = np.concatenate(label_pred)

            label_truth = y_true[label_name]
            label_truth = np.concatenate(label_truth)

            label_score = self.method(y_pred=label_pred,
                                      y_true=label_truth,
                                      **self.method_args)
            scores[label_name] = label_score

        avg_score = np.mean(list(scores.values()))
        return avg_score if not as_dict else {**scores, **{self.name: avg_score}}
