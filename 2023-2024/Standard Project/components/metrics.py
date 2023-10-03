from typing import Optional, Any

import numpy as np

from cinnamon_generic.components.metrics import LambdaMetric


class SequenceF1Metric(LambdaMetric):

    def run(
            self,
            y_pred: Optional[Any] = None,
            y_true: Optional[Any] = None,
            as_dict: bool = False
    ) -> Any:
        predictions = y_pred[self.output_key]
        ground_truth = y_true[self.output_key]

        scores = []
        for batch_pred, batch_truth in zip(predictions, ground_truth):
            for sample_pred, sample_truth in zip(batch_pred, batch_truth):
                # mask padding values
                valid_indexes = np.where(sample_truth != -1)[0]
                sample_pred = sample_pred[valid_indexes]
                sample_truth = sample_truth[valid_indexes]

                score = self.method(y_pred=sample_pred,
                                    y_true=sample_truth,
                                    **self.method_args)
                scores.append(score)

        avg_score = np.mean(scores)
        return avg_score if not as_dict else {self.name: avg_score}
