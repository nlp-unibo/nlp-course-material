from typing import Optional, Any

import numpy as np

from cinnamon_generic.components.metrics import LambdaMetric


class SequenceUnrolledF1Metric(LambdaMetric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        predictions = y_pred[self.output_key]
        ground_truth = y_true[self.output_key]

        unrl_predictions = []
        unrl_ground_truth = []

        for batch_pred, batch_truth in zip(predictions, ground_truth):
            for sample_pred, sample_truth in zip(batch_pred, batch_truth):
                # mask padding values
                valid_indexes = np.where(sample_truth != -1)[0]
                sample_pred = sample_pred[valid_indexes]
                sample_truth = sample_truth[valid_indexes]

                unrl_predictions.extend(sample_pred.tolist())
                unrl_ground_truth.extend(sample_truth.tolist())

        score = self.method(y_pred=unrl_predictions,
                            y_true=unrl_ground_truth,
                            **self.method_args)
        return score if not as_dict else {self.name: score}


class BaselineSequenceUnrolledF1Metric(LambdaMetric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        predictions = y_pred[self.output_key]
        predictions = np.array(predictions) if type(predictions) != np.ndarray else predictions

        ground_truth = y_true[self.output_key]
        ground_truth = np.array(ground_truth) if type(ground_truth) != np.ndarray else ground_truth

        score = self.method(y_pred=predictions,
                            y_true=ground_truth,
                            **self.method_args)
        return score if not as_dict else {self.name: score}


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


class BaselineSequenceF1Metric(LambdaMetric):

    def run(
            self,
            y_pred: Optional[Any] = None,
            y_true: Optional[Any] = None,
            as_dict: bool = False
    ) -> Any:
        predictions = y_pred[self.output_key]
        predictions = np.array(predictions) if type(predictions) != np.ndarray else predictions

        ground_truth = y_true[self.output_key]
        ground_truth = np.array(ground_truth) if type(ground_truth) != np.ndarray else ground_truth

        dialogue_id = y_pred['dialogue_id']
        dialogue_id = np.array(dialogue_id)

        scores = []
        for idx in np.unique(dialogue_id):
            dialogue_indexes = np.where(dialogue_id == idx)[0]
            dialogue_pred = predictions[dialogue_indexes]
            dialogued_truth = ground_truth[dialogue_indexes]

            score = self.method(y_pred=dialogue_pred,
                                y_true=dialogued_truth,
                                **self.method_args)
            scores.append(score)

        avg_score = np.mean(scores)
        return avg_score if not as_dict else {self.name: avg_score}