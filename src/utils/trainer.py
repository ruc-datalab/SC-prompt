import collections
from typing import Dict, List, Optional, NamedTuple, Union, Any, Tuple
import transformers.trainer_seq2seq
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_utils import PredictionOutput, speed_metrics
from torch.utils.data import DataLoader
from datasets.arrow_dataset import Dataset
from datasets.metric import Metric
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        os.remove(c_path)

class EvalPrediction(NamedTuple):
    predictions: List[str]
    label_ids: np.ndarray
    metas: List[dict]


class Seq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
        self,
        metric: Metric,
        *args,
        eval_examples: Optional[Dataset] = None,
        ignore_pad_token_for_loss: bool = True,
        target_with_db_id: bool = False,
        use_decomposition: bool = False,
        stage: str = 'structure',
        training_method: str = 'FT',
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.metric = metric
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.target_with_db_id = target_with_db_id
        self.use_decomposition = use_decomposition
        self.stage = stage
        self.training_method = training_method
        self.best_acc = 0

    def _compute_metrics(self, eval_prediction: EvalPrediction) -> dict:
        #utils.spider.py
        raise NotImplementedError()

    def _post_process_function(
        self, examples: Dataset, features: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        #utils.spider.py
        raise NotImplementedError()
    

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        eval_examples: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        start_time = time.time()
        
        print(f"epoch:{self.state.epoch}")
        if self.training_method == 'PT' and self.args.do_train:
            save_dir = self.args.output_dir
            os.makedirs(save_dir, exist_ok=True)

            del_file(save_dir)
            np.save(os.path.join(save_dir, f'head.npy'), self.model.prompt_head.cpu().detach().numpy())
            np.save(os.path.join(save_dir, f'tail.npy'), self.model.prompt_tail.cpu().detach().numpy())
            np.save(os.path.join(save_dir, f'mid1.npy'), self.model.prompt_mid1.cpu().detach().numpy())
            np.save(os.path.join(save_dir, f'mid2.npy'), self.model.prompt_mid2.cpu().detach().numpy())
            print(f"save embs to {save_dir}")
        
        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )
        if eval_examples is not None and eval_dataset is not None and self.compute_metrics is not None:
            eval_preds = self._post_process_function(
                eval_examples,
                eval_dataset,
                output.predictions,
                "eval_{}".format(self.state.epoch),
            )
            output.metrics.update(self.compute_metrics(eval_preds))

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)
        
        if output.metrics['eval_exact_match'] > self.best_acc and self.training_method != 'PT' and self.args.do_train:
            self.best_acc = output.metrics['eval_exact_match']
            save_dir = os.path.join(self.args.output_dir, "BEST_MODEL")
            os.makedirs(save_dir, exist_ok=True)
            del_file(save_dir)
            print(f"save model to {save_dir} acc={output.metrics['eval_exact_match']}")
            if self.training_method == 'FT':
                state_dict = self.model.state_dict()
                self.model.save_pretrained(save_dir, state_dict=state_dict)
                self.tokenizer.save_pretrained(save_dir)
                torch.save(self.args, os.path.join(save_dir, "training_args.bin"))
            else:
                state_dict = self.model.model.state_dict()
                self.model.model.save_pretrained(save_dir, state_dict=state_dict)
                self.tokenizer.save_pretrained(save_dir)
                torch.save(self.args, os.path.join(save_dir, "training_args.bin"))
                np.save(f'{save_dir}/head.npy', self.model.prompt_head.cpu().detach().numpy())
                np.save(f'{save_dir}/tail.npy', self.model.prompt_tail.cpu().detach().numpy())
                np.save(f'{save_dir}/mid1.npy', self.model.prompt_mid1.cpu().detach().numpy())
                np.save(f'{save_dir}/mid2.npy', self.model.prompt_mid2.cpu().detach().numpy())        
        
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)
        if self.args.do_train and self.training_method != 'PT':
            print("BEST EM result:", self.best_acc)
            print("BEST model save at:", os.path.join(self.args.output_dir, "BEST_MODEL"))

        return output.metrics

    def predict(
        self,
        test_dataset: Dataset,
        test_examples: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        max_time: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length
        self._max_time = max_time
        self._num_beams = num_beams

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if test_dataset is not None and not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output: PredictionOutput = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:
            # We might have removed columns from the dataset so we put them back.
            if isinstance(test_dataset, Dataset):
                test_dataset.set_format(
                    type=test_dataset.format["type"],
                    columns=list(test_dataset.features.keys()),
                )

            eval_preds = self._post_process_function(
                test_examples, test_dataset, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output
        
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
            "num_return_sequences": self.model.config.num_return_sequences,
        }

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]
        input_strs = self.tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=True)
        structures = []
        for input_str in input_strs:
            start_index = 59
            end_index = input_str.index(' | question:')
            structure = input_str[start_index:end_index]
            start_index = input_str.index('| database: ')+12
            end_index = input_str.index('. table: ')
            db_name = input_str[start_index:end_index]
            structures.append(db_name+' | '+structure)
        generated_tokens = self.model.generate(
            generation_inputs,
            structures=structures,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])
        
        if self.model.config.num_return_sequences > 1:
            preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            save_dir = os.path.join('/app', self.args.output_dir, "hypotheses.json")
            try:
                with open(save_dir) as f:
                    data = json.load(f)
            except:
                data = []
            new_data = []
            for idx in range(inputs['input_ids'].size(0)):
                new_data.append({'structure': structures[idx], 'topk_preds': []})
            for idx, pred in enumerate(preds):
                new_data[idx//self.model.config.num_beams]['topk_preds'].append(pred)
            data.extend(new_data)
            if data != None:
                with open(save_dir, 'w') as f:
                    json.dump(
                        data,
                        f,
                        indent=4,
                    )
            pick_list = []
            for idx in range(inputs['input_ids'].size(0)):
                pick_list.append(self.model.config.num_beams*idx)
            
            generated_tokens = torch.index_select(generated_tokens, 0, torch.tensor(pick_list).to(generated_tokens.device))
        
        with torch.no_grad():
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
            if has_labels:
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)