from copy import deepcopy
from typing import Optional, Union, Any, Callable, AsyncContextManager, List, Dict, Iterable, List, Tuple, Union
import inspect
import warnings
from dataclasses import dataclass, field
import collections
import asyncio
import sys
import subprocess
import warnings
from tenacity import retry, wait_random_exponential, stop_after_delay, before_sleep_log
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList, Constraint, BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.configuration_utils import PretrainedConfig
from transformers.generation_utils import GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput
from transformers.generation_logits_process import LogitsProcessor
from transformers.file_utils import copy_func
from transformers.models.auto.auto_factory import _get_model_class
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.models.auto import AutoModelForSeq2SeqLM
from transformers.pytorch_utils import torch_int_div
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import torch.nn as nn
logger = logging.getLogger(__name__)

try:
    from picard.clients import Picard
    from picard.types import (
        FeedException,
        FeedTimeoutFailure,
        FeedParseFailure,
        FeedPartialSuccess,
        FeedCompleteSuccess,
        SQLSchema,
        RegisterSQLSchemaException,
        Mode,
        ColumnType,
    )
    from thrift.py3.client import get_client
    from thrift.py3.common import Protocol
    from thrift.py3.exceptions import TransportError

    picard_available = True
except:
    logger.warning("Picard is not available.")
    Picard = Any
    SQLSchema = Any
    RegisterSQLSchemaFail = Any
    ColumnType = Any
    picard_available = False


@dataclass
class PicardArguments:
    """
    Arguments pertaining to Picard.
    """

    use_picard: bool = field(default=True, metadata={"help": "Whether or not to use Picard."})
    launch_picard: bool = field(
        default=True,
        metadata={"help": "Whether or not to launch Picard. If ``False``, an already running Picard is used."},
    )
    picard_host: str = field(default="localhost", metadata={"help": "The host name for Picard."})
    picard_port: int = field(default=9090, metadata={"help": "The port number for Picard."})
    picard_mode: str = field(
        default="parse_with_guards",
        metadata={
            "help": "Picard mode. Choose between ``lex``, ``parse_without_guards``, ``parse_with_guards``, and ``parse_with_guards_and_type_checking."
        },
    )
    picard_schedule: str = field(
        default="incremental",
        metadata={"help": "Picard schedule. Choose between ``incremental`` and ``finalizing``."},
    )
    picard_max_tokens_to_check: int = field(
        default=2,
        metadata={"help": "The maximum number of tokens to check with Picard."},
    )

    def __post_init__(self):
        self.use_picard = picard_available and self.use_picard
        self.launch_picard = self.use_picard and self.launch_picard


class PicardLauncher(subprocess.Popen):
    def __init__(self) -> None:
        try:
            super().__init__(["picard"])
        except FileNotFoundError:
            with subprocess.Popen(
                ["cabal", "install", "--overwrite-policy=always", "--install-method=copy", "exe:picard"]
            ) as picard_build_pid:
                picard_build_pid.wait(timeout=1000)
            super().__init__(["picard"])
        time.sleep(1)

    def __exit__(self, exc_type, value, traceback):
        self.kill()
        super().__exit__(exc_type, value, traceback)

    def __del__(self, _maxsize=sys.maxsize, _warn=warnings.warn):
        self.kill()
        super().__del__(_maxsize, _warn)


def with_picard(
    model_cls: AutoModelForSeq2SeqLM,
    picard_args: PicardArguments,
    tokenizer: PreTrainedTokenizerFast,
    schemas: Optional[Dict[str, dict]] = None,
    stage: str = 'content',
):
    #db_id | db_info: db_table_names, db_column_names, db_column_types, db_primary_keys, db_foreign_keys
    schema_cache: Dict[str, dict] = deepcopy(schemas) if schemas is not None else dict()

    def get_picard_client() -> AsyncContextManager[Picard]:
        return get_client(
            Picard,
            host=picard_args.picard_host,
            port=picard_args.picard_port,
            timeout=1,
            protocol=Protocol.BINARY,
        )

    async def _init_picard() -> None:
        async with get_picard_client() as client:
            for db_id, db_info in schema_cache.items():
                await _register_schema(db_id=db_id, db_info=db_info, picard_client=client)
            await _register_tokenizer(picard_client=client)

    async def _register_schema(db_id: str, db_info: dict, picard_client: Picard) -> None:
        sql_schema = get_picard_schema(**db_info)
        try:
            await picard_client.registerSQLSchema(db_id, sql_schema)
        except RegisterSQLSchemaException:
            # db already registered
            logger.debug(f"schema already registered: {db_id}")
            pass

    async def _register_schema_without_client(db_id: str, db_info: dict) -> None:
        async with get_picard_client() as client:
            await _register_schema(db_id=db_id, db_info=db_info, picard_client=client)

    async def _register_tokenizer(picard_client: Picard) -> None:
        assert isinstance(tokenizer, PreTrainedTokenizerFast)
        json_str = tokenizer.backend_tokenizer.to_str(pretty=False)
        await picard_client.registerTokenizer(json_str)

    def _add_schema(db_id: str, db_info: dict) -> None:
        if not db_id in schema_cache:
            schema_cache[db_id] = deepcopy(db_info)
            asyncio.run(_register_schema_without_client(db_id=db_id, db_info=db_info), debug=False)
        else:
            assert db_info == schema_cache[db_id], "unexpected schema change"

    @torch.no_grad()
    def _generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        structures: Optional[List[str]] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from `inputs`, all the arguments below will default to the value of the attribute of the same name inside
        the [`PretrainedConfig`] of the model. The default values indicated are the default values of those config.

        Most of these parameters are explained in more detail in [this blog
        post](https://huggingface.co/blog/how-to-generate).

        Parameters:
            inputs (`torch.Tensor` of shape `(batch_size, sequence_length)`, `(batch_size, sequence_length,
            feature_dim)` or `(batch_size, num_channels, height, width)`, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            max_length (`int`, *optional*, defaults to `model.config.max_length`):
                The maximum length of the sequence to be generated.
            max_new_tokens (`int`, *optional*, defaults to None):
                The maximum numbers of tokens to generate, ignore the current number of tokens. Use either
                `max_new_tokens` or `max_length` but not both, they serve the same purpose.
            min_length (`int`, *optional*, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (`bool`, *optional*, defaults to `False`):
                Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (`int`, *optional*, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (`float`, *optional*, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher
                are kept for generation.
            repetition_penalty (`float`, *optional*, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See [this
                paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            bos_token_id (`int`, *optional*):
                The id of the *beginning-of-sequence* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            length_penalty (`float`, *optional*, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            encoder_no_repeat_ngram_size (`int`, *optional*, defaults to 0):
                If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the
                `decoder_input_ids`.
            bad_words_ids(`List[List[int]]`, *optional*):
                List of token ids that are not allowed to be generated. In order to get the token ids of the words that
                should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True,
                add_special_tokens=False).input_ids`.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            max_time(`float`, *optional*, defaults to None):
                The maximum amount of time you allow the computation to run for in seconds. generation will still
                finish the current pass after allocated time has been passed.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values are in `[0, 1]`, 1 for tokens
                that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same shape
                as `input_ids` that masks the pad token. [What are attention masks?](../glossary#attention-mask)
            decoder_start_token_id (`int`, *optional*):
                If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.
            use_cache: (`bool`, *optional*, defaults to `True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            num_beam_groups (`int`, *optional*, defaults to 1):
                Number of groups to divide `num_beams` into in order to ensure diversity among different groups of
                beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.
            diversity_penalty (`float`, *optional*, defaults to 0.0):
                This value is subtracted from a beam's score if it generates a token same as any beam from other group
                at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is
                enabled.
            prefix_allowed_tokens_fn: (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
                `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
                on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
                for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                 Custom logits processors that complement the default logits processors built from arguments and a
                 model's config. If a logit processor is passed that is already created with the arguments or a model's
                 config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                 Custom stopping criteria that complement the default stopping criteria built from arguments and a
                 model's config. If a stopping criteria is passed that is already created with the arguments or a
                 model's config an error is thrown. This feature is intended for advanced users.
            constraints (`List[Constraint]`, *optional*):
                 Custom constraints that can be added to the generation to ensure that the output will contain the use
                 of certain tokens as defined by `Constraint` objects, in the most sensible way possible.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            forced_bos_token_id (`int`, *optional*):
                The id of the token to force as the first generated token after the `decoder_start_token_id`. Useful
                for multilingual models like [mBART](../model_doc/mbart) where the first generated token needs to be
                the target language token.
            forced_eos_token_id (`int`, *optional*):
                The id of the token to force as the last generated token when `max_length` is reached.
            remove_invalid_values (`bool`, *optional*):
                Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to
                crash. Note that using `remove_invalid_values` can slow down generation.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If the model
                is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs
                should be prefixed with *decoder_*.

        Return:
            [`~file_utils.ModelOutput`] or `torch.LongTensor`: A [`~file_utils.ModelOutput`] (if
            `return_dict_in_generate=True` or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchDecoderOnlyOutput`],
                    - [`~generation_utils.SampleDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSearchDecoderOnlyOutput`],
                    - [`~generation_utils.BeamSampleDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~file_utils.ModelOutput`] types are:

                    - [`~generation_utils.GreedySearchEncoderDecoderOutput`],
                    - [`~generation_utils.SampleEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSearchEncoderDecoderOutput`],
                    - [`~generation_utils.BeamSampleEncoderDecoderOutput`]

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> # do greedy decoding without providing a prompt
        >>> outputs = model.generate(max_length=40)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
        >>> document = (
        ...     "at least two people were killed in a suspected bomb attack on a passenger bus "
        ...     "in the strife-torn southern philippines on monday , the military said."
        ... )
        >>> # encode input context
        >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
        >>> # generate 3 independent sequences using beam search decoding (5 beams)
        >>> # with T5 encoder-decoder model conditioned on short news article.
        >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        >>> input_context = "The dog"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> # generate 3 candidates using sampling
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
        >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
        >>> # "Legal" is one of the control codes for ctrl
        >>> input_context = "Legal My neighbor is"
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> input_context = "My cute dog"
        >>> # get tokens of words that should not be generated
        >>> bad_words_ids = tokenizer(
        ...     ["idiot", "stupid", "shut up"], add_prefix_space=True, add_special_tokens=False
        >>> ).input_ids
        >>> # encode input context
        >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
        >>> # generate sequences without allowing bad_words to be generated
        >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
        >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        ```"""
        # 1. Set generation parameters if not already defined
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )

        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        if eos_token_id is None and hasattr(self.config, "decoder"):
            eos_token_id = self.config.decoder.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id

        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # 2. Define model inputs
        # inputs_tensor has to be defined
        # model_input_name is defined if model-specific keyword input is passed
        # otherwise model_input_name is None
        # all model-specific keyword inputs are removed from `model_kwargs`
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]

        # 3. Define other model kwargs
        model_kwargs["output_attentions"] = output_attentions
        model_kwargs["output_hidden_states"] = output_hidden_states
        model_kwargs["use_cache"] = use_cache

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs

        if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, pad_token_id, eos_token_id
            )

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created
            # and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name
            )

        # 4. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids = self._prepare_decoder_input_ids_for_generation(
                batch_size,
                decoder_start_token_id=decoder_start_token_id,
                bos_token_id=bos_token_id,
                model_kwargs=model_kwargs,
            )
        else:
            # if decoder-only then inputs_tensor has to be `input_ids`
            input_ids = inputs_tensor

        # 5. Prepare `max_length` depending on other stopping criteria
        # if `max_new_tokens` is passed, but not `max_length` -> set `max_length = max_new_tokens`
        if max_length is None and max_new_tokens is not None:
            max_length = max_new_tokens + input_ids.shape[-1]
        elif max_length is not None and max_new_tokens is not None:
            # Both are set, this is odd, raise a warning
            warnings.warn(
                "Both `max_length` and `max_new_tokens` have been set "
                f"but they serve the same purpose. `max_length` {max_length} "
                f"will take priority over `max_new_tokens` {max_new_tokens}.",
                UserWarning,
            )
        # default to config if still None
        max_length = max_length if max_length is not None else self.config.max_length

        if input_ids.shape[-1] >= max_length:
            input_ids_string = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            logger.warning(
                f"Input length of {input_ids_string} is {input_ids.shape[-1]}, but ``max_length`` is set to {max_length}. "
                "This can lead to unexpected behavior. You should consider increasing ``config.max_length`` or ``max_length``."
            )

        # 6. determine generation mode
        is_constraint_gen_mode = constraints is not None
        is_greedy_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_sample_gen_mode = (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        is_beam_gen_mode = (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and constraints is None
        is_beam_sample_gen_mode = (
            (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and constraints is None
        )
        is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and constraints is None

        if num_beam_groups > num_beams:
            raise ValueError("`num_beam_groups` has to be smaller or equal to `num_beams`")
        if is_group_beam_gen_mode and do_sample is True:
            raise ValueError(
                "Diverse beam search cannot be used in sampling mode. Make sure that `do_sample` is set to `False`."
            )

        # 7. prepare distribution pre_processing samplers
        logits_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=inputs_tensor,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
            logits_processor=logits_processor,
        )
        logits_processor.append(
            PicardLogitsProcessor(
                eos_token_id=eos_token_id,
                get_client=get_picard_client,
                max_tokens_to_check=picard_args.picard_max_tokens_to_check,
                mode=picard_args.picard_mode,
                schedule=picard_args.picard_schedule,
                tokenizer=tokenizer,
                stage=stage,
            )
        )

        # 8. prepare stopping criteria
        stopping_criteria = self._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        # 9. go into different generation modes
        if is_greedy_gen_mode:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )

            # 10. run greedy search
            return self.greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams
            )

            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample
            return self.sample(
                input_ids,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                structures=structures,
                **model_kwargs,
            )

        elif is_beam_sample_gen_mode:
            # 10. prepare logits warper
            logits_warper = self._get_logits_warper(
                top_k=top_k, top_p=top_p, typical_p=typical_p, temperature=temperature, num_beams=num_beams
            )

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size * num_return_sequences,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
            )

            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids,
                expand_size=num_beams * num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 13. run beam sample
            return self.beam_sample(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_group_beam_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if num_beams % num_beam_groups != 0:
                raise ValueError("`num_beams` should be divisible by `num_beam_groups` for group beam search.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            # 10. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                max_length=stopping_criteria.max_length,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
                num_beam_groups=num_beam_groups,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif is_constraint_gen_mode:
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")

            if stopping_criteria.max_length is None:
                raise ValueError("`max_length` needs to be a stopping_criteria for now.")

            if num_beams <= 1:
                raise ValueError("`num_beams` needs to be greater than 1 for constrained genertation.")

            if do_sample:
                raise ValueError("`do_sample` needs to be false for constrained generation.")

            if num_beam_groups is not None and num_beam_groups > 1:
                raise ValueError("`num_beam_groups` not supported yet for constrained generation.")

            # 10. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=constraints,
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids, expand_size=num_beams, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs
            )
            # 12. run beam search
            return self.constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                output_scores=output_scores,
                return_dict_in_generate=return_dict_in_generate,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        structures: Optional[List[str]] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using beam search decoding.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores, structures=structures)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]


    class _PicardAutoModelClass(model_cls):
        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
            config = kwargs.pop("config", None)
            kwargs["_from_auto"] = True
            if not isinstance(config, PretrainedConfig):
                config, kwargs = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
                )

            if type(config) in cls._model_mapping.keys():
                model_class = _get_model_class(config, cls._model_mapping)
                generate = copy_func(_generate)
                beam_search = copy_func(_beam_search)
                generate.__doc__ = model_class.generate.__doc__
                model_class.old_generate = copy_func(model_class.generate)
                model_class.generate = generate
                model_class.beam_search = beam_search
                model_class.add_schema = staticmethod(copy_func(_add_schema))
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
            raise ValueError(
                f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
                f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
            )

    asyncio.run(_init_picard(), debug=False)

    return _PicardAutoModelClass


class PicardLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        eos_token_id: int,
        tokenizer: PreTrainedTokenizerFast,
        get_client: Callable[[], AsyncContextManager[Picard]],
        filter_value: float = -float("Inf"),
        max_tokens_to_check: int = 1,
        mode: str = "parse_with_guards",
        schedule: str = "incremental",
        stage: str = "content",
    ):
        self.eos_token_id = eos_token_id
        self.get_client = get_client
        self.filter_value = filter_value
        self.max_tokens_to_check = max_tokens_to_check
        self.mode = mode
        self.schedule = schedule
        self.num_beams = None
        self.tokenizer = tokenizer
        self.stage = stage
        self.Structure_Vocab = {'where', '<pad>', 'from', 'by', '</s>', 'order', 'max', 'limit', '▁before', '▁after', '▁*', '▁order', 'col', 'tab', '],', '▁between', '▁sum', '▁not', '▁group', ']', '=', '▁where', '▁union', '▁=', '▁and', 'c', '▁count', '▁min', '▁des', '▁limit', '▁>', '▁having', ' <', '▁like', '▁as', '▁select', '▁[', '[', ')', '),', '▁from', '▁', 'distin', 'g', '▁by', '▁intersect', ' <=', 'val', '▁in', 'v', '!', '▁(', '▁or', '▁except', '▁distinct', '(', 't', 'a', '▁max'}

    async def _feed(self, client: Picard, input_ids: List[int], token: int) -> bool:
        if self.mode == "lex":
            mode = Mode.LEXING
        elif self.mode == "parse_without_guards":
            mode = Mode.PARSING_WITHOUT_GUARDS
        elif self.mode == "parse" or self.mode == "parse_with_guards":
            mode = Mode.PARSING_WITH_GUARDS
        elif self.mode == "parse_with_guards_and_type_checking":
            mode = Mode.PARSING_WITH_GUARDS_AND_TYPE_CHECKING
        else:
            raise ValueError("unexpected picard mode")

        try:
            res = await client.feed(input_ids, token, mode)
        except FeedException as e:
            logger.error(f"unexpected feed error: {e}, input ids were: {input_ids}, token was: {token}")
            raise e
        except TransportError as e:
            logger.error(f"unexpected transport error: {e}, input ids were: {input_ids}, token was: {token}")
            raise e
        if isinstance(res.value, FeedTimeoutFailure):
            logger.warning(f"timeout failure: {input_ids + [token]}")
            return False
        elif isinstance(res.value, FeedParseFailure):
            logger.debug(f"parsing failure: {input_ids + [token]}")
            return False
        elif isinstance(res.value, FeedPartialSuccess):
            logger.debug(f"parsing partial: {input_ids + [token]}")
            return True
        elif isinstance(res.value, FeedCompleteSuccess):
            logger.info(f"parsing success: {input_ids + [token]}")
            return True
        else:
            # unexpected parsing result
            raise ValueError("unexpected picard parsing result")

    async def _check_token(self, client: Picard, input_ids: List[int], token: int) -> bool:
        if self.schedule == "incremental":
            # check at every step
            return await self._feed(client=client, input_ids=input_ids, token=token)
        elif self.schedule == "finalizing":
            # only check when decoded string is finalized
            if token == self.eos_token_id:
                return await self._feed(client=client, input_ids=input_ids, token=token)
            else:
                return True
        else:
            raise ValueError("unexpected picard schedule")

    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_delay(600),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _mask(
        self,
        client: Picard,
        indices_to_remove: torch.Tensor,
        batch_idx: int,
        input_ids_batch: torch.Tensor,
        top_token: torch.Tensor,
    ) -> None:
        res = await self._check_token(client=client, input_ids=input_ids_batch.tolist(), token=top_token.item())
        if not res:
            indices_to_remove[batch_idx, top_token] = True

    async def _mask_top_k(
        self,
        indices_to_remove: torch.Tensor,
        input_ids: torch.Tensor,
        top_tokens: torch.Tensor,
    ) -> None:
        async with self.get_client() as client:
            futures = [
                self._mask(
                    client=client,
                    indices_to_remove=indices_to_remove,
                    batch_idx=batch_idx,
                    input_ids_batch=input_ids_batch,
                    top_token=top_token,
                )
                for batch_idx, (input_ids_batch, top_token_batch) in enumerate(zip(input_ids, top_tokens))
                for top_token in top_token_batch
            ]
            for f in asyncio.as_completed(futures):
                await f


    @retry(
        wait=wait_random_exponential(multiplier=1, max=60),
        stop=stop_after_delay(600),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _batch_mask_top_k(
        self,
        indices_to_remove: torch.Tensor,
        input_ids: torch.Tensor,
        top_tokens: torch.Tensor,
    ) -> None:
        if self.mode == "lex":
            mode = Mode.LEXING
        elif self.mode == "parse_without_guards":
            mode = Mode.PARSING_WITHOUT_GUARDS
        elif self.mode == "parse" or self.mode == "parse_with_guards":
            mode = Mode.PARSING_WITH_GUARDS
        elif self.mode == "parse_with_guards_and_type_checking":
            mode = Mode.PARSING_WITH_GUARDS_AND_TYPE_CHECKING
        else:
            raise ValueError("unexpected picard mode")

        async with self.get_client() as client:
            try:
                res = await client.batchFeed(input_ids.tolist(), top_tokens.tolist(), mode)
            except FeedException as e:
                logger.error(
                    f"unexpected feed error: {e}, input ids were: {input_ids.tolist()}, top tokens were: {top_tokens.tolist()}"
                )
                raise e
            except TransportError as e:
                logger.error(
                    f"unexpected transport error: {e}, input ids were: {input_ids.tolist()}, top tokens were: {top_tokens.tolist()}"
                )
                raise e
        for r in res:
            if isinstance(r.feedResult.value, FeedTimeoutFailure):
                logger.warning(f"timeout failure: {input_ids[r.batchId].tolist() + [r.topToken]}")
                indices_to_remove[r.batchId, r.topToken] = True
            elif isinstance(r.feedResult.value, FeedParseFailure):
                logger.debug(f"parsing failure: {input_ids[r.batchId].tolist() + [r.topToken]}")
                indices_to_remove[r.batchId, r.topToken] = True
            elif isinstance(r.feedResult.value, FeedPartialSuccess):
                logger.debug(f"parsing partial: {input_ids[r.batchId].tolist() + [r.topToken]}")
            elif isinstance(r.feedResult.value, FeedCompleteSuccess):
                logger.info(f"parsing success: {input_ids[r.batchId].tolist() + [r.topToken]}")
            else:
                # unexpected parsing result
                raise ValueError("unexpected picard parsing result")

    @torch.no_grad()
    def structure_constrained(
        self,
        input_ids: torch.Tensor,
        indices_to_remove: torch.Tensor,
        top_tokens: torch.Tensor,
    ) -> torch.Tensor:

        # predictions = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        for idx0 in range(top_tokens.size(0)):
            toks = self.tokenizer.convert_ids_to_tokens(top_tokens[idx0])
            for idx1, tok in enumerate(toks): # out of the keywords vocab, remove this token
                if tok not in self.Structure_Vocab:
                    indices_to_remove[idx0, top_tokens[idx0][idx1]] = True
        return indices_to_remove

    @torch.no_grad()
    def content_constrained(
        self,
        input_ids: torch.Tensor,
        indices_to_remove: torch.Tensor,
        structures: List[str],
        top_tokens: torch.Tensor,
    ) -> torch.Tensor:

        if not self.num_beams:
            self.num_beams = input_ids.size(0) // len(structures)
        
        predictions = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        extend_preds = []
        extend_input_ids = []
        need_check_flags = []

        for idx0, pred in enumerate(predictions):
            #print("pred:", pred)
            top_token = self.tokenizer.convert_ids_to_tokens(top_tokens[idx0])
            #print(f"top_tok: {top_token}")
            if 't' in top_token:
                need_check_flags.append(False)
                continue
            if '[table]' in pred or '[Col' in pred or '[value]' in pred or '[col)' in pred:
                indices_to_remove[idx0, :] = True
                indices_to_remove[idx0, self.eos_token_id] = False
            structure = structures[idx0 // self.num_beams]
            if (pred.count('[') > 0) and (pred.count('[') == pred.count(']')) and (784 not in top_tokens[idx0]) and (1 not in top_tokens[idx0]):
                flag = True
                s_idx = 0
                last_s_idx = 0
                while 1:
                    try:
                        idx1 = pred.index('[')
                    except:
                        break
                    try:
                        s_idx = structure[last_s_idx:].index('[') + last_s_idx
                    except:
                        flag = False
                        break
                    if pred[idx1:idx1+5] == structure[s_idx:s_idx+5]:
                        pred = pred[:max(idx1-1,0)] + structure[last_s_idx:s_idx] + pred[idx1+6:]
                        last_s_idx = s_idx + 5
                    else:
                        indices_to_remove[idx0, :] = True
                        indices_to_remove[idx0, self.eos_token_id] = False
                        flag = False
                        break
                if flag:
                    extend_preds.append(pred)
                    extend_input_ids.append([0] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(pred)))
                need_check_flags.append(flag)
            elif pred.count('[') < pred.count(']') or (len(pred) >= 5 and pred[:5] == '[col '):
                indices_to_remove[idx0, :] = True
                indices_to_remove[idx0, self.eos_token_id] = False           
                need_check_flags.append(False)    
            else:
                need_check_flags.append(False)
        
        if len(extend_preds) > 0:
            max_len = max([len(x) for x in extend_input_ids])
            extend_input_ids = [[0]*(max_len-len(x)) + x for x in extend_input_ids]
            extend_input_ids = torch.tensor(extend_input_ids)
            check_indices_to_remove = indices_to_remove[need_check_flags,:]
            check_top_tokens = top_tokens[need_check_flags,:]
            asyncio.run(
                self._batch_mask_top_k(
                    indices_to_remove=check_indices_to_remove,
                    input_ids=extend_input_ids,
                    top_tokens=check_top_tokens,
                )
                if self.schedule == "incremental"
                else self._mask_top_k(
                    indices_to_remove=indices_to_remove,
                    input_ids=input_ids,
                    top_tokens=top_tokens,
                ),
                debug=False,
            )
            indices_to_remove[need_check_flags,:] = check_indices_to_remove
        return indices_to_remove
    

    @torch.no_grad()
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, structures: List[str]) -> torch.FloatTensor:
        top_k = min(max(1, self.max_tokens_to_check), scores.size(-1))  # Safety check
        #print(top_k)
        top_scores, top_tokens = torch.topk(scores, top_k)
        
        for i in range(top_scores.size(0)):
            if top_scores[i][1] == -float("Inf"):
                if top_scores[i][0] == -float("Inf"):
                    top_scores[i][1] = 1.
                else:
                    top_scores[i][1] = top_scores[i][0]
        
        # Remove all tokens with a probability less than the last token of the top-k
        lowest_top_k_scores = top_scores[..., -1, None]
        del top_scores
        indices_to_remove = scores < lowest_top_k_scores
        del lowest_top_k_scores
        # Do not mask the EOS token because otherwise production can continue indefinitely if all other tokens are masked
        indices_to_remove[:, self.eos_token_id] = False
        if self.stage == 'content':
            # constrained decoding for content-stage
            indices_to_remove = self.content_constrained(input_ids, indices_to_remove, structures, top_tokens)
        elif self.stage == 'structure':
            # constrained decoding for structure-stage
            indices_to_remove = self.structure_constrained(input_ids, indices_to_remove, top_tokens) 
        del top_tokens
        # set the probability of the invalid token to -infty
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        del indices_to_remove
        return scores


def _get_picard_column_type(column_type: str) -> ColumnType:
    if column_type == "text":
        return ColumnType.TEXT
    elif column_type == "number":
        return ColumnType.NUMBER
    elif column_type == "time":
        return ColumnType.TIME
    elif column_type == "boolean":
        return ColumnType.BOOLEAN
    elif column_type == "others":
        return ColumnType.OTHERS
    else:
        raise ValueError(f"unexpected column type {column_type}")


def get_picard_schema(
    db_table_names: List[str],
    db_column_names: Dict[str, Union[List[str], List[int]]],
    db_column_types: List[str],
    db_primary_keys: Dict[str, List[int]],
    db_foreign_keys: Dict[str, List[int]],
) -> SQLSchema:
    star_id = next((c_id for c_id, c_name in enumerate(db_column_names["column_name"]) if c_name == "*"))
    column_names = dict(
        (str(c_id), c_name) for c_id, c_name in enumerate(db_column_names["column_name"]) if c_id != star_id
    )
    column_types = dict(
        (str(c_id), _get_picard_column_type(c_type)) for c_id, c_type in enumerate(db_column_types) if c_id != star_id
    )
    table_names = dict((str(t_id), t_name) for t_id, t_name in enumerate(db_table_names))
    column_to_table = dict(
        (str(c_id), str(t_id))
        for c_id, (t_id, _c_name) in enumerate(zip(db_column_names["table_id"], db_column_names["column_name"]))
        if c_id != star_id
    )
    table_to_columns = collections.defaultdict(list)
    for c_id, (t_id, _c_name) in enumerate(zip(db_column_names["table_id"], db_column_names["column_name"])):
        if c_id == star_id:
            continue
        table_to_columns[str(t_id)].append(str(c_id))
    foreign_keys = dict(
        (str(c_id), str(other_c_id))
        for c_id, other_c_id in zip(db_foreign_keys["column_id"], db_foreign_keys["other_column_id"])
        if c_id != star_id and other_c_id != star_id
    )
    primary_keys = [str(c_id) for c_id in db_primary_keys["column_id"] if c_id != star_id]
    return SQLSchema(
        columnNames=column_names,
        columnTypes=column_types,
        tableNames=table_names,
        columnToTable=column_to_table,
        tableToColumns=table_to_columns,
        foreignKeys=foreign_keys,
        primaryKeys=primary_keys,
    )
