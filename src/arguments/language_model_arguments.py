from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class LanguageModelHandlerArguments:
    lm_model_name: str = field(
        default="Qwen2/Qwen2.5-3B-Instruct",
        metadata={
            "help": "The pretrained language model to use. Default is 'HuggingFaceTB/SmolLM-360M-Instruct'."
        },
    )
    lm_device: str = field(
        default="cuda",
        metadata={
            "help": "The device type on which the model will run. Default is 'cuda' for GPU acceleration."
        },
    )
    lm_torch_dtype: str = field(
        default="float16",
        metadata={
            "help": "The PyTorch data type for the model and input tensors. One of `float32` (full-precision), `float16` or `bfloat16` (both half-precision)."
        },
    )
    user_role: str = field(
        default="user",
        metadata={
            "help": "Role assigned to the user in the chat context. Default is 'user'."
        },
    )
    init_chat_role: str = field(
        default="system",
        metadata={
            "help": "Initial role for setting up the chat context. Default is 'system'."
        },
    )
    init_chat_prompt: str = field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words.",
        metadata={
            "help": "The initial chat prompt to establish context for the language model. Default is 'You are a helpful AI assistant.'"
        },
    )
    chat_size: int = field(
        default=4,
        metadata={
            "help": "Number of user-assistant conversation pairs to keep in chat history."
        },
    )
    lm_max_tokens: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of tokens to keep in chat context. None for no token-based limitation."
        },
    )
    lm_preserve_recent: int = field(
        default=1,
        metadata={
            "help": "Number of recent conversation pairs to always preserve from deletion."
        },
    )

    # Advanced generation parameters
    lm_gen_do_sample: bool = field(
        default=False,
        metadata={
            "help": "Whether to use sampling; set this to False for deterministic outputs. Default is False."
        },
    )
    lm_gen_max_new_tokens: int = field(
        default=128,
        metadata={
            "help": "Maximum number of new tokens to generate in a single completion. Default is 128."
        },
    )
    lm_gen_min_new_tokens: int = field(
        default=None,
        metadata={
            "help": "Minimum number of new tokens to generate in a single completion. Default is None."
        },
    )
    lm_gen_temperature: float = field(
        default=None,
        metadata={
            "help": "Controls the randomness of the output. Set to 0.0 for deterministic (repeatable) outputs. Default is None (use model default)."
        },
    )
    lm_gen_top_p: float = field(
        default=None,
        metadata={
            "help": "Controls the diversity of the output. Default is None (use model default)."
        },
    )
    lm_gen_top_k: int = field(
        default=None,
        metadata={
            "help": "Controls the diversity of the output. Default is None (use model default)."
        },
    )
    lm_gen_repetition_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "The parameter for repetition penalty. 1.0 means no penalty. Default is None (use model default)."
        },
    )
    lm_gen_length_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "Exponential penalty to the length that is used with beam-based generation. Default is None (use model default)."
        },
    )
    lm_gen_no_repeat_ngram_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set to int > 0, all ngrams of that size can only occur once. Default is None (use model default)."
        },
    )
    lm_gen_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams for beam search. 1 means no beam search. Default is None (use model default)."
        },
    )
    lm_gen_early_stopping: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to stop the beam search when at least num_beams sentences are finished per batch. Default is None (use model default)."
        },
    )
    lm_gen_use_cache: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether or not the model should use the past last key/values attentions to speed up decoding. Default is None (use model default)."
        },
    )
    lm_gen_typical_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to typical_p or higher are kept for generation. Default is None (use model default)."
        },
    )
    lm_gen_epsilon_cutoff: Optional[float] = field(
        default=None,
        metadata={
            "help": "If set to float strictly between 0 and 1, only tokens with a conditional probability greater than epsilon_cutoff will be sampled. Default is None (use model default)."
        },
    )
    lm_gen_eta_cutoff: Optional[float] = field(
        default=None,
        metadata={
            "help": "Eta sampling, a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either eta_cutoff or sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits))). Default is None (use model default)."
        },
    )
    lm_gen_diversity_penalty: Optional[float] = field(
        default=None,
        metadata={
            "help": "This value is subtracted from a beam's score if it generates a token same as any beam from prior step at a particular time. Default is None (use model default)."
        },
    )
    lm_gen_forced_bos_token_id: Optional[int] = field(
        default=None,
        metadata={
            "help": "The id of the token to force as the first generated token after the decoder_start_token_id. Default is None (disabled)."
        },
    )
    lm_gen_forced_eos_token_id: Optional[int] = field(
        default=None,
        metadata={
            "help": "The id of the token to force as the last generated token when max_length is reached. Default is None (disabled)."
        },
    )
    lm_gen_remove_invalid_values: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. Default is None (use model default)."
        },
    )
    lm_gen_exponential_decay_length_penalty: Optional[Tuple[int, float]] = field(
        default=None,
        metadata={
            "help": "Tuple of (start_index, decay_factor) for exponentially increasing length penalty. Default is None (disabled)."
        },
    )
    lm_gen_suppress_tokens: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "A list of tokens that will be suppressed at generation. Default is None."
        },
    )
    lm_gen_begin_suppress_tokens: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "A list of tokens that will be suppressed at the beginning of the generation. Default is None."
        },
    )
    lm_gen_forced_decoder_ids: Optional[List[Tuple[int, int]]] = field(
        default=None,
        metadata={
            "help": "A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. Default is None."
        },
    )
