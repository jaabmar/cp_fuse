import gc
import os
import time
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
)
from transformers.generation.stopping_criteria import (
    EosTokenCriteria,
    MaxLengthCriteria,
    StoppingCriteriaList,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CPModel:
    """CP-Fuse Model for text generation using two language models.

    This class combines two language models and performs text generation by optimally
    combining the outputs of the two models at each generation step in a way that no 
    memorized training data is reproduced.

    Attributes:
        model1 (torch.nn.Module): The first language model.
        model2 (torch.nn.Module): The second language model.
        grid_size (int): The size of the grid for optimization.
        verbose (bool): If True, prints detailed logs.
        fixed_coef (Optional[float]): Fixed coefficients for the model combination.
        step_solve (int): The number of steps between solving the optimization problem.
        device (torch.device): The device on which models are located.
    """

    def __init__(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        grid_size: int = 10,
        verbose: bool = False,
        fixed_coef: Optional[float] = None,
        step_solve: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the CPModel with two language models.

        Args:
            model1 (torch.nn.Module): The first language model.
            model2 (torch.nn.Module): The second language model.
            grid_size (int, optional): The size of the grid for optimization. Defaults to 10.
            verbose (bool, optional): If True, prints detailed logs. Defaults to False.
            fixed_coef (Optional[float]): Fixed coefficients for the model combination. Defaults to None.
            step_solve (int, optional): The number of steps between solving the optimization problem. Defaults to 1.
        """
        self.config = model1.config
        self.model1 = model1
        self.model2 = model2
        self.grid_size = grid_size
        self.verbose = verbose
        self.fixed_coef = fixed_coef
        self.step_solve = step_solve

        # Device management
        self.device = device or next(self.model1.parameters()).device
        if next(self.model1.parameters()).device != self.device:
            self.model1.to(self.device)
            print(f"[INFO] Model 1 moved to device: {self.device}")
        if next(self.model2.parameters()).device != self.device:
            self.model2.to(self.device)
            print(f"[INFO] Model 2 moved to device: {self.device}")
            
        # Ensure models are in eval mode
        self.model1.eval()
        self.model2.eval()
            

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
        attention_mask: Optional[torch.Tensor] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        grid_size: Optional[int] = None,
        parallelize: bool = False,
        **model_kwargs: Any,
    ) -> GenerateDecoderOnlyOutput:
        """Generate text sequences using the combined models.

        Args:
            input_ids (torch.Tensor): Input token IDs of shape (batch_size, sequence_length).
            generation_config (GenerationConfig): Configuration for generation.
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            stopping_criteria (Optional[StoppingCriteriaList], optional): Criteria to stop generation. Defaults to None.
            logits_warper (Optional[LogitsProcessorList], optional): Logits processor for modifying logits. Defaults to None.
            grid_size (Optional[int], optional): Grid size for optimization. Defaults to None.
            parallelize (bool, optional): If True, performs parallel decoding for models. Defaults to False.
            **model_kwargs: Additional keyword arguments passed to the model.
            
        Returns:
            GenerateDecoderOnlyOutput: The generated sequences and optional logits.
        """
        # Use instance grid_size if not provided
        grid_size = grid_size if grid_size is not None else self.grid_size

        # Input validation
        self._validate_generate_inputs(input_ids, generation_config)

        # Move input_ids and attention_mask to the correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Prepare attention masks and special tokens
        attention_mask = self._prepare_attention_mask(input_ids, attention_mask, generation_config)
        pad_token_id = self._prepare_pad_token_id(generation_config)
        eos_token_id = generation_config.eos_token_id

        # Stopping criteria
        stopping_criteria = self._prepare_stopping_criteria(stopping_criteria, generation_config)

        # Logits warper
        logits_warper = self._prepare_logits_warper(logits_warper, generation_config)

        # Generate outputs
        output = self._decode(
            input_ids=input_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_warper=logits_warper,
            grid_size=grid_size,
            step_solve=self.step_solve,
            do_sample=generation_config.do_sample,
            **model_kwargs,
        )

        return output

    def _validate_generate_inputs(
        self,
        input_ids: torch.Tensor,
        generation_config: GenerationConfig,
    ) -> None:
        """Validate inputs for the generate method.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            generation_config (GenerationConfig): Configuration for generation.

        Raises:
            ValueError: If any of the validation checks fail.
        """
        if not (isinstance(generation_config.max_length, int) and generation_config.max_length > 0):
            raise ValueError("`max_length` should be a strictly positive integer.")

        if hasattr(generation_config, "max_new_tokens") and generation_config.max_new_tokens is not None:
            if not (isinstance(generation_config.max_new_tokens, int) and generation_config.max_new_tokens > 0):
                raise ValueError("`max_new_tokens` should be a strictly positive integer.")
            # Calculate the max length based on max_new_tokens and the input length
            max_length = input_ids.shape[1] + generation_config.max_new_tokens
            # Overwrite max_length with max_new_tokens value if it's smaller
            generation_config.max_length = max_length
        else:
            if generation_config.max_length is None:
                raise ValueError("`max_length` must be defined if `max_new_tokens` is not provided.")

        if generation_config.do_sample:
            if generation_config.temperature <= 0:
                raise ValueError("`temperature` should be positive for sampling decoding.")

        if generation_config.num_return_sequences != 1:
            raise ValueError("Only one generation is supported.")

        if generation_config.num_beams != 1:
            raise ValueError("Beam search is not supported.")

        if generation_config.pad_token_id != self.model1.config.pad_token_id:
            raise ValueError("Mismatch pad token with model 1.")

        if generation_config.pad_token_id != self.model2.config.pad_token_id:
            raise ValueError("Mismatch pad token with model 2.")

        if generation_config.eos_token_id != self.model1.config.eos_token_id:
            raise ValueError("Mismatch eos token with model 1.")

        if generation_config.eos_token_id != self.model2.config.eos_token_id:
            raise ValueError("Mismatch eos token with model 2.")

        if input_ids is None:
            raise ValueError("input_ids cannot be None.")

        if input_ids.dim() != 2:
            raise ValueError("Input prompt should be of shape (batch_size, sequence length).")

        if self.model1.config.vocab_size != self.model2.config.vocab_size:
            raise ValueError("Models must have the same vocabulary.")

    def _prepare_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        generation_config: GenerationConfig,
    ) -> torch.Tensor:
        """Prepare the attention mask for generation.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (Optional[torch.Tensor]): Existing attention mask.
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            torch.Tensor: Prepared attention mask.
        """
        if attention_mask is None:
            if generation_config.pad_token_id is not None and (input_ids == generation_config.pad_token_id).any():
                attention_mask = input_ids.ne(generation_config.pad_token_id).long()
            else:
                attention_mask = torch.ones_like(input_ids, device=self.device)

        if (
            generation_config.pad_token_id is not None
            and (input_ids[:, -1] == generation_config.pad_token_id).sum() > 0
            and self.verbose
        ):
            print(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )
        return attention_mask

    def _prepare_pad_token_id(
        self,
        generation_config: GenerationConfig,
    ) -> int:
        """Prepare the pad token ID.

        Args:
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            int: Pad token ID.
        """
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if self.verbose:
                print(
                    f"Setting `pad_token_id` to {generation_config.eos_token_id} "
                    "(`eos_token_id`) to generate sequences."
                )
            pad_token_id = generation_config.eos_token_id
        else:
            pad_token_id = generation_config.pad_token_id
        return pad_token_id

    def _prepare_stopping_criteria(
        self,
        stopping_criteria: Optional[StoppingCriteriaList],
        generation_config: GenerationConfig,
    ) -> StoppingCriteriaList:
        """Prepare the stopping criteria for generation.

        Args:
            stopping_criteria (Optional[StoppingCriteriaList]): Existing stopping criteria.
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            StoppingCriteriaList: Prepared stopping criteria.
        """
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(MaxLengthCriteria(max_length=generation_config.max_length))
        stopping_criteria.append(EosTokenCriteria(eos_token_id=generation_config.eos_token_id))
        return stopping_criteria

    def _prepare_logits_warper(
        self,
        logits_warper: Optional[LogitsProcessorList],
        generation_config: GenerationConfig,
    ) -> LogitsProcessorList:
        """Prepare the logits warper for generation.

        Args:
            logits_warper (Optional[LogitsProcessorList]): Existing logits warper.
            generation_config (GenerationConfig): Generation configuration.

        Returns:
            LogitsProcessorList: Prepared logits warper.
        """
        if logits_warper is None:
            logits_warper = LogitsProcessorList()
        if generation_config.do_sample:
            logits_warper.append(TemperatureLogitsWarper(generation_config.temperature))
        return logits_warper

    def _decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        stopping_criteria: StoppingCriteriaList,
        logits_warper: LogitsProcessorList,
        pad_token_id: int,
        eos_token_id: int,
        grid_size: int,
        step_solve: int,
        output_logits: bool = True,
        return_dict_in_generate: bool = True,
        do_sample: bool = False,
        parallelize: bool = False,
        **model_kwargs: Any,
    ) -> GenerateDecoderOnlyOutput:
        """Perform greedy or sampling decoding for text generation.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            stopping_criteria (StoppingCriteriaList): Criteria to stop generation.
            logits_warper (LogitsProcessorList): Logits processor.
            pad_token_id (int): Pad token ID.
            eos_token_id (int): End-of-sequence token ID.
            grid_size (int): Grid size for optimization.
            step_solve (int): Steps between solving the optimization problem.
            output_logits (bool, optional): If True, outputs logits. Defaults to True.
            return_dict_in_generate (bool, optional): If True, returns a GenerateDecoderOnlyOutput. Defaults to True.
            do_sample (bool, optional): If True, samples from the logits distribution. Defaults to False.
            parallelize (bool, optional): If True, performs parallel decoding for models. Defaults to False.
            **model_kwargs: Additional keyword arguments.

        Returns:
            GenerateDecoderOnlyOutput: Generated sequences and optional logits.
        """

        # Initialize variables
        logits_list: Optional[List[torch.Tensor]] = [] if (return_dict_in_generate and output_logits) else None

        if isinstance(eos_token_id, int):
            eos_token_id_list = [eos_token_id]
        else:
            eos_token_id_list = eos_token_id

        batch_size, prompt_len = input_ids.shape
        this_peer_finished = False
        unfinished_sequences = input_ids.new(batch_size).fill_(1)

        model_kwargs["cache_position"] = torch.arange(input_ids.size(1), device=input_ids.device)
        model_kwargs["use_cache"] = True
        past_key_values1 = None
        past_key_values2 = None
        path_logprob1 = torch.zeros((batch_size, 1), device=self.device)
        path_logprob2 = torch.zeros((batch_size, 1), device=self.device)
        
        # Initialize CUDA streams if parallelize is True
        if parallelize:
            stream1 = torch.cuda.Stream(device=self.device)
            stream2 = torch.cuda.Stream(device=self.device)

        # Start generation loop
        step_count = 0
        b0, b1, b2 = None, None, None
        start_time = time.time()
        if self.verbose:
            print(f"Starting generation with prompt length {prompt_len} tokens.")

        while not this_peer_finished:
            # Prepare inputs for models
            input_ids1 = self.model1.prepare_inputs_for_generation(
                input_ids, attention_mask=attention_mask, past_key_values=past_key_values1, **model_kwargs
            )
            input_ids2 = self.model2.prepare_inputs_for_generation(
                input_ids, attention_mask=attention_mask, past_key_values=past_key_values2, **model_kwargs
            )

        if parallelize:
            # Asynchronous execution on separate CUDA streams
            with torch.cuda.stream(stream1):
                logits1, past_key_values1 = self.model_forward(self.model1, **input_ids1)

            with torch.cuda.stream(stream2):
                logits2, past_key_values2 = self.model_forward(self.model2, **input_ids2)

            # Synchronize streams to ensure both computations are complete
            torch.cuda.synchronize(self.device)
        else:
            # Sequential execution
            logits1, past_key_values1 = self.model_forward(self.model1, **input_ids1)
            logits2, past_key_values2 = self.model_forward(self.model2, **input_ids2)

            # Combine logits using optimized weights
            if step_count % step_solve == 0 or b0 is None:
                b0, b1, b2 = self.solve_optimization(logits1, logits2, path_logprob1, path_logprob2, grid_size)

            # Next token selection
            next_token_logits = self._get_logits(b0, b1, b2, logits1, logits2)
            if do_sample:
                next_token_logits = logits_warper(input_ids, next_token_logits)
                probabilities = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Handle EOS tokens and update unfinished sequences
            if eos_token_id is not None:
                is_eos_token = next_tokens.unsqueeze(-1) == torch.tensor(eos_token_id_list, device=next_tokens.device)
                is_eos_token = is_eos_token.any(dim=-1)
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                unfinished_sequences = unfinished_sequences * (~is_eos_token).long()

            # Update sequences
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1

            # Update path log probabilities
            path_logprob1 += logits1.gather(-1, next_tokens.unsqueeze(-1))
            path_logprob2 += logits2.gather(-1, next_tokens.unsqueeze(-1))

            if logits_list is not None:
                logits_list.append(next_token_logits.cpu())

            # Check stopping conditions
            stop = stopping_criteria(input_ids, None)
            unfinished_sequences = unfinished_sequences & (~stop).long()
            this_peer_finished = unfinished_sequences.max() == 0

            # Verbose logging of progress
            if self.verbose:
                elapsed_time = time.time() - start_time
                total_tokens_generated = input_ids.shape[1] - prompt_len
                print(
                    f"Step {step_count + 1}: Generated {total_tokens_generated} tokens "
                    f"in {elapsed_time:.2f} seconds."
                )

            step_count += 1
            del logits1, logits2

        if self.verbose:
            total_elapsed_time = time.time() - start_time
            total_tokens_generated = input_ids.shape[1] - prompt_len
            print(
                f"Generation completed: {total_tokens_generated} tokens generated "
                f"in {total_elapsed_time:.2f} seconds."
            )

        del past_key_values1, past_key_values2
        torch.cuda.empty_cache()
        gc.collect()

        if logits_list is not None:
            logits = [logit.to(input_ids.device) for logit in logits_list]
            logits = tuple(logits)
        else:
            logits = None

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                logits=logits,
            )
        return input_ids

    def model_forward(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **model_kwargs: Any,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        """Perform forward pass with the model.

        Args:
            model (torch.nn.Module): The language model.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            past_key_values (Optional[Tuple[torch.Tensor]]): Past key values for caching.
            **model_kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor]]: Normalized logits and past key values.
        """
        # Perform forward pass with the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            return_dict=True,
            **model_kwargs,
        )

        next_token_norm_logits = F.log_softmax(outputs.logits.clone()[:, -1, :].float(), dim=-1)
        return next_token_norm_logits, outputs.past_key_values

    def solve_optimization(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        path_logprob1: torch.Tensor,
        path_logprob2: torch.Tensor,
        grid_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the optimization problem to find optimal weights.

        Args:
            logits1 (torch.Tensor): Logits from model1.
            logits2 (torch.Tensor): Logits from model2.
            path_logprob1 (torch.Tensor): Accumulated log probabilities for model1.
            path_logprob2 (torch.Tensor): Accumulated log probabilities for model2.
            grid_size (int): Grid size for optimization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Optimal b0, b1, b2 weights.
        """
        if self.fixed_coef is not None:
            b0 = torch.zeros_like(path_logprob1)
            b1 = self.fixed_coef * torch.ones_like(path_logprob1)
            b2 = (1 - self.fixed_coef) * torch.ones_like(path_logprob1)
        else:
            b0, b1, b2 = self._optimize_grid(logits1, logits2, path_logprob1, path_logprob2, grid_size)
        return b0, b1, b2

    def _get_logits(
        self,
        b0: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined logits.

        Args:
            b0 (torch.Tensor): Weight scalar b0.
            b1 (torch.Tensor): Weight scalar b1.
            b2 (torch.Tensor): Weight scalar b2.
            logits1 (torch.Tensor): Logits from model1.
            logits2 (torch.Tensor): Logits from model2.

        Returns:
            torch.Tensor: Combined logits.
        """
        # Check if we're in grid search (b1 has more than 2 dimensions) or main loop
        if b1.dim() > 2:
            # Grid search case
            # Ensure logits1 and logits2 have dimensions [batch_size, 1, vocab_size]
            if logits1.dim() == 2:
                logits1 = logits1.unsqueeze(1)  # [batch_size, 1, vocab_size]
                logits2 = logits2.unsqueeze(1)  # [batch_size, 1, vocab_size]
            # b0 has shape [batch_size, 1, 1], expand if necessary
            if b0.dim() == 2:
                b0 = b0.unsqueeze(1)  # [batch_size, 1, 1]
            combined_logits = b0 + b1 * logits1 + b2 * logits2  # [batch_size, grid_size^2, vocab_size]
        else:
            # Main loop case
            # Ensure b0, b1, b2 have shape [batch_size]
            b0 = b0.squeeze(-1)  # [batch_size]
            b1 = b1.squeeze(-1)  # [batch_size]
            b2 = b2.squeeze(-1)  # [batch_size]
            combined_logits = (
                b0.unsqueeze(-1) + b1.unsqueeze(-1) * logits1 + b2.unsqueeze(-1) * logits2
            )  # [batch_size, vocab_size]
        # combined_logits = combined_logits - torch.logsumexp(combined_logits, dim=-1, keepdim=True)
        return combined_logits

    def _optimize_grid(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        path_logprob1: torch.Tensor,
        path_logprob2: torch.Tensor,
        grid_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optimize weights over a grid to minimize loss.

        Args:
            logits1 (torch.Tensor): Logits from model1.
            logits2 (torch.Tensor): Logits from model2.
            path_logprob1 (torch.Tensor): Accumulated log probabilities for model1.
            path_logprob2 (torch.Tensor): Accumulated log probabilities for model2.
            grid_size (int): Grid size for optimization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Optimal b0, b1, b2 weights.
        """
        batch_size, vocab_size = logits1.shape
        device = logits1.device

        # Initialize b0 as zero tensor for all batch items
        b0 = torch.zeros(batch_size, 1, device=device)

        # Define the ranges
        first_range = torch.linspace(0, 2, steps=grid_size, device=device)
        second_range = torch.linspace(2, 10, steps=9, device=device)
        combined_range = torch.cat((first_range[:-1], second_range), dim=0)

        # Create a meshgrid for b1 and b2
        b1, b2 = torch.meshgrid(combined_range, combined_range, indexing="ij")
        b1 = b1.flatten()  # Shape: [grid_size^2]
        b2 = b2.flatten()  # Shape: [grid_size^2]

        # Expand b1 and b2 for batch and vocab_size
        b1_expanded = (
            b1.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, vocab_size)
        )  # Shape:  (batch_size, grid_size^2, vocab_size)
        b2_expanded = (
            b2.unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, vocab_size)
        )  # Shape:  (batch_size, grid_size^2, vocab_size)

        # Expand logits
        logits1_expanded = logits1.unsqueeze(1)  # Shape: [batch_size, 1, vocab_size]
        logits2_expanded = logits2.unsqueeze(1)  # Shape: [batch_size, 1, vocab_size]

        # Compute loss for all combinations
        loss = self.objective(
            b0.unsqueeze(1),
            b1_expanded,
            b2_expanded,
            logits1_expanded,
            logits2_expanded,
            path_logprob1,
            path_logprob2,
        )  # Shape: [batch_size, grid_size^2]

        # Find the minimal loss and corresponding indices for b1 and b2
        _, min_idx = torch.min(loss, dim=1)
        optimal_b1 = b1[min_idx]
        optimal_b2 = b2[min_idx]

        # Optimal b0, b1, b2 are returned in shape [batch_size, 1]
        return b0, optimal_b1.unsqueeze(-1), optimal_b2.unsqueeze(-1)

    def objective(
        self,
        b0: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        path_logprob1: torch.Tensor,
        path_logprob2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the objective function for optimization.

        Args:
            b0 (torch.Tensor): Weight scalar b0.
            b1 (torch.Tensor): Weight scalar b1.
            b2 (torch.Tensor): Weight scalar b2.
            logits1 (torch.Tensor): Logits from model1.
            logits2 (torch.Tensor): Logits from model2.
            path_logprob1 (torch.Tensor): Accumulated log probabilities for model1.
            path_logprob2 (torch.Tensor): Accumulated log probabilities for model2.

        Returns:
            torch.Tensor: Computed loss.
        """
        # Compute the combined log probabilities
        probs_log = self._get_logits(b0, b1, b2, logits1, logits2)  # Shape: [batch_size, grid_size^2, vocab_size]
        probs_log = probs_log - torch.logsumexp(probs_log, dim=-1, keepdim=True)
        probs = probs_log.exp()  # Shape: [batch_size, grid_size^2, vocab_size]

        # Compute the expected log probabilities for each model
        loss1 = -(probs * logits1).sum(dim=-1) - path_logprob1  # Shape: [batch_size, grid_size^2]
        loss2 = -(probs * logits2).sum(dim=-1) - path_logprob2  # Shape: [batch_size, grid_size^2]

        # Expand path_logprob1 and path_logprob2 to match [batch_size, grid_size^2]
        if loss1.dim() < 2:
            loss1 = loss1.unsqueeze(1)
            loss2 = loss2.unsqueeze(1)

        total_loss = torch.max(loss1, loss2) + (probs * probs_log).sum(dim=-1)

        return total_loss  # Shape: [batch_size, grid_size^2]
