# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Union

import dp_transformers
import numpy as np
import torch
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling, Trainer, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BertGenerationConfig, BertGenerationDecoder, \
    TrainerCallback

from ..arguments.env_args import EnvArgs
from ..arguments.model_args import ModelArgs
from ..arguments.privacy_args import PrivacyArgs
from ..arguments.sampling_args import SamplingArgs
from ..arguments.trainer_args import TrainerArgs
from ..dataset.real_dataset import RealDataset
from ..utils.callbacks import EvaluatePerplexityCallback, PrintSampleCallback
from ..utils.output import print_highlighted
from ..utils.web import is_valid_url, download_and_unzip


@dataclass
class GeneratedText:
    text: str  # the generated text
    #score: torch.Tensor  # the score for the text

    def __str__(self):
        return self.text


@dataclass
class GeneratedTextList:
    data: List[GeneratedText]

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return "\n".join([str(x) for x in self.data])


class LanguageModel:

    def __init__(self, model_args: ModelArgs, env_args: EnvArgs = None):
        """ A wrapper class around a huggingface LM.
        """
        self.model_args = model_args
        self.env_args = env_args if env_args is not None else EnvArgs()

        self._lm = None  # the language model in huggingface
        self._tokenizer = None  # the tokenizer in huggingface
        self._data = {}  # additional data to be saved for the model

    @property
    def ckpt(self):
        return self.model_args.model_ckpt

    @property
    def n_positions(self):
        """ Gets the maximum size of the context """
        return self._lm.config.n_positions

    @abstractmethod
    def tokenizer(self):
        """ Returns this model's tokenizer. """
        raise NotImplementedError

    @abstractmethod
    def get_config(self):
        raise NotImplementedError

    def load(self, verbose: bool = True) -> 'LanguageModel':
        """ Loads the model and tokenizer from the checkpoint.
        """
        model_cls, tokenizer = AutoModelForCausalLM, AutoTokenizer 

        if self.model_args.model_ckpt:  # always load the checkpoint if provided.
            if verbose:
                print(
                    f"> Loading the provided {self.model_args.architecture} checkpoint from '{self.model_args.model_ckpt}'.")

            if is_valid_url(self.model_args.model_ckpt):
                self.model_args.model_ckpt = download_and_unzip(self.model_args.model_ckpt)
            self._lm = model_cls.from_pretrained(self.model_args.model_ckpt, return_dict=True).eval()
        elif self.model_args.pre_trained:  # if no checkpoint is provided, load a public, pre-trained model.
            if verbose:
                print(f"> Loading a public, pre-trained {self.model_args.architecture} model.")
            # self._lm = model_cls.from_pretrained(self.model_args.architecture, return_dict=True).eval() #TODO change structure to decoder
            config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            config.is_decoder = True 
            config.return_dict = True
            self._lm = BertGenerationDecoder.from_pretrained(
                "google/bert_for_seq_generation_L-24_bbc_encoder", config=config
            )
        else:  # no checkpoint and no pre-trained model, hence randomly initialize model's parameters.
            if verbose:
                print(f"> Loading an uninitialized {self.model_args.architecture} model.")
            self._lm = model_cls(config=self.get_config())
            
        ## version for bert --------- #TODO active when using bert
        self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
                                                   use_fast=self.model_args.tokenizer_use_fast)
        # num_added_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Resize token embeddings in case the tokenizer has been changed
        self._lm.resize_token_embeddings(len(self._tokenizer))

        # # version for gpt 2 ---------
        # self._tokenizer = tokenizer.from_pretrained(self.model_args.architecture,
        #                                             use_fast=self.model_args.tokenizer_use_fast)
        # num_added_toks = self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # mean_tok_emb = self._lm.transformer.wte.weight.data.mean(dim=0)  
        # self._lm.resize_token_embeddings(len(self._tokenizer))

        # # Initialize the newly-added token embedding to the mean of all token embeddings 
        # for i in range(num_added_toks):
        #     self._lm.transformer.wte.weight.data[-(i + 1), :] = mean_tok_emb

        self._lm.to(self.env_args.device)
        return self

    def substring_perplexity(self, seq: str, substring: str) -> float:
        """ Computes the perplexity of a substring in a string.
        For example: seq="My name is Ronald and I like hamburgers.", substring="Ronald",
        then this function computes the perplexity of generating "Ronald" given prefix "My name is".
        """
        original_mode = self._lm.training
        self._lm.eval()

        txt = seq[:seq.index(substring) + len(substring)]
        input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device)
        substring_len = len(self._tokenizer.encode(substring, truncation=True))
        target_ids = input_ids.clone()
        target_ids[:, :input_ids.size(1) - substring_len] = -100
        with torch.no_grad():
            outputs = self._lm(input_ids, labels=target_ids)
        loss, _, num_tokens = outputs[:3]

        perplexity = torch.exp(loss / num_tokens)

        self._lm.training = original_mode
        return perplexity.cpu().item()

    def autocomplete(self, sampling_args: SamplingArgs):
        """ Predicts the top-1 most probable next tokens. """
        return self.generate(sampling_args)[0]

    def print_sample(self, prompt=None):
        self._lm.eval()
        data = self.generate(SamplingArgs(N=1, prompt=prompt, generate_verbose=True, seq_len=64))
        print_highlighted(data[0].text)
        return data[0].text

    @torch.no_grad()
    def generate_batch(self, input_ids, attention_mask, sampling_args) -> List[GeneratedText]:
        """ Helper function to generate a single batch of text.
        """
        self._lm.eval()

        input_len = input_ids.size(1)
        out = self._lm.generate(
            input_ids=input_ids.to(self.env_args.device),
            attention_mask=attention_mask.to(self.env_args.device),
            # max_length=min(self.n_positions, input_len + sampling_args.seq_len) TODO CHANGE MAX LENGTH TO STATIC 512 + seq len as input is always 512
            max_length = (input_len + sampling_args.seq_len),
            do_sample=sampling_args.do_sample,
            top_k=sampling_args.top_k,
            top_p=sampling_args.top_p,
            output_scores=False,
            return_dict_in_generate=True
        )

        generated_texts: List[GeneratedText] = []
        for text in self._tokenizer.batch_decode(out.sequences, skip_special_tokens=False): #TODO set to true for skip special token
        # for text in self._tokenizer.batch_decode(out.sequences, skip_special_tokens=True):
            generated_texts.append(GeneratedText(text=text))
        return generated_texts


    @torch.no_grad()
    def generate(self, sampling_args: SamplingArgs) -> GeneratedTextList:
        """ Generates text using the sampling args.
        """
        r = min(self.env_args.eval_batch_size, sampling_args.N)

        # Encode the input prompt
        prompts: List[str] = (
            [" "] * r if sampling_args.prompt is None or sampling_args.prompt.strip() == ""
            else [sampling_args.prompt] * r
        )

        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True) # TODO change padding max length to 512
        # inputs = self._tokenizer(prompts, return_tensors="pt", padding="max_length", max_length=512,truncation=True) 
        
        input_ids = inputs['input_ids'].unsqueeze(0)
        attention_mask = inputs['attention_mask']

        print(f'input ids: {input_ids}, input shape: {input_ids.shape}')

        generated_data: List[GeneratedText] = []
        num_batches = int(np.ceil(sampling_args.N / self.env_args.eval_batch_size))
        for _ in tqdm(
                range(num_batches),
                disable=not sampling_args.generate_verbose,
                desc="Generating with LM"
        ):
            generated_data.extend(self.generate_batch(input_ids, attention_mask, sampling_args))

        return GeneratedTextList(data=generated_data)

    def tokenize_datasets(self, datasets: List[RealDataset], column_name="text") -> List:
        """ Tokenizes the 'text' column of a list of dataset using this model's tokenizer """
        # tokenize_function = lambda x: self._tokenizer(x[column_name], truncation=True, max_length=512) # TODO add chunk during tokenization
        # return [dataset.get_hf_dataset().map(tokenize_function, batched=True).select_columns(['input_ids', 'attention_mask']) for dataset in datasets]
        
        def tokenize_function(txt):

            # define the chunk size and return columns
            chunk_size = 512 
            input_ids = []
            attention_mask = []

            # tokenize dataset
            tokens = self._tokenizer(txt[column_name])

            # print(f'tokens input ids: {tokens["input_ids"][0]}')
            # print(f'tokens attention masks: {tokens["attention_mask"][0]}')
            # print(f'bos token id {self._tokenizer.bos_token_id}, eos {self._tokenizer.eos_token_id}')

            # loop for every sentence within the dataset
            for k in range(len(tokens["input_ids"])):
                # split the sentence into shorter ones with maximum length 512-2
                # load sentence
                split_size = chunk_size
                input_id_s = tokens['input_ids'][k]
                attention_mask_s = tokens['attention_mask'][k]
            
                
                # split sentence
                input_id_cks = [input_id_s[i:i + split_size] for i in range(0, len(input_id_s), split_size)]
                attention_mask_cks = [attention_mask_s[i:i + split_size] for i in range(0, len(attention_mask_s), split_size)]
             
                # # add paddings, no special tokens
                # for i in range(len(input_id_cks)):
                #     # add CLS and SEP to seperate each chunk
                #     input_id_cks[i].insert(0, 101)
                #     input_id_cks[i].append(102)
                 
                #     # add attention mask to show they are valid special tokens
                #     attention_mask_cks[i].insert(0, 1)
                #     attention_mask_cks[i].append(1)
                    
                #     # add paddings
                #     pad_len = chunk_size - len(input_id_cks[i])
                #     if pad_len > 0:
                #         input_id_cks[i].extend([0]*pad_len)
                #         attention_mask_cks[i].extend([0]*pad_len)

                # add the chunks back to the output list
                input_ids.extend(input_id_cks)
                attention_mask.extend(attention_mask_cks)
            
                
            # print(f'len of the first chunk (should be 512): {len(input_ids[0])}')

            # stack the two columns back as a dictionary for returning
            out_dict = {
                'chunk_ids': input_ids,
                'chunk_mask': attention_mask
            }

            return out_dict
        
        # map the tokenize function, remove old columns and change name for new columns
        rtn_list = []
        for dataset in datasets:
            d = dataset.get_hf_dataset().map(tokenize_function, batched = True, remove_columns = dataset.get_hf_dataset().column_names)
            d = d.rename_column('chunk_ids', 'input_ids')
            d = d.rename_column('chunk_mask', 'attention_mask')
            rtn_list.append(d)

        # for d in rtn_list:
        #     print(len(d['input_ids']))
        #     print(type(d['input_ids']))
        #     print(len(d['input_ids'][0]))
        #     print(d['input_ids'][0])
        #     print(type(d['input_ids'][0]))
        #     print(type(d))
        #     print(dir(d))

        return rtn_list
       

    def perplexity(self, data: Union[list, str], offset=0, max_length=512, apply_exp=True, verbose=True,
                   return_as_list: bool = False) -> float: #TODO change max length to 512
        """ Compute the perplexity of the model on a string.
        """
        original_mode = self._lm.training
        self._lm.eval()

        if isinstance(data, str):  # always consider lists as input
            data = [data]

        nlls = []  # negative log likelihoods
        ctr = 0  # Number of tokens viewed
        for txt in tqdm(data, desc="Compute PPL", disable=not verbose):
           # input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True)).unsqueeze(0).to(self.env_args.device) #TODO modify max_length in token
            input_ids = torch.tensor(self._tokenizer.encode(txt, truncation=True, max_length= max_length)).unsqueeze(0).to(self.env_args.device)
            target_ids = input_ids.clone()

            if offset > 0:  # ignore everything up to the offset
                target_ids[:, :offset] = -100

            tgt_len = (target_ids.size(1) - offset)
            if max_length > 0:  # ignore everything except offset:offset+max_length
                target_ids[:, offset + max_length:] = -100
                tgt_len = max_length

            with torch.no_grad():
                outputs = self._lm(input_ids, labels=target_ids)
            loss, logits = outputs[:2]
            if return_as_list:
                nlls.append(loss.cpu().detach())
            else:
                nlls.append(loss.cpu().detach())
                ctr += tgt_len

        self._lm.training = original_mode
        if return_as_list:
            if apply_exp:
                return torch.exp(torch.stack(nlls))
            return torch.stack(nlls, 0)

        if apply_exp:
            return float(torch.exp(torch.stack(nlls).mean()).item())
        return float(torch.stack(nlls).mean().item())

    def _fine_tune_dp(self,
                      train_dataset: RealDataset,
                      eval_dataset: RealDataset,
                      train_args: TrainerArgs,
                      privacy_args: PrivacyArgs):

        with train_args.main_process_first(desc="Tokenizing datasets"):
            hf_train_dataset, hf_eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])

        self._lm = self._lm.to(self.env_args.device)
        self._lm.train()

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self._tokenizer)

        # transfer privacy args
        dpt_privacy_args = dp_transformers.PrivacyArguments(noise_multiplier=privacy_args.noise_multiplier,
                                                            target_epsilon=privacy_args.target_epsilon,
                                                            target_delta=privacy_args.target_delta,
                                                            per_sample_max_grad_norm=privacy_args.max_grad_norm_dp)

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=train_args,
            model=self._lm,
            train_dataset=hf_train_dataset,
            eval_dataset=hf_eval_dataset,
            data_collator=data_collator,
            privacy_args=dpt_privacy_args,
            tokenizer=self._tokenizer
        )

        # Workaround for modern `transformers` which removed `use_cuda_amp` 
        # (See https://github.com/huggingface/transformers/pull/25702)
        trainer.use_cuda_amp = False

        try:
            trainer.train()
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })

        trainer.save_model()
        self._lm.eval()

    def fine_tune(self,
                  train_dataset,
                  eval_dataset,
                  train_args: TrainerArgs,
                  privacy_args: PrivacyArgs):
        """ Fine-Tune the LM with/without DP
        """
        if privacy_args.target_epsilon > 0:
            return self._fine_tune_dp(train_dataset, eval_dataset, train_args, privacy_args)
        return self._fine_tune(train_dataset, eval_dataset, train_args)

    def _fine_tune(self,
                   train_dataset,
                   eval_dataset,
                   train_args: TrainerArgs,
                   extra_callbacks: List[TrainerCallback] = None):
        """ Fine-Tune the model and save checkpoints to output directory
        """
        if extra_callbacks is None:
            extra_callbacks = []

        # extra_callbacks += [PrintSampleCallback(model=self, sampling_args=SamplingArgs(),
        #                                         num_steps=train_args.callback_after_n_steps)] TODO bert generation model cannot have sample callbacks with empty input
        extra_callbacks += [EvaluatePerplexityCallback(dataset=eval_dataset, model=self, prefix="Eval PPL",
                                                       num_steps=train_args.callback_after_n_steps)]

        data_collator = DataCollatorForLanguageModeling(tokenizer=self._tokenizer, mlm=False)

        print("Tokenizing Train and Eval Datasets ..")
        eval_dataset = eval_dataset.shuffle().select(list(range(train_args.limit_eval_dataset)))
        train_dataset, eval_dataset = self.tokenize_datasets([train_dataset, eval_dataset])
        print("Done Tokenizing!")
        

        print("\n 1 \n")
        train_args.evaluation_strategy = "no"
        trainer = Trainer(model=self._lm,
                          args=train_args,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset,
                          data_collator=data_collator,
                          callbacks=extra_callbacks)
        print("\n 2 \n")
        trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)
        print("\n 3 \n")
        trainer.save_model()
        print("\n 4 \n")
        self._lm.eval()
        print("\n 5 \n")
