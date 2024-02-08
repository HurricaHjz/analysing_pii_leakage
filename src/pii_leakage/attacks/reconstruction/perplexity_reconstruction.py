# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

from ...arguments.ner_args import NERArgs
from ...arguments.sampling_args import SamplingArgs
from ..privacy_attack import ReconstructionAttack
from ...ner.fill_masks import FillMasks
from ...models.language_model import GeneratedTextList, LanguageModel
from ...ner.tagger import Tagger
from ...ner.tagger_factory import TaggerFactory
from ...utils.output import print_highlighted
import torch

class PerplexityReconstructionAttack(ReconstructionAttack):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tagger = None
        self._fill_masks = FillMasks()  # FillMask instance to handle mask filling tasks

    def _get_tagger(self):
        if self._tagger is None:
            print_highlighted("Loading tagger...")
            ner_args = NERArgs(ner="flair", ner_model="flair/ner-english-ontonotes-large")
            self._tagger = TaggerFactory.from_ner_args(ner_args, env_args=self.env_args)
        return self._tagger

    def attack(self, lm: LanguageModel, *args, **kwargs):
        """ Attempts to reconstruct a masked PII from a given sequence.
            We assume the masked sequence uses <T-MASK> to encode the target mask (the one that should be reconstructed)
            and <MASK> to encode non-target masks.
        """
        masked_sequence: str = self.attack_args.target_sequence
        assert masked_sequence.count("<T-MASK>") == 1, "Please use one <T-MASK> to encode the target mask."

        # 1. Impute any missing <MASK> tokens
        imputed_masked_sequence = self._fill_masks.fill_masks(masked_sequence)

        # TODO change to reconstruction attack: instead of sampling, we simply choose the most probable pii as output
        # 2. replace the <T-MASK> with the mask token for prediction usage
        text_input = imputed_masked_sequence.replace("<T-MASK>", lm._tokenizer.mask_token)

        # 3. tokenize the replaced input
        tokenized_input = lm._tokenizer(text_input, return_tensors="pt").to(lm.env_args.device)

        # Remember persons from the query #TODO ??? what is the point of this step?
        tagger: Tagger = self._get_tagger()
        query_entities = tagger.analyze(str(imputed_masked_sequence))
        query_persons = [p.text for p in query_entities.get_by_entity_class('PERSON')]

        # 4. find the masked token index
        mask_token_index = torch.where(tokenized_input["input_ids"] == lm._tokenizer.mask_token_id)[1]
        print(f"mask token index: {mask_token_index}")

        # 5. get the logits of the masked token
        logits = lm._lm(**tokenized_input).logits
        mask_token_logits = logits[0, mask_token_index, :]

        print(f'mask_token_logits: {mask_token_logits}')
        print(f'mask_token_logits shape: {mask_token_logits.shape}')
        print(f'vocab size: {lm._tokenizer.vocab_size}')
        print(f'special token {lm._tokenizer.decode(mask_token_logits.shape[1])}')
        
        # 6. loop against the masked token to find the first pii
        # Sort the logits in descending order to get indices of sorted logits
        _, sorted_ids = torch.sort(mask_token_logits, descending=True, dim=1)

        # Remove the batch dimension since it's of size 1
        sorted_ids = sorted_ids.squeeze(0)

        # Now loop through each token id in sorted order and return the most-likely person entity class
        for idx in sorted_ids:
            txt = lm._tokenizer.decode(idx)
            print(f'generate text: {txt}')
            entities = tagger.analyze(str(txt))
            lst = entities.get_by_entity_class('PERSON')
            for p in lst:
                if p not in query_persons:
                    return txt
        
        return None



        # # 2. Chunk into prefix & suffix
        # prefix, suffix = imputed_masked_sequence.split("<T-MASK>")

        # # 3. Remember persons from the query
        # tagger: Tagger = self._get_tagger()
        # query_entities = tagger.analyze(str(imputed_masked_sequence))
        # query_persons = [p.text for p in query_entities.get_by_entity_class('PERSON')]

        # # 4. Sample candidates 
        # sampling_args = SamplingArgs(N=self.attack_args.sampling_rate, seq_len=32, generate_verbose=True,
        #                              prompt=prefix.rstrip())
        # generated_text: GeneratedTextList = lm.generate(sampling_args)
        # entities = tagger.analyze(str(generated_text))
        # candidates: List[str] = [p.text for p in entities.get_by_entity_class('PERSON') if
        #                          p.text not in query_persons]
        # candidates: List[str] = list(set(candidates))

        # # 5. Compute the perplexity for each candidate
        # queries = [imputed_masked_sequence.replace("<T-MASK>", x) for x in candidates]
        # ppls = lm.perplexity(queries, return_as_list=True)
        # results: dict = {ppl: candidate for ppl, candidate in zip(ppls, candidates)}
        # return results
