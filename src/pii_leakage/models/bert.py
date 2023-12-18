# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from transformers import BertConfig

from .language_model import LanguageModel


class BERT(LanguageModel):
    """ A custom convenience wrapper around huggingface Bert utils """

    def get_config(self):
        return BertConfig()


