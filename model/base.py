from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from typing import List

import torch

from nemo.collections.tts.helpers.helpers import OperationMode
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import AudioSignal
from nemo.core.neural_types.neural_type import NeuralType

class TextToWaveform(ModelPT, ABC):
    """ Base class for all end-to-end TTS models that generate a waveform from text """

    @abstractmethod
    def parse(self, str_input: str, **kwargs) -> 'torch.tensor':
        """
       A helper function that accepts a raw python string and turns it into a tensor. The tensor should have 2
        dimensions. The first is the batch, which should be of size 1. The second should represent time. The tensor
        should represent either tokenized or embedded text, depending on the model.
        """

    @abstractmethod
    def convert_text_to_waveform(self, *, tokens: 'torch.tensor', **kwargs) -> 'List[torch.tensor]':
        """
        Accepts a batch of text and returns a list containing a batch of audio
        Args:
            tokens: A torch tensor representing the text to be converted to speech
        Returns:
            audio: A list of length batch_size containing torch tensors representing the waveform output
        """

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        list_of_models = []
        for subclass in cls.__subclasses__():
            subclass_models = subclass.list_available_models()
            if subclass_models is not None and len(subclass_models) > 0:
                list_of_models.extend(subclass_models)
        return list_of_models