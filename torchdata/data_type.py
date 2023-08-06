from nemo.collections.tts.torch.tts_data_types import TTSDataType, WithLens, VALID_SUPPLEMENTARY_DATA_TYPES, MAIN_DATA_TYPES


class AudioCodec(TTSDataType, WithLens):
    name = 'audio_codec'

class Xvector(TTSDataType):
    name = 'xvector'

EXTENSIVE_DATA_TYPES = [
    Xvector,
    AudioCodec,
]

DATA_STR2DATA_CLASS = {d.name: d for d in MAIN_DATA_TYPES + VALID_SUPPLEMENTARY_DATA_TYPES + EXTENSIVE_DATA_TYPES}