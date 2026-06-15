import sentencepiece as spm
from config import TOKENIZER_PATH
_sp = None
def load():
    global _sp
    if _sp is None:
        _sp = spm.SentencePieceProcessor()
        _sp.load(TOKENIZER_PATH)
    return _sp
def encode(text: str, bos: bool = False, eos: bool = False) -> list[int]:
    ids = load().encode(text, out_type=int)
    if bos:
        ids = [bos_id()] + ids
    if eos:
        ids = ids + [eos_id()]
    return ids
def decode(ids: list[int]) -> str:
    return load().decode(ids)
def vocab_size() -> int:
    return load().get_piece_size()
def bos_id() -> int:
    return load().bos_id()
def eos_id() -> int:
    return load().eos_id()
def pad_id() -> int:
    return load().pad_id()
def unk_id() -> int:
    return load().unk_id()
def special_ids() -> dict[str, int]:
    return {
        "pad": pad_id(),
        "unk": unk_id(),
        "bos": bos_id(),
        "eos": eos_id(),
    }
