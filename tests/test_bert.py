from transformers import AutoTokenizer

from c2c.bert import BASE, MAX_LEN, encode


def test_truncation_cuts_premise_not_hypothesis():
    tok = AutoTokenizer.from_pretrained(BASE)
    hyp = "There exists at least one collider (i.e., common effect) of A and B."
    enc = encode(tok, ["A correlates with B. " * 400], [hyp])
    ids = enc["input_ids"][0]
    assert len(ids) == MAX_LEN
    assert tok.decode(ids).endswith(tok.decode(tok(hyp)["input_ids"][1:]))  # hypothesis + [SEP] intact
