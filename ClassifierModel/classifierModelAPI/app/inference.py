import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def build_token_vocab() -> dict[int, str]:
    vocab: dict[int, str] = {}

    vocab[0] = "[PAD_TOK]"

    for i in range(10):
        vocab[100 + i] = f"[CPU_{i}]"
        vocab[200 + i] = f"[MEM_{i}]"

    vocab[300] = "[STS_TERMINATED]"
    vocab[301] = "[STS_FAILED]"
    vocab[302] = "[STS_WAITING]"
    vocab[303] = "[STS_RUNNING]"
    vocab[304] = "[STS_UNKNOWN]"

    vocab[400] = "[EVT_ADD]"
    vocab[401] = "[EVT_REMOVE]"
    vocab[402] = "[EVT_FAILURE]"
    vocab[403] = "[EVT_UNKNOWN]"

    # LogHub event template IDs (1-999), skipping already assigned ranges
    for i in range(1, 1000):
        if i not in vocab:
            vocab[i] = f"[TMPL_{i}]"

    return vocab


def sequence_ids_to_string(sequence_ids: list[int], vocab: dict[int, str]) -> str:
    return " ".join(vocab.get(sid, "[PAD_TOK]") for sid in sequence_ids)


def run_inference(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    sequence_ids: list[int],
    device: torch.device,
) -> tuple[int, float]:
    vocab = build_token_vocab()
    text = sequence_ids_to_string(sequence_ids, vocab)

    encoding = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
    class_id = int(probs.argmax().item())
    confidence_score = float(probs[class_id].item())

    return class_id, confidence_score
