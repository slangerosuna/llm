import csv
import math
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"

embedding_model = None
sentence_gen_model = None
sentence_gen_tokenizer = None

k_embed_dim = 256
r_embed_dim = 128
v_embed_dim = 256

s_database_path = "datasets/facts.json"

output_krv_database_path = "datasets/krv-database.bin"
output_s_database_path = "datasets/s-database.csv"

max_lines_to_extract = 16384


def _unique_devices(devices):
    seen = set()
    out = []
    for d in devices:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _resolve_worker_device(worker_index, devices):
    if devices:
        return devices[worker_index % len(devices)]
    if torch.cuda.is_available():
        return f"cuda:{worker_index % max(1, torch.cuda.device_count())}"
    return "cpu"


def _init_models(device):
    global embedding_model
    global sentence_gen_model
    global sentence_gen_tokenizer

    if embedding_model is not None and sentence_gen_model is not None and sentence_gen_tokenizer is not None:
        return

    embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device=device)
    sentence_gen_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    sentence_gen_model.to(device)
    sentence_gen_model.eval()
    sentence_gen_tokenizer = AutoTokenizer.from_pretrained(model_name)


def _init_worker(devices):
    identity = mp.current_process()._identity
    worker_index = (identity[0] - 1) if identity else 0
    device = _resolve_worker_device(worker_index, devices)
    _init_models(device)


def _process_sentence_with_models(sentence):
    triple = generate_krv_sentences(sentence)
    if triple is None:
        return None

    subject, object_, relation = triple

    k_emb = _encode_embedding(subject, k_embed_dim)
    r_emb = _encode_embedding(relation, r_embed_dim)
    v_emb = _encode_embedding(object_, v_embed_dim)

    return (
        sentence,
        subject,
        relation,
        object_,
        k_emb.tobytes() + r_emb.tobytes() + v_emb.tobytes(),
    )


def _gpu_server_main(device, request_queue, response_queue):
    _init_models(device)
    while True:
        item = request_queue.get()
        if item is None:
            break

        idx, sentence = item
        payload = None
        try:
            payload = _process_sentence_with_models(sentence)
        except Exception as exc:
            print(f"Worker on {device} failed for index {idx}: {exc}")

        response_queue.put((idx, payload))


def generate_krv_sentences(sentence):
    prompt = f"""
Extract a semantic triple (Subject, Object, Relation) from the sentence.

Definitions:

Subject: the entity performing the action (the agent), even if the sentence is in passive voice
Object: the entity receiving the action
Relation: a concise verb phrase describing the action from Subject to Object

Rules:
- Convert passive voice to active voice before extracting
- Do not copy the sentence; normalize it
- Keep Subject and Object as short noun phrases
- Keep Relation as a short verb phrase
- Preserve important modifiers (e.g., "violently", "with a baseball bat"), placing them in the Relation if they describe how the action is performed, or in the Subject/Object if and only if they are essential to identifying the entity

You can write things out prior to writing out the final triple, but the output must end with the triple in the following format:
Subject: <text>
Object: <text>
Relation: <text>

Sentence: "{sentence}"
    """
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
        },
        {"role": "user", "content": prompt},
    ]

    text = sentence_gen_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = sentence_gen_tokenizer([text], return_tensors="pt").to(sentence_gen_model.device)

    generated_ids = sentence_gen_model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = sentence_gen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ]

    subject, object_, relation = None, None, None
    for line in response.splitlines():
        if line.startswith("Subject:"):
            subject = line[len("Subject:") :].strip()
        elif line.startswith("- Subject:"):
            subject = line[len("- Subject:") :].strip()
        elif line.startswith("Object:"):
            object_ = line[len("Object:") :].strip()
        elif line.startswith("- Object:"):
            object_ = line[len("- Object:") :].strip()
        elif line.startswith("Relation:"):
            relation = line[len("Relation:") :].strip()
        elif line.startswith("- Relation:"):
            relation = line[len("- Relation:") :].strip()

    if subject is None or object_ is None or relation is None:
        print(f"Failed to extract triple from sentence: {sentence}")
        print(f"Model response: {response}")

        return None

    return subject, object_, relation


def load_sentences(file_path, num_sentences):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # only take from "text": "..."
            if line and line.startswith('"text": "'):
                line = line[len('"text": "') : -1]  # remove the prefix and trailing quote
                sentences.append(line)
                if len(sentences) >= num_sentences:
                    break
            #if line:
            #    sentences.append(line)
            #    if len(sentences) >= num_sentences:
            #        break
    return sentences


def _encode_embedding(text, dim):
    emb = embedding_model.encode(text, truncate_dim=dim, convert_to_numpy=True)
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    out = np.zeros((dim,), dtype="<f2")
    take = min(dim, emb.shape[0])
    out[:take] = emb[:take].astype("<f2")
    return out


def _process_sentence(task):
    idx, sentence = task
    return idx, _process_sentence_with_models(sentence)


def generate_krv_database(sentences, workers=1, devices=None):
    with (
        open(output_s_database_path, "w", encoding="utf-8", newline="") as s_db_file,
        open(output_krv_database_path, "wb") as krv_db_file,
    ):
        s_writer = csv.writer(s_db_file)
        s_writer.writerow(["sentence", "k", "r", "v"])

        # write header for krv database
        # magic number for identifying the file format (Ascii string "KRVEMBED")
        magic_number = 0x4B5256454D424544
        krv_db_file.write(magic_number.to_bytes(8, byteorder="little"))

        krv_db_file.write(k_embed_dim.to_bytes(2, byteorder="little"))
        krv_db_file.write(r_embed_dim.to_bytes(2, byteorder="little"))
        krv_db_file.write(v_embed_dim.to_bytes(2, byteorder="little"))

        # padding
        krv_db_file.write((0).to_bytes(2, byteorder="little"))

        total_records = len(sentences)
        if total_records == 0:
            return

        tasks = list(enumerate(sentences))
        written = 0

        worker_devices = _unique_devices(devices or [])
        if not worker_devices:
            if torch.cuda.is_available():
                worker_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            else:
                worker_devices = ["cpu"]

        # CPU-only path: keep simple local processing unless explicitly parallelized.
        if (not torch.cuda.is_available()) or workers <= 1:
            _init_models(worker_devices[0])
            for i, task in enumerate(tasks, start=1):
                print(f"Processing record {i}/{total_records}...")
                _, payload = _process_sentence(task)
                if payload is None:
                    continue
                sentence, subject, relation, object_, payload_bytes = payload
                s_writer.writerow([sentence, subject, relation, object_])
                krv_db_file.write(payload_bytes)
                written += 1
            print(f"Wrote {written}/{total_records} records")
            return

        # GPU path: exactly one model server process per GPU device.
        # "workers" controls logical per-GPU request lanes, not model replicas.
        workers = max(1, workers)
        worker_devices = [d for d in worker_devices if d.startswith("cuda")]
        if not worker_devices:
            worker_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

        num_devices = len(worker_devices)
        workers_per_gpu = max(1, math.ceil(workers / num_devices))
        max_in_flight = max(1, workers_per_gpu * num_devices * 2)

        print(
            f"Starting {num_devices} model server(s) with {workers_per_gpu} worker lane(s) per GPU "
            "sharing one model instance per device."
        )

        pending = {}
        next_idx = 0
        processed = 0
        submitted = 0
        in_flight = 0

        ctx = mp.get_context("spawn")
        response_queue = ctx.Queue()
        request_queues = []
        servers = []

        try:
            for device in worker_devices:
                rq = ctx.Queue(maxsize=max(2, workers_per_gpu * 2))
                p = ctx.Process(target=_gpu_server_main, args=(device, rq, response_queue), daemon=True)
                p.start()
                request_queues.append(rq)
                servers.append(p)

            while processed < total_records:
                while submitted < total_records and in_flight < max_in_flight:
                    idx, sentence = tasks[submitted]
                    device_slot = idx % num_devices
                    request_queues[device_slot].put((idx, sentence))
                    submitted += 1
                    in_flight += 1

                idx, payload = response_queue.get()
                processed += 1
                in_flight -= 1
                print(f"Processed record {processed}/{total_records}...")
                pending[idx] = payload

                while next_idx in pending:
                    ordered_payload = pending.pop(next_idx)
                    if ordered_payload is not None:
                        sentence, subject, relation, object_, payload_bytes = ordered_payload
                        s_writer.writerow([sentence, subject, relation, object_])
                        krv_db_file.write(payload_bytes)
                        written += 1
                    next_idx += 1
        finally:
            for rq in request_queues:
                rq.put(None)
            for p in servers:
                p.join()

        print(f"Wrote {written}/{total_records} records")


def _default_workers():
    if torch.cuda.is_available():
        # One worker per GPU keeps one model instance per device.
        return max(1, torch.cuda.device_count())
    return 1


def _default_devices(workers):
    if not torch.cuda.is_available():
        return []
    gpus = torch.cuda.device_count()
    max_workers = min(max(1, workers), gpus)
    return [f"cuda:{i}" for i in range(max_workers)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Total logical worker lanes across GPUs (0 = auto from --workers-per-gpu)",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="When CUDA is available and --workers=0, logical worker lanes per GPU (one model load per device)",
    )
    parser.add_argument("--max-lines", type=int, default=max_lines_to_extract, help="Max number of source lines")
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="Comma-separated device list (e.g. cuda:0,cuda:1 or cpu)",
    )
    args = parser.parse_args()

    if args.workers > 0:
        workers = args.workers
    else:
        if torch.cuda.is_available():
            workers = max(1, torch.cuda.device_count() * max(1, args.workers_per_gpu))
        else:
            workers = _default_workers()

    workers = max(1, workers)
    if args.devices.strip():
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        devices = _default_devices(workers)

    devices = _unique_devices(devices)
    if len(devices) == 1 and devices[0].startswith("cuda"):
        print(f"Using single-device mode on {devices[0]}: one model instance will be loaded.")

    sentences = load_sentences(s_database_path, args.max_lines)
    generate_krv_database(sentences, workers=workers, devices=devices)
