import csv

import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Instruct-2507"

sentence_gen_model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
sentence_gen_tokenizer = AutoTokenizer.from_pretrained(model_name)

k_embed_dim = 256
r_embed_dim = 128
v_embed_dim = 256

s_database_path = "datasets/books_large_p1.txt"

output_krv_database_path = "krv-database.bin"
output_s_database_path = "s-database.csv"

total_lines_to_extract = 16384


def generate_krv_sentences(sentence):
    prompt = f"""
Extract a semantic triple (Subject, Object, Relation) from the sentence.

Definitions:
- Subject: the entity performing the action (the agent), even if the sentence is in passive voice
- Object: the entity receiving the action
- Relation: a concise verb phrase describing the action from Subject to Object

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
    model_inputs = sentence_gen_tokenizer([text], return_tensors="pt").to(
        sentence_gen_model.device
    )

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
        elif line.startswith("Object:"):
            object_ = line[len("Object:") :].strip()
        elif line.startswith("Relation:"):
            relation = line[len("Relation:") :].strip()

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
            if line:
                sentences.append(line)
                if len(sentences) >= num_sentences:
                    break
    return sentences


def generate_krv_database(sentences):
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

        for sentence in sentences:
            triple = generate_krv_sentences(sentence)
            if triple is None:
                continue

            subject, object_, relation = triple

            # Generate embeddings
            subject_embedding = embedding_model.encode(
                subject,
                truncate_dim=k_embed_dim,
                convert_to_numpy=True,
            )
            object_embedding = embedding_model.encode(
                object_,
                truncate_dim=v_embed_dim,
                convert_to_numpy=True,
            )
            relation_embedding = embedding_model.encode(
                relation,
                truncate_dim=r_embed_dim,
                convert_to_numpy=True,
            )

            # Ensure exact dimensions and little-endian f16 payload.
            k_emb = np.zeros((k_embed_dim,), dtype="<f2")
            r_emb = np.zeros((r_embed_dim,), dtype="<f2")
            v_emb = np.zeros((v_embed_dim,), dtype="<f2")

            subject_embedding = np.asarray(subject_embedding, dtype=np.float32).reshape(-1)
            relation_embedding = np.asarray(relation_embedding, dtype=np.float32).reshape(-1)
            object_embedding = np.asarray(object_embedding, dtype=np.float32).reshape(-1)

            k_take = min(k_embed_dim, subject_embedding.shape[0])
            r_take = min(r_embed_dim, relation_embedding.shape[0])
            v_take = min(v_embed_dim, object_embedding.shape[0])

            k_emb[:k_take] = subject_embedding[:k_take].astype("<f2")
            r_emb[:r_take] = relation_embedding[:r_take].astype("<f2")
            v_emb[:v_take] = object_embedding[:v_take].astype("<f2")

            # Save to databases
            s_writer.writerow([sentence, subject, relation, object_])
            krv_db_file.write(
                k_emb.tobytes()
                + r_emb.tobytes()
                + v_emb.tobytes()
            )


if __name__ == "__main__":
    sentences = load_sentences(s_database_path, total_lines_to_extract)
    generate_krv_database(sentences)
