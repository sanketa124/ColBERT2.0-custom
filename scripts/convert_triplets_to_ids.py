import json
import argparse
from collections import defaultdict

def read_collection(collection_path):
    text_to_pid = {}
    with open(collection_path, 'r', encoding='utf-8') as f:
        for line in f:
            pid, passage = line.strip().split('\t', 1)
            text_to_pid[passage] = pid
    return text_to_pid

def build_query_dict(jsonl_path):
    queries = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            queries[i] = data["query"]
    return queries

def convert_triplets(jsonl_path, collection_path, output_path):
    text_to_pid = read_collection(collection_path)
    missing = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            obj = json.loads(line)
            query = obj["query"]
            pos = obj["positive"]
            negs = obj["negatives"]

            try:
                pid_pos = text_to_pid[pos]
                pid_negs = [text_to_pid[neg] for neg in negs if neg in text_to_pid]

                if pid_negs:
                    for neg_id in pid_negs:
                        f_out.write(f"{query}\t{pid_pos}\t{neg_id}\n")
                else:
                    print(f"Skipping triplet with no valid negatives: {query}")
            except KeyError as e:
                missing.add(str(e))

    if missing:
        print("Some passages not found in collection:")
        for miss in list(missing)[:10]:
            print("  ", miss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--triplets', required=True, help='Path to JSONL triplets file')
    parser.add_argument('--collection', required=True, help='Path to collection.tsv')
    parser.add_argument('--output', required=True, help='Path to save triplets.train.tsv')

    args = parser.parse_args()
    convert_triplets(args.triplets, args.collection, args.output)
