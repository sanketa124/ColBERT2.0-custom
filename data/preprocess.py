import json
import random

asqa_path = "asqa.json"
collection_path = "asqa_collection.tsv"
queries_path = "asqa_queries.tsv"
triplets_path = "asqa_triplets.json"

with open(asqa_path) as f:
    data = json.load(f)

queries = []
collection = []
triplets = []

passage_to_pid = {}
query_id = 0

# Flatten all answers in the order they appear and index them
for ex in data["dev"].values():
    for qa in ex["qa_pairs"]:
        for answer in qa["short_answers"]:
            if answer not in passage_to_pid:
                pid = len(collection)
                passage_to_pid[answer] = pid
                collection.append(f"{pid}\t{answer}")

# Process queries and triplets
for ex in data["dev"].values():
    # Ambiguous question triplet (with unique query_id)
    ambiguous_query_id = query_id  # Store original query_id for the ambiguous question
    queries.append(f"{query_id}\t{ex['ambiguous_question']}")
    pos_pids = set()  # Use set to ensure unique PIDs for the ambiguous question

    for qa in ex["qa_pairs"]:
        # Add sub-question triplet
        query_id += 1  # Increment query_id for each sub-question
        queries.append(f"{query_id}\t{qa['question']}")

        # Collect all associated positive passage IDs in the order of short_answers
        pos_pids_qa = [passage_to_pid[ans] for ans in qa["short_answers"] if ans in passage_to_pid]
        pos_pids.update(pos_pids_qa)  # Add positive pids to the set to ensure uniqueness

        # Create a triplet for this sub-question
        all_pids = set(passage_to_pid.values())
        neg_pids = random.sample(all_pids - pos_pids, min(2, len(all_pids) - len(pos_pids)))

        # Combine and give random scores for positive and negative passages
        scored_pos = [[pid, round(random.uniform(7.5, 10.0), 6)] for pid in pos_pids_qa]
        scored_neg = [[pid, round(random.uniform(1.0, 7.0), 6)] for pid in neg_pids]

        # Format triplet in exact requested format: commas between pid, score pairs
        triplet = f"[{query_id},"
        triplet += ",".join([f"[{pid},{score}]" for pid, score in scored_pos + scored_neg]) + "]"
        triplets.append(triplet)

    # Triplet for ambiguous question
    query_id += 1
    if pos_pids:
        all_pids = set(passage_to_pid.values())
        neg_pids = random.sample(all_pids - pos_pids, min(2, len(all_pids) - len(pos_pids)))
        scored_pos = [[pid, round(random.uniform(7.5, 10.0), 6)] for pid in pos_pids]
        scored_neg = [[pid, round(random.uniform(1.0, 7.0), 6)] for pid in neg_pids]

        # Reference ambiguous question as triplet for ambiguous question
        triplet = f"[{ambiguous_query_id},"
        triplet += ",".join([f"[{pid},{score}]" for pid, score in scored_pos + scored_neg]) + "]"
        triplets.append(triplet)

# Write files
with open(collection_path, "w", encoding="utf-8") as f:
    f.write("\n".join(collection))

with open(queries_path, "w", encoding="utf-8") as f:
    f.write("\n".join(queries))

with open(triplets_path, "w", encoding="utf-8") as f:
    f.write("\n".join(triplets))

print("Files generated successfully!")
