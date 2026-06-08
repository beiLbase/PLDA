import json
import random
def load_fewrel():
    def load_file(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        rel2id = {r:i for i,r in enumerate(data.keys())}
        for rel_id, instances in data.items():
            rel_desc = REL_DESCRIPTION.get(rel_id, "relation")
            for ins in instances:
                out.append({
                    "sent": " ".join(ins["tokens"]),
                    "e1": ins["h"][0],
                    "e2": ins["t"][0],
                    "rel": rel2id[rel_id],
                    "rel_desc": rel_desc
                })
        return out
    return load_file("train.json"), load_file("val.json"), load_file("test.json")

def sample_episode(dataset, N_way=5, K_shot=1):
    all_rels = list({d["rel"] for d in dataset})
    selected = random.sample(all_rels, N_way)
    support, query, labels, rel_descs = [], [], [], []

    for idx, rel in enumerate(selected):
        samples = [d for d in dataset if d["rel"] == rel]
        selected_samples = random.sample(samples, K_shot + 1)


        for s in selected_samples[:K_shot]:
            support.append((s["sent"], s["e1"], s["e2"], s["rel_desc"]))
    
        q = selected_samples[-1]
        query.append((q["sent"], q["e1"], q["e2"], q["rel_desc"]))
        labels.append(idx)
        rel_descs.append(q["rel_desc"])

    return support, query, labels, rel_descs