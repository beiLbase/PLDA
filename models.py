import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from config import DEVICE, MAX_LEN, LOCAL_BERT_PATH
from prompt_template import ALL_PROMPTS

class UltimateProtoNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True
        )
        self.bert = BertModel.from_pretrained(
            LOCAL_BERT_PATH,
            local_files_only=True
        ).to(DEVICE)
        self.temperature = 0.07

    def encode_sent(self, sent, e1, e2):
        feats = []
        for prompt in ALL_PROMPTS:
            text = prompt.build(sent, e1, e2)
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
            mask = (inputs.input_ids[0] == self.tokenizer.mask_token_id).nonzero()
            idx = mask[0].item() if len(mask) > 0 else 0
            feats.append(self.bert(**inputs).last_hidden_state[0, idx])
        return torch.stack(feats).mean(0)

    def encode_rel(self, desc):
        inputs = self.tokenizer(desc, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
        return self.bert(**inputs).last_hidden_state[:, 0, :].squeeze(0)

    def forward(self, support, query, rel_descs):
        s_feat = []
        for sent, e1, e2, desc in support:
            s_sent = self.encode_sent(sent, e1, e2)
            s_rel = self.encode_rel(desc)
            fused = (s_sent + s_rel) / 2
            s_feat.append(fused)
        s_feat = torch.stack(s_feat)
        q_feat = []
        for sent, e1, e2, desc in query:
            q_sent = self.encode_sent(sent, e1, e2)
            q_rel = self.encode_rel(desc)
            fused = (q_sent + q_rel) / 2
            q_feat.append(fused)
        q_feat = torch.stack(q_feat)
        prototypes = []
        n_way = len(query)
        shot = len(support) // n_way
        for i in range(n_way):
            g = s_feat[i*shot:(i+1)*shot]
            prototypes.append(g.mean(0))
        prototypes = torch.stack(prototypes)
        q_feat = torch.nn.functional.normalize(q_feat, dim=-1)
        prototypes = torch.nn.functional.normalize(prototypes, dim=-1)
        return torch.matmul(q_feat, prototypes.T) / self.temperature