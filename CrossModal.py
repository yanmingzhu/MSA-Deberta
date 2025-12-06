from transformers import DebertaV2Model
import torch
import torch.nn as nn
from config import *

class CrossAttenion(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super().__init__()
        self.dim = dim
        self.nhead = nhead

        self.audio_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.vision_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
           nn.Linear(dim, dim*4),
           nn.ReLU(),
           nn.Linear(dim*4, dim),
           nn.Dropout(dropout)
        )

    def forward(self, text, audio, vision, masks):
        audio_output, _ = self.audio_attention(
            query=text, 
            key=audio,   
            value=audio,      
            key_padding_mask=None, 
            need_weights=False
        )
        vision_output, _ = self.vision_attention(
            query=text, 
            key=vision,   
            value=vision,      
            key_padding_mask=None, 
            need_weights=False
        )

        fused_embedding = text + audio_output + vision_output
        fused_embedding = self.layer_norm(fused_embedding)
        mlp_out = self.mlp(fused_embedding)

        return fused_embedding + mlp_out


class CrossModalDebertaMM(nn.Module):
  def __init__(self, hidden_dim=512, nhead=8, nlayer=4):
    super(CrossModalDebertaMM, self).__init__()
    self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')
    self.cross_attention = CrossAttenion(TEXT_DIM, nhead=8, dropout=0.2)

    self.audio_proj = nn.Linear(AUDIO_DIM, TEXT_DIM)
    self.vision_proj = nn.Linear(VISION_DIM, TEXT_DIM)

    self.fc1 = nn.Linear(TEXT_DIM, hidden_dim)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(hidden_dim, 1)

  def forward(self, text, audio, vision, masks=None):
    text_embedding = self.deberta.embeddings(text)
    audio_embedding = self.audio_proj(audio)
    vision_embedding = self.vision_proj(vision)

    injection_layer = int(len(self.deberta.encoder.layer) / 2) # inject cross attention half in the stack

    embedding = text_embedding

    for i in range(injection_layer):
       print(f"layer {i}")
       layer = self.deberta.encoder.layer[i]
       layer_out = layer(hidden_states=embedding, 
                         attention_mask=masks, 
                         output_attentions=False)
       embedding = layer_out[0]
    
    embedding = self.cross_attention(embedding, audio_embedding, vision_embedding, masks)

    rel_embeddings = self.deberta.encoder.get_rel_pos_embeddings()

    for i in range(injection_layer, len(self.deberta.encoder.layer)):
       embedding = self.deberta.encoder.layer[i](embedding, attention_masks=masks, rel_pos=rel_embeddings)

    bert_output = self.deberta(inputs_embeds=embedding, attention_mask=masks)

    last_hidden = bert_output.last_hidden_state
    cls = last_hidden[:, 0]

    pred = self.fc1(cls)
    pred = torch.relu(pred)
    pred = self.dropout(pred)
    pred = self.fc2(pred)
    #print(f"final pred ==== {pred[2:4, :].squeeze()}")

    return pred