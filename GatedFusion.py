from transformers import DebertaV2Model
import torch
import torch.nn as nn
from config import *

class GatedDebertaMM(nn.Module):
  def __init__(self, hidden_dim=512, nhead=8, nlayer=4):
    super(GatedDebertaMM, self).__init__()
    self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')

    audio_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)

    self.audio_proj = nn.Linear(AUDIO_DIM, TEXT_DIM)
    self.vision_proj = nn.Linear(VISION_DIM, TEXT_DIM)

    self.audio_norm = nn.LayerNorm(TEXT_DIM)
    self.vision_norm = nn.LayerNorm(TEXT_DIM)

    self.audioGate = nn.Sequential(
      nn.Linear(TEXT_DIM*3, TEXT_DIM),
      nn.Sigmoid()
    )

    self.visionGate = nn.Sequential(
      nn.Linear(TEXT_DIM*3, TEXT_DIM),
      nn.Sigmoid()
    )

    self.fc1 = nn.Linear(TEXT_DIM, hidden_dim)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(hidden_dim, 1)

  def forward(self, text, audio, vision, masks=None):
    text_embedding = self.deberta.embeddings(text)
    audio_embedding = self.audio_proj(audio)
    #audio_embedding = self.audio_norm(audio_embedding)

    vision_embedding = self.vision_proj(vision)
    #vision_embedding = self.vision_norm(vision_embedding)

    embedding_concat = torch.cat([text_embedding, audio_embedding, vision_embedding], dim=-1)
    audio_gate = self.audioGate(embedding_concat)
    vision_gate = self.visionGate(embedding_concat)

    fused_embedding = text_embedding + audio_gate * audio_embedding + vision_gate * vision_embedding

    bert_output = self.deberta(inputs_embeds=fused_embedding, attention_mask=masks)

    last_hidden = bert_output.last_hidden_state
    cls = last_hidden[:, 0]

    pred = self.fc1(cls)
    pred = torch.relu(pred)
    pred = self.dropout(pred)
    pred = self.fc2(pred)
    #print(f"final pred ==== {pred[2:4, :].squeeze()}")

    return pred