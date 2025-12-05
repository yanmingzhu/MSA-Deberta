from transformers import DebertaV2Model
import torch
import torch.nn as nn
from config import *

class EarlyDebertaMM(nn.Module):
  def __init__(self, hidden_dim=512, nhead=8, nlayer=4):
    super(EarlyDebertaMM, self).__init__()
    self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-base')

    self.audio_proj = nn.Linear(AUDIO_DIM, TEXT_DIM)
    self.vision_proj = nn.Linear(VISION_DIM, TEXT_DIM)

    self.fc1 = nn.Linear(TEXT_DIM, hidden_dim)
    self.dropout = nn.Dropout(0.2)
    self.fc2 = nn.Linear(hidden_dim, 1)

  def forward(self, text, audio, vision, masks=None):
    '''
    text[torch.isinf(text)] = 0.0
    audio[torch.isinf(audio)] = 0.0
    vision[torch.isinf(vision)] = 0.0

    x_concat = torch.cat((text, audio, vision), dim=2)

    if torch.isinf(x_concat).any():
      inf_indices = torch.nonzero(torch.isinf(x_concat), as_tuple=False)
      print(f"infinities {inf_indices}")
      print(x_concat[inf_indices[0], inf_indices[1], :])
    '''

    text_embedding = self.deberta.embeddings(text)
    audio_embedding = self.audio_proj(audio)
    vision_embedding = self.vision_proj(vision)
    fused_embedding = text_embedding + audio_embedding + vision_embedding
    #fused_embedding = self.deberta.embeddings.LayerNorm(fused_embedding)

    bert_output = self.deberta(inputs_embeds=fused_embedding, attention_mask=masks)

    last_hidden = bert_output.last_hidden_state
    cls = last_hidden[:, 0]

    pred = self.fc1(cls)
    pred = torch.relu(pred)
    pred = self.dropout(pred)
    pred = self.fc2(pred)
    #print(f"final pred ==== {pred[2:4, :].squeeze()}")

    return pred