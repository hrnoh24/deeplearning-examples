import torch
import torch.nn as nn
from transformers import Wav2Vec2ForPreTraining

# TODO: 'wav2vec2.encoder.pos_conv_embed.conv.weight_g', 'wav2vec2.encoder.pos_conv_embed.conv.weight_v'
#      위 두 키에 대해서 읽을 수 있도록 체크포인트 내부의 키를 변경해주어야 함
class MMS(nn.Module):
    def __init__(self,
                 load_pretrained_mms=True, 
                 device='cpu',
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mms = None
        self.device = device
        
        if load_pretrained_mms: self._load_mms()
        
    def _load_mms(self):
        with torch.no_grad():
            mms = Wav2Vec2ForPreTraining.from_pretrained("facebook/mms-1b")
            mms = mms.eval()
            self.mms = mms.to(self.device)
        
    def feats(self, x, layer=None):
        """Extract mms features

        Args:
            x (torch.FloatTensor): audio waveforms [B, T]
            layer (int, optional): mms layer. Defaults to None.

        Returns:
            mms_features: l-th layer's feature. [B, T//320-1, 1280]
        """
        with torch.no_grad():
            outputs = self.mms(x, output_hidden_states=True)
            
        if layer is None:
            return outputs.hidden_states
        else:
            return outputs.hidden_states[layer]
        
        
class MMSSoft(MMS):
    def __init__(self, 
                 num_label_embeddings=2000,
                 mms_layer=9,
                 load_pretrained_mms=True, 
                 device='cpu', 
                 *args, 
                 **kwargs) -> None:
        super().__init__(load_pretrained_mms, device, *args, **kwargs)
        self.mms_layer = mms_layer
        self.load_pretrained_mms = load_pretrained_mms
        self.proj = nn.Linear(1280, 256)
        self.label_embedding = nn.Embedding(num_label_embeddings, 256)
        
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """logits between soft features and discrete codes

        Args:
            x (torch.Tensor): soft features [B, T, 256]

        Returns:
            logits (torch.Tensor): logits [B, T, num_label_embeddings]
        """
        logits = torch.cosine_similarity(
            x.unsqueeze(2),
            self.label_embedding.weight.unsqueeze(0).unsqueeze(0),
            dim=-1,
        )
        return logits / 0.1
    
    def units(self, x):
        """Extract soft units

        Args:
            x (torch.FloatTensor) [B, T] : audio waveform or mms features. 
                                           If load_pretrained_mms is True, it has to be audio waveforms. 

        Returns:
            soft units (torch.FloatTensor): [B, T//320-1, 256]
        """
        if self.load_pretrained_mms:
            feats = super().feats(x, layer=self.mms_layer)
        else:
            feats = x
        units = self.proj(feats)
        return units
    
if __name__=="__main__":
    x = torch.rand(1, 16000)
    model = MMSSoft()
    
    units = model.units(x)
    feats = model.feats(x, layer=9)
    logit = model.logits(units)
        
    print(feats.shape, units.shape, logit.shape)