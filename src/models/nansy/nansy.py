from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Nansy(nn.Module):
  def __init__(self,
               config,
               cqt,
               wav2vec2,
               melspec,
               pitch,
               linguistic,
               timbre,
               frame_synthesizer,
               synthesizer
               ) -> None:
    super().__init__()
    self.config = config
    self.cqt = cqt
    self.wav2vec2 = wav2vec2
    self.melspec = melspec
    self.pitch = pitch
    self.linguistic = linguistic
    self.timbre = timbre
    self.frame_synthesizer = frame_synthesizer
    self.synthesizer = synthesizer

    self.register_buffer(
        'pitch_bins',
        # linear space in log-scale
        torch.linspace(
            np.log(config.pitch_start),
            np.log(config.pitch_end),
            config.pitch_f0_bins).exp())

    self.cqt_bins = self.cqt.bins
    self.pitch_freq = self.pitch.freq
    self.cqt_center = (self.cqt_bins - self.pitch_freq) // 2

  def analyze_pitch(self, inputs: torch.Tensor, index: Optional[int] = None) \
                    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate the pitch and periodical, aperiodical amplitudes.
        Args:
            inputs: [torch.float32; [B, T]], input speech signal.
            index: CQT start index, use `cqt_center` if index is None
        Returns:
            [torch.float32; [B, cqt_bins, N]], CQT features.
            [torch.float2; [B, N]], frame-level pitch and amplitude sequence.
        """
        # [B, cqt_bins, N(=T / cqt_hop)]
        ## TODO: log-scale or not.
        cqt = self.cqt(inputs)
        print(cqt.size())
        if cqt.size(2) % 4 != 0:
          padsize = 4 - cqt.size(2) % 4
          cqt = F.pad(cqt, (0, padsize), mode="constant")
        print(cqt.size())
        # alias
        freq = self.pitch_freq
        if index is None:
            index = self.cqt_center
        # [B, N, f0_bins], [B, N], [B, N]
        pitch_bins, p_amp, ap_amp = self.pitch.forward(
            cqt[:, index:index + freq])
        # [B, N]
        pitch = (pitch_bins * self.pitch_bins).sum(dim=-1)
        # [B, cqt_bins, N], [B, N]
        return cqt, pitch, p_amp, ap_amp

  def analyze_linguistic(self, inputs: torch.Tensor) -> torch.Tensor:
      """Analyze the linguistic informations from inputs.
      Args:
          inputs: [torch.float32; [B, T]], input speech signal.
      Returns:
          [torch.float32; [B, ling_hiddens, S]], linguistic informations.
      """
      # [B, S, w2v2_channels]
      w2v2 = self.wav2vec2.forward(inputs)
      # [B, ling_hiddens, S]
      return self.linguistic.forward(w2v2.transpose(1, 2))

  def analyze_timbre(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      """Analyze the timbre informations from inputs.
      Args:
          inputs: [torch.float32; [B, T]], input speech signal.
      Returns:
          [torch.float32; [B, timb_global]], global timbre emebddings.
          [torch.float32; [B, timb_timbre, timb_tokens]], timbre token bank.
      """
      # [B, mel, T / mel_hop]
      mel = self.melspec.forward(inputs)
      # [B, timb_global], [B, timb_timbre, timb_tokens]
      return self.timbre.forward(mel)

  def analyze(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
      """Analyze the input signal.
      Args:
          inputs: [torch.float32; [B, T]], input speech signal.
      Returns;
          analyzed featuers,
              cqt: [torch.float32; []], CQT features.
              pitch, p_amp, ap_amp: [torch.float2; [B, N]],
                  frame-level pitch and amplitude sequence.
              ling: [torch.float32; [B, ling_hiddens, S]], linguistic informations.
              timbre_global: [torch.float32; [B, timb_global]], global timbre emebddings.
              timbre_bank: [torch.float32; [B, timb_timbre, timb_tokens]], timbre token bank.
      """
      # [], [B, N]
      cqt, pitch, p_amp, ap_amp = self.analyze_pitch(inputs)
      # [B, ling_hiddens, S]
      ling = self.analyze_linguistic(inputs)
      # [B, timb_global], [B, timb_timbre, timb_tokens]
      timbre_global, timbre_bank = self.analyze_timbre(inputs)
      return {
          'cqt': cqt,
          'pitch': pitch,
          'p_amp': p_amp,
          'ap_amp': ap_amp,
          'ling': ling,
          'timbre_global': timbre_global,
          'timbre_bank': timbre_bank}

  def synthesize(self,
                  pitch: torch.Tensor,
                  p_amp: torch.Tensor,
                  ap_amp: torch.Tensor,
                  ling: torch.Tensor,
                  timbre_global: torch.Tensor,
                  timbre_bank: torch.Tensor,
                  noise: Optional[torch.Tensor] = None) \
          -> Tuple[torch.Tensor, torch.Tensor]:
      """Synthesize the signal.
      Args:
          pitch, p_amp, ap_amp: [torch.float32; [B, N]], frame-level pitch, amplitude sequence.
          ling: [torch.float32; [B, ling_hiddens, S]], linguistic features.
          timbre_global: [torch.float32; [B, timb_global]], global timbre.
          timbre_bank: [torch.float32; [B, timb_timbre, timb_tokens]], timbre token bank.
          noise: [torch.float32; [B, T]], predefined noise for excitation signal, if provided.
      Returns:
          [torch.float32; [B, T]], excitation and synthesized speech signal.
      """
      # S
      ling_len = ling.shape[-1]
      # [B, 3, S]
      pitch_rel = F.interpolate(torch.stack([pitch, p_amp, ap_amp], dim=1), size=ling_len)
      # [B, 3 + ling_hiddens + timb_global, S]
      contents = torch.cat([
          pitch_rel, ling, timbre_global[..., None].repeat(1, 1, ling_len)], dim=1)
      # [B, timbre_global, S]
      timbre_sampled = self.timbre.sample_timbre(contents, timbre_global, timbre_bank)
      # [B, ling_hiddens, S]
      frame = self.frame_synthesizer.forward(ling, timbre_sampled)
      # [B, T], [B, T]
      return self.synthesizer.forward(pitch, p_amp, ap_amp, frame, noise)

  def forward(self, inputs: torch.Tensor, noise: Optional[torch.Tensor] = None) \
          -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
      """Reconstruct input audio.
      Args:
          inputs: [torch.float32; [B, T]], input signal.
          noise: [torch.float32; [B, T]], predefined noise for excitation, if provided.
      Returns:
          [torch.float32; [B, T]], reconstructed.
          auxiliary outputs, reference `Nansypp.analyze`.
      """
      features = self.analyze(inputs)
      # [B, T]
      excitation, synth = self.synthesize(
          features['pitch'][:, :-1], # 길이를 맞추기 위해 f0, p_amp, ap_amp 모두 1씩 줄임
          features['p_amp'][:, :-1],
          features['ap_amp'][:, :-1],
          features['ling'],
          features['timbre_global'],
          features['timbre_bank'],
          noise=noise)
      # update
      features['excitation'] = excitation
      return synth, features


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nansy.yaml")
    nansy = hydra.utils.instantiate(cfg.nansy)

    x = torch.rand(2, 24000)
    # analysis_feats = nansy.analyze(x)
    # for k, v in analysis_feats.items():
    #     print(f"{k}: {v.size()}")

    synth, analysis_feats = nansy(x)
    for k, v in analysis_feats.items():
        print(f"{k}: {v.size()}")
    print(f"output: {synth.size()}")