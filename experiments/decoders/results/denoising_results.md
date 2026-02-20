# Denoising Results

Two denoising strategies using trained fMRI flat map MAE models.

- **Mask reconstruction denoising**: Generate 20 random masked reconstruction samples and average into a single composite, only including predictions for unobserved patches.
- **Autoencoder denoising**: Run the encoder on fully observed inputs (no masking) and decode all patches, including visible.

## Results

| Model | Dataset | Link |
|-------|---------|------|
| attn_reg1_pep4 | HCP | [results](denoising_attn_reg1_pep4_hcp.md) |
| attn_reg1_pep4 | NSD | [results](denoising_attn_reg1_pep4_nsd.md) |
| crossreg_reg4_pep4 | HCP | [results](denoising_crossreg_reg4_pep4_hcp.md) |
| crossreg_reg4_pep4 | NSD | [results](denoising_crossreg_reg4_pep4_nsd.md) |
