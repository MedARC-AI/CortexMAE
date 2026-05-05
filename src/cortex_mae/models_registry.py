import cortex_mae.models_mae as models_mae

HF_PREFIX = "hf://medarc/CortexMAE"

CORTEX_MAE_MODEL_REGISTRY = {
    # input space
    "cortex_mae_flat": "input_space_v3/flat_lr1e-3_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r2": "input_space_v3/flat_lr1e-3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r3": "input_space_v3/flat_lr1e-3_3/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r4": "input_space_v3/flat_lr1e-3_4/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r5": "input_space_v3/flat_lr1e-3_5/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r6": "input_space_v3/flat_lr1e-3_6/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r7": "input_space_v3/flat_lr1e-3_7/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_r8": "input_space_v3/flat_lr1e-3_8/pretrain/checkpoint-last.pth",
    "cortex_mae_volume": "input_space_v3/mni_cortex_lr1e-3_1/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r2": "input_space_v3/mni_cortex_lr1e-3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r3": "input_space_v3/mni_cortex_lr1e-3_3/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r4": "input_space_v3/mni_cortex_lr1e-3_4/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r5": "input_space_v3/mni_cortex_lr1e-3_5/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r6": "input_space_v3/mni_cortex_lr1e-3_6/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r7": "input_space_v3/mni_cortex_lr1e-3_7/pretrain/checkpoint-last.pth",
    "cortex_mae_volume_r8": "input_space_v3/mni_cortex_lr1e-3_8/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel": "input_space_v3/schaefer400_lr3e-4_1/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r2": "input_space_v3/schaefer400_lr3e-4_2/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r3": "input_space_v3/schaefer400_lr3e-4_3/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r4": "input_space_v3/schaefer400_lr3e-4_4/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r5": "input_space_v3/schaefer400_lr3e-4_5/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r6": "input_space_v3/schaefer400_lr3e-4_6/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r7": "input_space_v3/schaefer400_lr3e-4_7/pretrain/checkpoint-last.pth",
    "cortex_mae_parcel_r8": "input_space_v3/schaefer400_lr3e-4_8/pretrain/checkpoint-last.pth",
    # subcortical
    "cortex_mae_a424": "subcortical/a424_lr3e-4_1/pretrain/checkpoint-last.pth",
    "cortex_mae_a424_r2": "subcortical/a424_lr3e-4_2/pretrain/checkpoint-last.pth",
    "cortex_mae_a424_r3": "subcortical/a424_lr3e-4_3/pretrain/checkpoint-last.pth",
    "cortex_mae_a424_r4": "subcortical/a424_lr3e-4_4/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3": "subcortical/schaefer400_tians3_lr3e-4_1/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3_r2": "subcortical/schaefer400_tians3_lr3e-4_2/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3_r3": "subcortical/schaefer400_tians3_lr3e-4_3/pretrain/checkpoint-last.pth",
    "cortex_mae_s400ts3_r4": "subcortical/schaefer400_tians3_lr3e-4_4/pretrain/checkpoint-last.pth",
    # data scaling
    "cortex_mae_flat_n100": "data_scaling/n100_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n100_r2": "data_scaling/n100_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n200": "data_scaling/n200_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n200_r2": "data_scaling/n200_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n400": "data_scaling/n400_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n400_r2": "data_scaling/n400_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n800": "data_scaling/n800_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n800_r2": "data_scaling/n800_2/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n1600": "data_scaling/n1600_1/pretrain/checkpoint-best.pth",
    "cortex_mae_flat_n1600_r2": "data_scaling/n1600_2/pretrain/checkpoint-best.pth",
    # model scaling (depth)
    "cortex_mae_flat_d3": "model_scaling/d3/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d3_r2": "model_scaling/d3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d6": "model_scaling/d6/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d6_r2": "model_scaling/d6_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d9": "model_scaling/d9/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d9_r2": "model_scaling/d9_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d12": "input_space_v3/flat_lr1e-3_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d12_r2": "input_space_v3/flat_lr1e-3_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d15": "model_scaling/d15/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_d15_r2": "model_scaling/d15_2/pretrain/checkpoint-last.pth",
    # temporal patch size
    "cortex_mae_flat_pt1": "t_patch_size/pt-1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt2": "t_patch_size/pt-2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt4": "t_patch_size/pt-4/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt8": "t_patch_size/pt-8/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_pt16": "t_patch_size/pt-16/pretrain/checkpoint-last.pth",
    # spatial patch size
    "cortex_mae_flat_p8": "patch_size/patch8/pretrain/checkpoint-last.pth",
    # denoising model
    "cortex_mae_flat_denoise": "decoders/attn_reg1_pep4/pretrain/checkpoint-last.pth",
    # cross-register decoding
    "cortex_mae_flat_crossreg1": "decoders/crossreg_reg1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_crossreg4": "decoders/crossreg_reg4/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_crossreg16": "decoders/crossreg_reg16/pretrain/checkpoint-last.pth",
    # training data
    "cortex_mae_flat_rest": "rest_v_task/rest_ep50_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_rest_r2": "rest_v_task/rest_ep50_2/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_task": "rest_v_task/task_ep50_1/pretrain/checkpoint-last.pth",
    "cortex_mae_flat_task_r2": "rest_v_task/task_ep50_2/pretrain/checkpoint-last.pth",
}


def create_model(
    model_name: str, pretrained: bool = True, **kwargs
) -> models_mae.MaskedAutoencoderViT:
    ckpt_path = CORTEX_MAE_MODEL_REGISTRY[model_name]
    ckpt_path = f"{HF_PREFIX}/{ckpt_path}"
    model = models_mae.MaskedAutoencoderViT.from_checkpoint(ckpt_path, **kwargs)
    if not pretrained:
        model.init_weights()
    return model


def list_models() -> list[str]:
    return list(CORTEX_MAE_MODEL_REGISTRY)


def get_model_input_space(model_name: str):
    input_space = model_name.split("_")[2]
    input_space = {
        "volume": "mni_cortex",
        "parcel": "schaefer400",
        "s400ts3": "schaefer400_tians3",
    }.get(input_space, input_space)
    return input_space
