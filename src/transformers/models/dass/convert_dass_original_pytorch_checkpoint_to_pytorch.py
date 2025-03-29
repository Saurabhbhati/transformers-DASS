import json
import torch
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

def rename_keys(state_dict, architecture):
    new_state_dict = {}
    for name in state_dict:
        param = state_dict[name]
        if not name.startswith("head"):
            new_state_dict["dass." + name] = param
        else:
            new_state_dict[name] = param

    return new_state_dict

@torch.no_grad()
def convert_dass_checkpoint(pytorch_file, converted_pytorch_save_path, model_size="small"):
    old_state_dict = torch.load(pytorch_file,weights_only=True)
    mod_state_dict = {}
    for key in old_state_dict:
        if 'module.v.classifier.head' in key:
            mod_key = key.replace("module.v.classifier.", "")
        else:
            mod_key = key.replace("module.v", "dass")
        if 'patch_embed' in key:
            mod_key = mod_key.replace("patch_embed", "patch_embeddings.projection")
            if '2' in key:
                mod_key = mod_key.replace("2", "1")
            if '5' in key:
                mod_key = mod_key.replace("5", "3")
            if '7' in key:
                mod_key = mod_key.replace("7", "4")
        if 'x_proj_weight' in key:
            mod_key = mod_key.replace("x_proj_weight", "x_proj.weight")
        if 'dt_projs_weight' in key:
            mod_key = mod_key.replace("dt_projs_weight", "dt_projs.weight")
        if 'downsample' in key:
            mod_key = mod_key.replace("downsample.1", "downsample.down")
            mod_key = mod_key.replace("downsample.3", "downsample.norm")
        if 'classifier' in key:
            mod_key = mod_key.replace("classifier.", "")
        mod_state_dict[mod_key] = old_state_dict[key]

    from transformers import DASSConfig, DASSModel, DASSForAudioClassification, DASSFeatureExtractor
    config = DASSConfig()
    id2label = json.load(open(hf_hub_download('huggingface/label-files', 'audioset-id2label.json', repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    #config.register_for_auto_class()
    #DASSModel.register_for_auto_class("AutoModel")
    #DASSForAudioClassification.register_for_auto_class("AutoModelForAudioClassification")
    #DASSFeatureExtractor.register_for_auto_class("AutoFeatureExtractor")

    model = DASSForAudioClassification(config)
    model.eval()
    model.load_state_dict(mod_state_dict,strict=True)
    feature_extractor = DASSFeatureExtractor()
    # torch.save(mod_state_dict, converted_pytorch_save_path)

    if converted_pytorch_save_path is not None:
        Path(converted_pytorch_save_path).mkdir(exist_ok=True)
        print(f"Saving model {model_size} to {converted_pytorch_save_path}")
        model.save_pretrained(converted_pytorch_save_path)
        print(f"Saving feature extractor to {converted_pytorch_save_path}")
        feature_extractor.save_pretrained(converted_pytorch_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pytorch_file",type=str,default=None,required=True,
        help="Path to local pytorch file of a DASS checkpoint.",
    )
    parser.add_argument(
        "--converted_pytorch_save_path",default=None,type=str,required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--model_size",default="small",type=str,
        help="The size of the model to convert. Can be 'small' or 'medium'.",
    )

    args = parser.parse_args()
    convert_dass_checkpoint(args.pytorch_file, args.converted_pytorch_save_path, args.model_size)