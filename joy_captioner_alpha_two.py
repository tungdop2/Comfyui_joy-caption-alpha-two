import os
import logging
from PIL import Image
from pathlib import Path
import torch
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from torch import nn
from torchvision import transforms as T
import torchvision.transforms.functional as TVF

import comfy.model_management as model_management
import comfy.utils
import folder_paths
from string import Template

logger = logging.getLogger("JoyCaptionerAlphaTwo")
logger.setLevel(logging.INFO)

# Configuration
repo_dir = os.path.dirname(os.path.realpath(__file__))
LLM_REPO_ID = "unsloth/Meta-Llama-3.1-8B-Instruct"
CLIP_PATH = "siglip-so400m-patch14-384"
ADAPTER_PATH = os.path.join(folder_paths.models_dir, "joy-caption-alpha-two")
LLM_PATH = os.path.join(folder_paths.models_dir, "llm", LLM_REPO_ID)
CHECKPOINT_PATH = os.path.join(ADAPTER_PATH, "cgrkzexw-599808")


# Define Image Adapter Model
class ImageAdapter(nn.Module):
    def __init__(
            self,
            input_features: int,
            output_features: int,
            ln1: bool,
            pos_emb: bool,
            num_image_tokens: int,
            deep_extract: bool,
    ):
        super().__init__()
        self.deep_extract = deep_extract

        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = (
            None
            if not pos_emb
            else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        )

        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat(
                (
                    vision_outputs[-2],
                    vision_outputs[3],
                    vision_outputs[7],
                    vision_outputs[13],
                    vision_outputs[20],
                ),
                dim=-1,
            )
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)
        if self.pos_emb is not None:
            x = x + self.pos_emb
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        other_tokens = self.other_tokens(
            torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(
                x.shape[0], -1
            )
        )
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)

        return x


class JoyCaptioner:
    PROMPT_MAP = {
        "descriptive": "write a $lengh descriptive caption for this image in a $tone tone.",
        "training": "write a $length stable diffusion prompt for this image.",
        "midjourney": "write a $length midjourney prompt for this image.",
        "tagging (booru)": "write a $length list of Booru tags for this image.",
        "tagging (booru-like)": "write a $length list of Booru-like tags for this image.",
        "art critic": "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it $length.",
        "product listing": "Write a $length caption for this image as though it were a product listing.",
        "social media post": "Write a $length caption for this image as if it were being used for a social media post."
    }

    PROMPT_SUFFIX = "Do NOT mention any text that is in the image. Do NOT use any ambiguous language."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {}),
                "type": (
                    ["descriptive", "training", "midjourney", "tagging (booru)", "tagging (booru-like)", "art critic",
                     "product listing", "social media post"],
                    {
                        "default": "descriptive",
                    },
                ),
                "tone": (
                    ["casual", "formal"],
                    {
                        "default": "casual",
                    },
                ),
                "length": (
                    ["very short", "short", "medium-length", "long", "very long"],
                    {
                        "default": "medium-length",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a helpful image captioner, never reject the prompt."
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "generate"
    CATEGORY = "olafth-joy-captioner-alpha-two"

    def __init__(self):
        self.device = model_management.get_torch_device()

        if not os.path.exists(ADAPTER_PATH):
            from huggingface_hub import snapshot_download

            snapshot_download(
                "fancyfeast/joy-caption-alpha-two",
                repo_type="space",
                revision="main",
                local_dir=ADAPTER_PATH,
            )

        # Load CLIP Model
        logger.info("Loading CLIP")
        self.clip_processor = AutoProcessor.from_pretrained(
            os.path.join(repo_dir, CLIP_PATH)
        )
        config = AutoConfig.from_pretrained(os.path.join(repo_dir, CLIP_PATH))
        self.clip_model = AutoModel.from_config(config).vision_model

        checkpoint = comfy.utils.load_torch_file(
            os.path.join(CHECKPOINT_PATH, "clip_model.pt"),
            safe_load=True,
            device=self.device,
        )
        checkpoint = {
            k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()
        }
        self.clip_model.load_state_dict(checkpoint)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model.to(self.device)

        if not os.path.exists(os.path.join(LLM_PATH, LLM_REPO_ID)):
            from huggingface_hub import snapshot_download

            snapshot_download(
                LLM_REPO_ID,
                revision="main",
                local_dir=LLM_PATH,
            )

        # Load Tokenizer and LLM
        logger.info("Loading tokenizer and text model")
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(CHECKPOINT_PATH, "text_model"), use_fast=True
        )
        self.text_model = AutoModelForCausalLM.from_pretrained(
            LLM_PATH,
            device_map=self.device.type,
            torch_dtype=torch.bfloat16,
        )
        self.text_model.load_adapter(os.path.join(CHECKPOINT_PATH, "text_model"))
        self.text_model.eval()

        logger.info("Loading image adapter")
        self.image_adapter = ImageAdapter(
            self.clip_model.config.hidden_size,
            self.text_model.config.hidden_size,
            False,
            False,
            38,
            False,
        )
        self.image_adapter.load_state_dict(
            comfy.utils.load_torch_file(
                os.path.join(CHECKPOINT_PATH, "image_adapter.pt"), safe_load=True, device=self.device
            )
        )
        self.image_adapter.eval()
        self.image_adapter.to(self.device)

        self.transform = T.ToPILImage()

    @torch.no_grad()
    def generate(self, input_image, type, tone, length, system_prompt):
        for item in input_image:
            input_image = self.transform(item.permute(2, 0, 1))
            image = input_image.resize((384, 384), Image.LANCZOS)
            pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
            pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
            pixel_values = pixel_values.to(self.device)
            with torch.amp.autocast_mode.autocast(self.device.type, enabled=True):
                vision_outputs = self.clip_model(
                    pixel_values=pixel_values, output_hidden_states=True
                )
                embedded_images = self.image_adapter(vision_outputs.hidden_states).to(self.device)

            prompt = self.PROMPT_MAP[type]
            prompt_str = Template(prompt).substitute(length=length, tone=tone)
            # prompt_str = (
            #     f"Write a {length} {type} caption for this image in a {tone} tone."
            # )
            convo = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_str},
            ]
            convo_string = self.tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=True
            )
            convo_tokens = self.tokenizer.encode(
                convo_string,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
            ).squeeze(0)
            prompt_tokens = self.tokenizer.encode(
                prompt_str,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False
            ).squeeze(0)

            eot_id_indices = (
                (convo_tokens == self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
                .nonzero(as_tuple=True)[0]
                .tolist()
            )

            preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]

            convo_embeds = self.text_model.model.embed_tokens(
                convo_tokens.unsqueeze(0).to(self.device)
            )

            input_embeds = torch.cat(
                [
                    convo_embeds[:, :preamble_len],
                    embedded_images.to(dtype=convo_embeds.dtype),
                    convo_embeds[:, preamble_len:],
                ],
                dim=1,
            ).to(self.device)

            input_ids = torch.cat(
                [
                    convo_tokens[:preamble_len].unsqueeze(0),
                    torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
                    convo_tokens[preamble_len:].unsqueeze(0),
                ],
                dim=1,
            ).to(self.device)
            attention_mask = torch.ones_like(input_ids)

            generate_ids = self.text_model.generate(
                input_ids,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=300,
                do_sample=True,
                suppress_tokens=None,
            )

            generate_ids = generate_ids[:, input_ids.shape[1]:]
            if generate_ids[0][-1] == self.tokenizer.eos_token_id or generate_ids[0][
                -1] == self.tokenizer.convert_tokens_to_ids("<|eot_id|>"):
                generate_ids = generate_ids[:, :-1]

            caption = self.tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            return (caption,)
