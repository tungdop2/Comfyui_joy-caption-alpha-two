# import sys
# import os

# repo_dir = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, repo_dir)
# original_modules = sys.modules.copy()


from .joy_captioner_alpha_two import JoyCaptioner

NODE_CLASS_MAPPINGS = {
    "JoyCaptioner": JoyCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JoyCaptioner": "Joy Captioner 𝛂2️⃣"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
