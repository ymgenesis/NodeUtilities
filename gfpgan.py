import cv2
import os
import numpy as np
from PIL import Image
from pathlib import Path
import requests
from tqdm import tqdm
from gfpgan import GFPGANer
from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


@invocation(
    "gfpgan_face_restoration",
    title="GFPGAN",
    tags=["image", "gfpgan", "face", "restoration"],
    category="image",
    version="1.0.0",
)
class GfpganInvocation(BaseInvocation):
    """Face Restoration using GFPGAN."""

    def download_model(self, url, file_name):
        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        with open(file_name, "wb") as f:
            with tqdm(
                r.iter_content(chunk_size=1024), total=total_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    image: ImageField = InputField(description="Input image")
    strength: float = InputField(default=0.5, description="Restoration strength")
    upscale: int = InputField(default=1, description="Upscale multiplier")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        models_dir = context.services.configuration.models_path
        model_path = f"{models_dir}/any/face_restoration/gfpgan/GFPGANv1.4.pth"
        model_dirpath = os.path.dirname(model_path)
        model_filepath = Path(model_path)

        # Check if the directory exists, and if not, create it
        if not os.path.exists(model_dirpath):
            context.services.logger.warning(f"{model_dirpath} does not exist, creating")
            os.makedirs(model_dirpath)

        model_url = (
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        )

        if not model_filepath.is_file():
            context.services.logger.warning("GFPGAN Model Missing. Downloading. Please wait..")
            context.services.logger.warning(f"Downloading to {model_filepath}...")
            self.download_model(model_url, model_filepath)

        gfpgan = GFPGANer(
            model_path=model_path,
            upscale=self.upscale,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            models_dir=models_dir,
        )

        # Convert RGB to RGBA & store original alpha for reinsertion after codeformer
        if image.mode == "RGB":
            image = image.convert("RGBA")
            alpha_channel = np.array(image)[:, :, 3]
        elif image.mode == "RGBA":
            alpha_channel = np.array(image)[:, :, 3]
        else:
            print("The image has un unexpected colour mode:", image.mode)
            image = image.convert("RGBA")
            alpha_channel = np.array(image)[:, :, 3]

        # GFPGAN expects BGR image data
        bgrImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)

        # Execute gfpgan.enhance
        _, _, restored_img = gfpgan.enhance(
            bgrImage,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
        )

        # Upscale alpha and insert back into image
        if self.upscale > 1:
            new_width = int(alpha_channel.shape[1] * self.upscale)
            new_height = int(alpha_channel.shape[0] * self.upscale)
            alpha_channel = cv2.resize(alpha_channel, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        restored_img = np.dstack((restored_img, alpha_channel))

        # Convert back to RGB for PIL
        res = Image.fromarray(cv2.cvtColor(restored_img, cv2.COLOR_BGRA2RGBA))

        if self.strength < 1.0:
            # Resize the image to the new image if the sizes have changed
            if restored_img.size != image.size:
                image = image.resize(res.size)
            res = Image.blend(image, res, self.strength)

        image_dto = context.services.images.create(
            image=res,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
