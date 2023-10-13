import os
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from gfpgan.archs.gfpgan_bilinear_arch import GFPGANBilinear
from gfpgan.archs.gfpganv1_arch import GFPGANv1
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from PIL import Image
from torchvision.transforms.functional import normalize
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


@invocation(
    "gfpgan_face_restoration",
    title="GFPGAN",
    tags=["image", "gfpgan", "face", "restoration"],
    category="image",
    version="1.0.1",
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

        model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"

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


class GFPGANer:
    """Helper for restoration with GFPGAN.

    It will detect and crop faces, and then resize the faces to 512x512.
    GFPGAN is used to restored the resized faces.
    The background is upsampled with the bg_upsampler.
    Finally, the faces will be pasted back to the upsample background image.

    Args:
        model_path (str): The path to the GFPGAN model. It can be urls (will first download it automatically).
        upscale (float): The upscale of the final output. Default: 2.
        arch (str): The GFPGAN architecture. Option: clean | original. Default: clean.
        channel_multiplier (int): Channel multiplier for large networks of StyleGAN2. Default: 2.
        bg_upsampler (nn.Module): The upsampler for the background. Default: None.
    """

    def __init__(
        self, model_path, upscale=2, arch="clean", channel_multiplier=2, bg_upsampler=None, device=None, models_dir=None
    ):
        self.upscale = upscale
        self.bg_upsampler = bg_upsampler

        # initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        # initialize the GFP-GAN
        if arch == "clean":
            self.gfpgan = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "bilinear":
            self.gfpgan = GFPGANBilinear(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "original":
            self.gfpgan = GFPGANv1(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=channel_multiplier,
                decoder_load_path=None,
                fix_decoder=True,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        elif arch == "RestoreFormer":
            from gfpgan.archs.restoreformer_arch import RestoreFormer

            self.gfpgan = RestoreFormer()
        # initialize face helper
        self.face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=f"{models_dir}/any/face_restoration/weights",
        )

        if model_path.startswith("https://"):
            model_path = load_file_from_url(
                url=model_path,
                model_dir=os.path.join(models_dir, "any/face_restoration/weights"),
                progress=True,
                file_name=None,
            )
        loadnet = torch.load(model_path)
        if "params_ema" in loadnet:
            keyname = "params_ema"
        else:
            keyname = "params"
        self.gfpgan.load_state_dict(loadnet[keyname], strict=True)
        self.gfpgan.eval()
        self.gfpgan = self.gfpgan.to(self.device)

    @torch.no_grad()
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5):
        self.face_helper.clean_all()

        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            # get face landmarks for each face
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            # eye_dist_threshold=5: skip faces whose eye distance is smaller than 5 pixels
            # TODO: even with eye_dist_threshold, it will still introduce wrong detections and restorations.
            # align and warp each face
            self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.gfpgan(cropped_face_t, return_rgb=False, weight=weight)[0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f"\tFailed inference for GFPGAN: {error}.")
                restored_face = cropped_face

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = self.bg_upsampler.enhance(img, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None
