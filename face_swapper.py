import gc
from pathlib import Path

import cv2
import insightface
import numpy as np
import requests
import torch
from PIL import Image
from PIL.Image import Image as ImageType
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


@invocation("face_swapper", title="Face Swapper", tags=["image", "face", "swap"], category="image", version="1.0.0")
class FaceSwapperInvocation(BaseInvocation):
    """Replace face using InsightFace"""

    image: ImageField = InputField(description="Image that you want to replace the face of")
    face: ImageField = InputField(description="Face you want to replace with")

    def get_provider(self):
        if torch.cuda.is_available():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def make_cv2_image(self, image: ImageType):
        cv2_image = np.array(image)
        cv2_image = cv2_image[:, :, ::-1]
        cv2_image = cv2_image[:, :, :3]
        return cv2_image

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

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        models_path = context.services.configuration.models_path
        swapped_image = image
        image = self.make_cv2_image(image)

        face = context.services.images.get_pil_image(self.face.image_name)
        face = self.make_cv2_image(face)

        providers = self.get_provider()

        models_dir = f"{models_path}/any/faceswapper"

        # Initializing The Analyzer
        context.services.logger.info("Initializing Face Analyzer..")
        insightface_analyzer_path = f"{models_dir}/insightface"
        face_analyser = insightface.app.FaceAnalysis(
            name="buffalo_l", root=insightface_analyzer_path, providers=providers
        )
        face_analyser.prepare(0)

        # Initializing The Swapper
        insightface_model_path_url = (
            "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
        )
        insightface_model_path = Path(models_dir) / "inswapper_128.onnx"
        if not insightface_model_path.is_file():
            context.services.logger.warning("Model Missing. Downloading. Please wait..")
            self.download_model(insightface_model_path_url, insightface_model_path)
        context.services.logger.info("Initializing Face Swapper..")
        face_swapper = insightface.model_zoo.get_model(insightface_model_path.as_posix(), providers=providers)

        # Search For Faces
        source_face = face_analyser.get(image)
        if not source_face:
            context.services.logger.warning("No faces found in source image")
            output_image = context.services.images.create(
                image=swapped_image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
                workflow=self.workflow,
            )
            return ImageOutput(
                image=ImageField(image_name=output_image.image_name),
                width=output_image.width,
                height=output_image.height,
            )

        target_face = face_analyser.get(face)
        if not target_face:
            context.services.logger.warning("No faces found in target image")
            output_image = context.services.images.create(
                image=swapped_image,
                image_origin=ResourceOrigin.INTERNAL,
                image_category=ImageCategory.GENERAL,
                node_id=self.id,
                session_id=context.graph_execution_state_id,
                is_intermediate=self.is_intermediate,
            )
            return ImageOutput(
                image=ImageField(image_name=output_image.image_name),
                width=output_image.width,
                height=output_image.height,
            )

        context.services.logger.info("Swapping Faces...")
        swapped_image = face_swapper.get(image, source_face[0], target_face[0], paste_back=True)  # type: ignore
        swapped_image = cv2.cvtColor(swapped_image, cv2.COLOR_BGR2RGB)
        swapped_image = Image.fromarray(swapped_image)

        swapped_pil_image = context.services.images.create(
            image=swapped_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        del face_swapper
        del face_analyser
        gc.collect()

        return ImageOutput(
            image=ImageField(image_name=swapped_pil_image.image_name),
            width=swapped_pil_image.width,
            height=swapped_pil_image.height,
        )
