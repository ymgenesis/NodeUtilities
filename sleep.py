## Sleep 1.2
## A node for InvokeAI, written by YMGenesis/Matthew Janik

import time

import torch
from tqdm import tqdm

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.backend.util.devices import choose_torch_device


@invocation("sleep_node", title="Sleep", tags=["sleep", "pause"], category="utility", version="1.0.2", use_cache=False)
class SleepInvocation(BaseInvocation):
    """Sleeps for a given interval in seconds. Optionally clears VRAM cache."""

    image: ImageField = InputField(description="Input image")
    interval: int = InputField(default=0, description="Time to sleep in seconds")
    clear_vram_cache: bool = InputField(default=False, description="Whether to clear the VRAM cache before sleeping")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        if self.clear_vram_cache:
            if choose_torch_device() == torch.device("cuda"):
                torch.cuda.empty_cache()
                context.services.logger.info("Sleep --> Cleared cuda VRAM cache")
            if choose_torch_device() == torch.device("mps"):
                from torch import mps

                mps.empty_cache()
                context.services.logger.info("Sleep --> Cleared mps VRAM cache")
            else:
                context.services.logger.warning("Sleep --> Device is neither cuda or mps. Not clearing VRAM cache.")
                pass

        context.services.logger.warning(f"Sleep --> Sleeping for {self.interval} second(s)")
        for _ in tqdm(range(self.interval), desc="Sleeping"):
            time.sleep(1)
        context.services.logger.info(f"Sleep --> Slept for {self.interval} second(s)")

        image_dto = context.services.images.create(
            image=image,
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
