## Adaptive EQ 2.0
## A node for InvokeAI, written by YMGenesis/Matthew Janik

import numpy as np
from PIL import Image
from skimage import exposure

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


@invocation("adaptive_eq", title="Adaptive EQ", tags=["image", "adaptive", "eq"], category="image", version="1.0.0")
class AdaptiveEQInvocation(BaseInvocation):
    """Adaptive Histogram Equalization using skimage."""

    image: ImageField = InputField(description="Input image")
    strength: float = InputField(default=1.5, description="Adaptive EQ strength")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        if self.strength > 0:
            strength = self.strength / 222
            nimage = np.array(image)
            img_adapteq = exposure.equalize_adapthist(nimage, clip_limit=strength)
            image = Image.fromarray((img_adapteq * 255).astype(np.uint8))

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
