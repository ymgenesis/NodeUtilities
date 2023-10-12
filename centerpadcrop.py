## Center Pad Crop 2.0
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from PIL import Image

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


@invocation(
    "img_pad_crop", title="Center Pad Crop", tags=["image", "center", "pad", "crop"], category="image", version="1.0.0"
)
class CenterPadCropInvocation(BaseInvocation):
    """Pad or crop an image's sides from the center by specified pixels. Positive values are outside of the image."""

    image: ImageField = InputField(description="The image to crop")
    left: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the left (negative values crop inwards, positive values pad outwards)",
    )
    right: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the right (negative values crop inwards, positive values pad outwards)",
    )
    top: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the top (negative values crop inwards, positive values pad outwards)",
    )
    bottom: int = InputField(
        default=0,
        description="Number of pixels to pad/crop from the bottom (negative values crop inwards, positive values pad outwards)",
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)

        # Calculate and create new image dimensions
        new_width = image.width + self.right + self.left
        new_height = image.height + self.top + self.bottom
        image_crop = Image.new(mode="RGBA", size=(new_width, new_height), color=(0, 0, 0, 0))

        # Paste new image onto input
        image_crop.paste(image, (self.left, self.top))

        image_dto = context.services.images.create(
            image=image_crop,
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
