## Insert Image Channel 1.1
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from typing import Literal

from PIL import Image
import numpy as np

from invokeai.app.invocations.baseinvocation import BaseInvocation, InputField, InvocationContext, invocation
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin


COLOR_CHANNELS = Literal[
    "Red (RGBA)",
    "Green (RGBA)",
    "Blue (RGBA)",
    "Alpha (RGBA)",
    "Cyan (CMYK)",
    "Magenta (CMYK)",
    "Yellow (CMYK)",
    "Black (CMYK)",
    "Hue (HSV)",
    "Saturation (HSV)",
    "Value (HSV)",
    "Luminosity (LAB)",
    "A (LAB)",
    "B (LAB)",
    "Y (YCbCr)",
    "Cb (YCbCr)",
    "Cr (YCbCr)",
]

CHANNEL_FORMATS = {
    "Red (RGBA)": ("RGBA", 0),
    "Green (RGBA)": ("RGBA", 1),
    "Blue (RGBA)": ("RGBA", 2),
    "Alpha (RGBA)": ("RGBA", 3),
    "Cyan (CMYK)": ("CMYK", 0),
    "Magenta (CMYK)": ("CMYK", 1),
    "Yellow (CMYK)": ("CMYK", 2),
    "Black (CMYK)": ("CMYK", 3),
    "Hue (HSV)": ("HSV", 0),
    "Saturation (HSV)": ("HSV", 1),
    "Value (HSV)": ("HSV", 2),
    "Luminosity (LAB)": ("LAB", 0),
    "A (LAB)": ("LAB", 1),
    "B (LAB)": ("LAB", 2),
    "Y (YCbCr)": ("YCbCr", 0),
    "Cb (YCbCr)": ("YCbCr", 1),
    "Cr (YCbCr)": ("YCbCr", 2),
}


@invocation("insert_image_channel", title="Insert Image Channel", tags=["image", "channel", "insert"], category="image", version="1.0.0")
class InsertImageChannelInvocation(BaseInvocation):
    """Overwrites a specified image channel with a given image."""

    image: ImageField = InputField(description="Input image to alter")
    channel_image: ImageField = InputField(description="Channel image to insert into input image")
    channel: COLOR_CHANNELS = InputField(description="Which channel to overwrite")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        channel_image = context.services.images.get_pil_image(self.channel_image.image_name)

        # extract the channel and mode from the input and reference tuple
        mode = CHANNEL_FORMATS[self.channel][0]
        channel_number = CHANNEL_FORMATS[self.channel][1]

        # Convert PIL image to new format
        image_np = np.array(image.convert(mode)).astype(int)
        channel_image_np = np.array(channel_image.convert(mode)).astype(int)

        # Overwrite channel with channel image
        image_np[:, :, channel_number] = channel_image_np[:, :, 0]

        # Convert back to RGBA format and output
        image = Image.fromarray(image_np.astype(np.uint8), mode=mode).convert("RGBA")

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
