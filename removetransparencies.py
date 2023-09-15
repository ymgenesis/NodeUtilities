## Remove Transparencies 2.0
## A node for InvokeAI, written by YMGenesis/Matthew Janik

from PIL import Image, ImageOps
from invokeai.app.models.image import (ImageCategory, ResourceOrigin)
from invokeai.app.invocations.primitives import ImageField
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InputField,
    OutputField,
    invocation,
    invocation_output
    )

@invocation_output("remove_transparencies_output")
class RemoveTransparenciesOutput(BaseInvocationOutput):
    """Base class for Remove Transparencies output"""

    image:      ImageField = OutputField(description="The output image")
    width:      int = OutputField(description="The width of the image in pixels")
    height:     int = OutputField(description="The height of the image in pixels")
    mask:       ImageField = OutputField(description="The output mask")

@invocation("remove_transparencies", title="Remove Transparencies", tags=["image", "remove", "transparencies", "crop"], category="image", version="1.0.0")
class RemoveTransparenciesInvocation(BaseInvocation):
    """Outputs an image with transparent pixels removed. Uses a transparency threshold to identify pixels for removal. Optionally crop to remaining pixels with a transparent border (px)."""

    image:                      ImageField  = InputField(description="Image to remove transparencies from")
    transparency_threshold:     float = InputField(default=0.5, description="Transparency threshold pixels meet to be removed. 0 = transparent, 1 = opaque.")
    crop:                       bool = InputField(default=False, description="Whether to crop to remaining pixels. H&W both a multiple of 8.")
    border:                     int = InputField(default=0, description="If cropping, the transparent border (px) to draw around the cropped area. >0 & multiple of 8.")
    rectangle_mask:             bool = InputField(default=False, description="Whether the mask equals the entire bounding box dimensions of the subject(s). Off when crop is on. Off will pass a form-fitting mask. On provides a larger mask for painting to work with.")
    invert_mask:                bool = InputField(default=False, description="Invert the mask")

    def invoke(self, context: InvocationContext) -> RemoveTransparenciesOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        image = image.convert("RGBA")

        # Make pixels transparent based on the transparency threshold
        pixels = image.load()
        width, height = image.size
        for x in range(width):
            for y in range(height):
                r, g, b, a = pixels[x, y]
                if a / 255.0 < self.transparency_threshold:
                    pixels[x, y] = (0, 0, 0, 0)

        bbox = image.getbbox()
        bboxcrop = image.crop(bbox)

        if self.crop:
            (width, height) = bboxcrop.size
            width8 = (width // 8) * 8
            height8 = (height // 8) * 8
            # border
            if self.border > 0:
                border = self.border
                border8 = (border // 8) * 8
                width8 += border8 * 2
                height8 += border8 * 2
            else:
                border8 = 0
            # Create a new image object for the output image
            image_out = Image.new("RGBA", (width8, height8), (0, 0, 0, 0))
            # Paste the cropped image onto the new image
            image_out.paste(bboxcrop, (border8, border8))

            # Create a mask image with subject in black and background in white
            mask = Image.new("L", (width8, height8), (0))
            mask_pixels = mask.load()
            for x in range(width8):
                for y in range(height8):
                    r, g, b, a = image_out.getpixel((x, y))
                    if a == 0:  # If pixel is transparent in the output image
                        mask_pixels[x, y] = (255)  # Set it to white in the mask

        elif not self.crop:
            # no border if cropping is false
            border = 0
            # Create a new image object for the output image
            image_out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            # Paste the cropped image onto the new image
            image_out.paste(image, (border, border))

            if self.rectangle_mask:
                # Create a mask image with a black rectangle of the dimensions of bboxcrop
                mask = Image.new("L", (image.width, image.height), (255))
                mask.paste((0), (bbox[0], bbox[1], bbox[2], bbox[3]))

            elif not self.rectangle_mask:
                # Create a mask image with subject in black and background in white
                mask = Image.new("L", (width, height), (0))
                mask_pixels = mask.load()
                for x in range(width):
                    for y in range(height):
                        r, g, b, a = image_out.getpixel((x, y))
                        if a == 0:  # If pixel is transparent in the output image
                            mask_pixels[x, y] = (255)  # Set it to white in the mask

        if self.invert_mask:
            mask = ImageOps.invert(mask)

        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            workflow=self.workflow,
        )
        mask_dto = context.services.images.create(
            image=mask,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.MASK,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return RemoveTransparenciesOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
            mask=ImageField(image_name=mask_dto.image_name)
        )
