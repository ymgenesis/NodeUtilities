# NodeUtilities

A nodes extension for use with
[InvokeAI](https://github.com/invoke-ai/InvokeAI "InvokeAI").

NodeUtilities is a collection of utility nodes written by me over time.

## Installation

To install, place the desired `.py` files into your InvokeAI invocations folder
located here:

Windows - `invokeai\.venv\Lib\site-packages\invokeai\app\invocations\`
<br>Mac/Linux - `invokeai/.venv/lib/python3.10/site-packages/invokeai/app/invocations/`

## Nodes

### Remove Background

Outputs an image with the background removed behind the subject using rembg.

Remove Background requires the `rembg` [Python package](https://pypi.org/project/rembg/).

To install `rembg`, first activate InvokeAI's virtual environment. From your 
InvokeAI root folder execute: `source .venv/bin/activate`. Then execute: 
`pip install rembg`.

Remove Background will pass through the unaltered original image if `rembg`
isn't installed in InvokeAI's virtual environment.

Note: rembg installs its onnx models to a hidden `u2net` directory in your user
home folder.

![rembg](https://github.com/ymgenesis/FaceTools/assets/25252829/9b47938a-7689-4d8a-a027-4f0f083fcca1)
![rembgresult](https://github.com/ymgenesis/FaceTools/assets/25252829/7b1a4e09-e2d0-41df-bf04-3f6797628aca)

## Remove Transparencies

Outputs an image with some pixels made transparent. Uses a transparency
threshold to identify pixels to be made transparent. Optionally crop to
remaining pixels (subject) with a transparent border (px). Mask is form-fit to
the subject in black by default. Rectangle mask draws a black bounding box
around the subject to give painting more room to work with (when altering the
subject). Invert the mask to alter the background. Rectangle mask is off when
cropping to subject's bounding box (whole mask would be black). Cropped H&W are
multiples of 8, and border is >=0 & a multiple of 8.

![removetransparencies](https://github.com/ymgenesis/NodeUtilities/assets/25252829/e3a740ba-7570-4679-80ea-8e585718a544)
<br>before:<br>
![before](https://github.com/ymgenesis/NodeUtilities/assets/25252829/a3269124-3d6f-463f-a5e9-0f1528dbd1d9)
<br>after:<br>
![result](https://github.com/ymgenesis/NodeUtilities/assets/25252829/788f4bbc-b317-4e9f-8e5c-473971347169)

### Adaptive EQ

Adaptive Histogram Equalization using skimage.

![eq](https://github.com/ymgenesis/FaceTools/assets/25252829/eb6d65eb-8f91-4981-a713-21f428860f4e)
![eqresult](https://github.com/ymgenesis/FaceTools/assets/25252829/4233d8b9-21d2-4549-b629-2aa0bf3083c6)

### Center Pad Crop

Pad or crop an image's sides from the center by specified pixels. Positive 
values are outside of the image. Checkerboard added to background
of result for demonstration. Also takes negative numbers for inward cropping.

![cpc](https://github.com/ymgenesis/FaceTools/assets/25252829/f8269881-bd91-437e-b2ee-daeb4192d7e9)
![cpcresultcheckerboard](https://github.com/ymgenesis/NodeUtilities/assets/25252829/9d60881c-cd25-4f53-9368-d471daef637c)

### Value To Value

Outputs a given value as a string, float, or integer. Rounds in the case of
float to integer. Value input types include string, float, and integer.

![v2v](https://github.com/ymgenesis/NodeUtilities/assets/25252829/e7d1e31b-ace8-4f5e-998c-995a20fd7252)

### Sleep

Sleeps for a given interval in seconds. Optionally clears VRAM cache.

![sleep](https://github.com/ymgenesis/NodeUtilities/assets/25252829/b48776d3-6171-42c1-98ae-0f0fa6d8b1d0)

### Insert Channel Image

Overwrite a channel in the input image with a channel image (channel image is 
any image).

### Face Swapper

Blend/place a face into/onto an existing face. Downloads non-commercial 
research models to `invokeai/models/any/faceswapper`. NOTE: requires 
`insightface` library. Activate invokeai's venv. From the invokeai folder:
`source .venv/bin/activate`
Next, do:
`pip install insightface`

### CodeFormer

Face restoration using CodeFormer. Downloads non-commercial
research models to `invokeai/models/any/face_restoration/`.

### GFPGAN

Face restoration using GFPGAN. Downloads models to
`invokeai/models/any/face_restoration/`. NOTE: requires provided `utils.py` to 
be placed at:
`invokeai/.venv/lib/python3.10/site-packages/gfpgan/utils.py` (mac)
or 
`invokeai/.venv/Lib/site-packages/gfpgan/utils.py` (windows)
<hr>

#### Source Image

![orig](https://github.com/ymgenesis/FaceTools/assets/25252829/eabd6361-722c-4215-ae95-ef62ac489547)
