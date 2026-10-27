"""Reusable input gate for inference integration; current public route remains unconfigured."""
from io import BytesIO
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES=8*1024*1024
MAX_IMAGE_PIXELS=4096*4096
MAX_IMAGE_SIDE=8192
CLIENT_TIMEOUT_SECONDS=60
PROPOSED_WORKER_TIMEOUT_SECONDS=30


class ImageInputError(ValueError):
    def __init__(self,code,status,message):
        super().__init__(message);self.code=code;self.status=status


def decode_image(payload:bytes):
    if not payload:raise ImageInputError('EMPTY_IMAGE',422,'Image is empty')
    if len(payload)>MAX_UPLOAD_BYTES:raise ImageInputError('UPLOAD_TOO_LARGE',413,'Upload exceeds8MiB')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.format not in ('PNG','JPEG'):raise ImageInputError('UNSUPPORTED_IMAGE',415,'Use PNG or JPEG')
                w,h=image.size
                if w*h>MAX_IMAGE_PIXELS or max(w,h)>MAX_IMAGE_SIDE:
                    raise ImageInputError('IMAGE_DIMENSIONS_EXCEEDED',413,'Image dimensions exceed configured limit')
                if getattr(image,'n_frames',1)!=1:raise ImageInputError('ANIMATED_IMAGE',415,'Use a single still image')
                return ImageOps.exif_transpose(image).convert('RGB')
    except ImageInputError:raise
    except (Image.DecompressionBombWarning,Image.DecompressionBombError) as e:
        raise ImageInputError('IMAGE_DIMENSIONS_EXCEEDED',413,'Image dimensions exceed configured limit') from e
    except (UnidentifiedImageError,OSError,ValueError,SyntaxError) as e:
        raise ImageInputError('INVALID_IMAGE',422,'Image cannot be decoded') from e
