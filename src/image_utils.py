from PIL import Image


def get_resized_dimensions(img: Image, new_size: int) -> tuple[int, int]:
    """
    Calculate new dimensions for the image to resize while preserving its aspect ratio.

    :param img: The input image to be resized
    :param new_size: The target size for the longest dimension (width or height)
    :return: A tuple containing the new width and height
    """
    width, height = img.size
    ratio = width / height
    if width > height:
        new_width = new_size
        new_height = int(new_size / ratio)
    else:
        new_height = new_size
        new_width = int(new_size * ratio)
    return new_width, new_height


def add_padding_to_image(
    img: Image,
    new_size: int,
    color: tuple[int, int, int],
    mode: str = "RGB",
) -> Image:
    """
    Adds padding to an image to make it square with a specified color and size.

    :param img: The image to which padding is added
    :param new_size: The size of the final square image
    :param color: The color of the padding
    :param mode: The mode of the new image (default is "RGB")
    :return: A new square image with padding added
    """
    width, height = img.size

    if width == height:
        return img

    new_image = Image.new(mode, (new_size, new_size), color)
    new_image.paste(img, (0, 0))
    return new_image
