from PIL import Image, ImageDraw

#find strongly connected components with the mask color
def find_regions(image):
    pixel = image.load()
    neighbors = dict()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            if is_green(pixel[x,y]):
                neighbors[x, y] = {(x,y)}
    for x, y in neighbors:
        candidates = (x + 1, y), (x, y + 1)
        for candidate in candidates:
            if candidate in neighbors:
                neighbors[x, y].add(candidate)
                neighbors[candidate].add((x, y))
    closed_list = set()

    def connected_component(pixel):
        region = set()
        open_list = {pixel}
        while open_list:
            pixel = open_list.pop()
            closed_list.add(pixel)
            open_list |= neighbors[pixel] - closed_list
            region.add(pixel)
        return region

    regions = []
    for pixel in neighbors:
        if pixel not in closed_list:
            regions.append(connected_component(pixel))
    regions.sort(key = len, reverse = True)
    return regions

def expand_bounding(img, region, expand_factor=1.5, min_size=256, max_size=512):
    x, y = zip(*region)
    min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)

    width, height = img.size

    # +1 because max coordinates are inclusive.
    region_width = max_x - min_x + 1
    region_height = max_y - min_y + 1

    # Keep enough room for the ENTIRE region.
    target_size = int(max(region_width, region_height) * expand_factor)
    target_size = max(target_size, min_size)

    # IMPORTANT:
    # Never make the crop smaller than the actual censored region.
    target_size = max(
        target_size,
        region_width,
        region_height
    )

    # The old max_size=512 was cutting off long bars.
    # Allow larger crops; they are still resized to 512x512 later.
    if max_size is not None:
        target_size = max(target_size, min(region_width, max_size))
        target_size = max(target_size, min(region_height, max_size))

    x_center = (min_x + max_x) // 2
    y_center = (min_y + max_y) // 2

    half = target_size // 2

    x1 = x_center - half
    y1 = y_center - half
    x2 = x1 + target_size
    y2 = y1 + target_size

    # Shift the square back inside the image.
    if x1 < 0:
        x2 -= x1
        x1 = 0

    if y1 < 0:
        y2 -= y1
        y1 = 0

    if x2 > width:
        shift = x2 - width
        x1 -= shift
        x2 = width

    if y2 > height:
        shift = y2 - height
        y1 -= shift
        y2 = height

    # Final safety clamp.
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    # print(
    #     "CROP:",
    #     "region=", (region_width, region_height),
    #     "crop=", (x2 - x1, y2 - y1),
    #     "box=", (x1, y1, x2, y2)
    # )

    return x1, y1, x2, y2


def is_green(pixel):
    r, g, b = pixel
    return r == 0 and g == 255 and b == 0

if __name__ == '__main__':
    image = Image.open('')
    no_alpha_image = image.convert('RGB')
    draw = ImageDraw.Draw(no_alpha_image)
    for region in find_regions(no_alpha_image):
        draw.rectangle(expand_bounding(no_alpha_image, region), outline=(0, 255, 0))
    no_alpha_image.show()