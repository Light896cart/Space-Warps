import matplotlib.pyplot as plt
import numpy
import requests
import warnings

from astropy.io import fits
from astropy.visualization import PercentileInterval, AsinhStretch
from astropy.table import Table
from io import BytesIO
from PIL import Image


def get_image_table(ra, dec, filters="grizy"):
    """
    Query ps1filenames.py service to get a list of images

    ra, dec = position in degrees
    filters = string with filters to include. includes all by default
    Returns a table with the results
    """
    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    # The final URL appends our query to the PS1 image service
    url = f"{service}?ra={ra}&dec={dec}&filters={filters}"
    # Read the ASCII table returned by the url
    table = Table.read(url, format='ascii')
    return table


def get_imurl(ra, dec, size=240, output_size=None, filters="grizy", im_format="jpg", color=False):
    """
    Get URL for images in the table

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include. choose from "grizy"
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """
    # Check for user input errors
    if color and (im_format == "fits"):
        raise ValueError("color images are available only for jpg or png formats")
    if im_format not in ("jpg", "png", "fits"):
        raise ValueError("format must be one of jpg, png, fits")

    # Call the original helper function to get the table
    table = get_image_table(ra, dec, filters=filters)
    url = (f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           f"ra={ra}&dec={dec}&size={size}&format={im_format}")

    # Append an output size, if requested
    if output_size:
        url = url + f"&output_size={output_size}"

    # Sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[numpy.argsort(flist)]

    if color:
        # We need at least 3 filters to create a color image
        if len(table) < 3:
            raise ValueError("at least three filters are required for an RGB color image")
        # If more than 3 filters, pick 3 filters from the availble results
        if len(table) > 3:
            table = table[[0, len(table) // 2, len(table) - 1]]
        # Create the red, green, and blue files for our image
        for i, param in enumerate(["red", "green", "blue"]):
            url = url + f"&{param}={table['filename'][i]}"

    else:
        # If not a color image, only one filter should be given.
        if len(table) > 1:
            warnings.warn('Too many filters for monochrome image. Using only 1st filter.')
        # Use red for monochrome images
        urlbase = url + "&red="
        url = []
        filename = table[0]['filename']
        url = urlbase + filename
    return url


def get_im(ra, dec, size=240, output_size=None, filters="g", im_format="jpg", color=False):
    """
    Get image at a sky position. Depends on get_imurl

    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png")
    Returns the image
    """
    # For either format, we need the image URL
    url = get_imurl(ra, dec, size=size, filters=filters, output_size=output_size, im_format=im_format, color=color)
    if im_format == "fits":
        fh = fits.open(url)
        # The image is contained within the data unit
        fits_im = fh[0].data
        # Set contrast to something reasonable
        transform = AsinhStretch() + PercentileInterval(99.5)
        im = transform(fits_im)
    else:
        # JPEG is easy. Request the file, read the bytes
        r = requests.get(url)
        im = Image.open(BytesIO(r.content))
    return im


# Crab Nebula Coordinates

ra,dec = 35.266138,-6.04728

# Set image size
size = 75
# #
# # Greyscale image
# gim = get_im(ra, dec, size=size, im_format='fits', filters="i", color=False)
# plt.imshow(gim)
# plt.show()