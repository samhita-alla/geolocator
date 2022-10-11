import base64
from io import BytesIO


def read_b64_string(b64_string, return_data_type=False):
    """Read a base64-encoded string into an in-memory file-like object."""
    data_header, b64_data = split_and_validate_b64_string(b64_string)
    b64_buffer = BytesIO(base64.b64decode(b64_data))
    if return_data_type:
        return get_b64_filetype(data_header), b64_buffer
    else:
        return b64_buffer


def get_b64_filetype(data_header):
    """Retrieves the filetype information from the data type header of a base64-encoded object."""
    _, file_type = data_header.split("/")
    return file_type


def split_and_validate_b64_string(b64_string):
    """Return the data_type and data of a b64 string, with validation."""
    header, data = b64_string.split(",", 1)
    assert header.startswith("data:")
    assert header.endswith(";base64")
    data_type = header.split(";")[0].split(":")[1]
    return data_type, data
