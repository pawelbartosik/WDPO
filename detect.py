import json
from pathlib import Path
from typing import Dict
import click
import cv2
from tqdm import tqdm
import numpy as np

def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """

    #TODO: Implement detection method.
    red = 0
    yellow = 0
    green = 0
    purple = 0

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype('float32')

    resize_img = cv2.resize(img, (500, 500))

    red_lower = np.array([11, 104, 255])
    red_upper = np.array([3, 255, 255])
    red_lower2 = np.array([0, 42, 0])
    red_upper2 = np.array([7, 255, 255])

    return {'red': red, 'yellow': yellow, 'green': green, 'purple': purple}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
