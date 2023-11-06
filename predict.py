# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image
from pix2tex.cli import LatexOCR


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = LatexOCR()

    def predict(
        self,
        image_path: Path = Input(description="Input image"),
    ) -> str:
        """Run a single prediction on the model"""
        image_data = load_image(str(image_path))
        return self.model(image_data)


def load_image(image_path: str) -> Image:
    if image_path.startswith('http') or image_path.startswith('https'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')
    return image
