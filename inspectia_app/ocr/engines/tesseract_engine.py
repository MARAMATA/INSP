# -*- coding: utf-8 -*-
from __future__ import annotations
import pytesseract
from PIL import Image
from typing import Dict, Any


def ocr_image(path: str, lang: str = "eng+fra") -> Dict[str, Any]:
    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang=lang)
    return {"text": text} 