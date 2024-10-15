import asyncio
import os
from pathlib import Path
from PIL import Image
from tqdm.asyncio import tqdm_asyncio


async def resize_image(input_path, output_path, size=(224, 224)):
    with Image.open(input_path) as img:
        img_resized = img.resize(size, Image.LANCZOS)
        img_resized.save(output_path)


async def process_images(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    tasks = []
    for file in input_folder.glob("*"):
        if file.suffix.lower() in (".jpg", ".jpeg"):
            output_path = output_folder / f"{file.stem}.resized{file.suffix}"
            tasks.append(resize_image(file, output_path))

    await tqdm_asyncio.gather(*tasks, desc="Resizing images")


if __name__ == "__main__":
    input_folder = "next"
    output_folder = "next"

    asyncio.run(process_images(input_folder, output_folder))
    print("Image resizing completed!")
