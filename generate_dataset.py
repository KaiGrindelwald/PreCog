# generate_dataset.py

import os
import random
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
import pandas as pd

# Define the list of 100 unique words (all in lower case)
words = [
    'algorithm', 'biology', 'cryptocurrency', 'dichotomy', 'encyclopedia', 'flabbergasted', 'gregarious', 'hypothesis',
    'ineffable', 'juxtaposition', 'kinematics', 'laryngitis', 'metamorphosis', 'neurology', 'ophthalmology', 'photosynthesis',
    'quadrilateral', 'rhythm', 'saccharine', 'taxonomy', 'ubiquitous', 'vocabulary', 'wavelength', 'xenophobia', 'youthful',
    'zephyr', 'abbreviation', 'benevolent', 'conundrum', 'doppelgänger', 'esoteric', 'facetious', 'hierarchy', 'idiosyncrasy',
    'juxtapositions', 'knapsack', 'lexicography', 'mnemonic', 'onomatopoeia', 'paradigm', 'quagmire', 'resilience', 'serendipity',
    'tangential', 'vicarious', 'whimsical', 'xenon', 'yacht', 'zealous', 'axiomatic', 'blasé', 'camaraderie', 'decipher', 'ephemeral',
    'fathom', 'harbinger', 'inertia', 'kaleidoscope', 'lament', 'magnanimous', 'nonchalant', 'obfuscate', 'plethora', 'quintessential',
    'rendezvous', 'sagacious', 'tenacious', 'umbrage', 'verbose', 'winsome', 'xylophone', 'yearning', 'zenith', 'ameliorate', 'bombastic',
    'candid', 'deleterious', 'epitome', 'fortuitous', 'hubris', 'iconoclast', 'jubilant', 'misanthrope', 'nuance', 'ostentatious', 'pragmatic',
    'quixotic', 'recalcitrant', 'sanguine', 'taciturn', 'vociferous', 'wane', 'xenial', 'youth', 'zealot', 'apocryphal', 'bellicose', 'candor',
    'dystopian', 'effervescent'
]


def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Adds Gaussian noise to an image.
    """
    np_img = np.array(image).astype(np.float32)
    noise = np.random.normal(mean, sigma, np_img.shape)
    noisy_img = np_img + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def add_salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5):
    """
    Adds salt and pepper noise to an image.
    """
    np_img = np.array(image)
    row, col, ch = np_img.shape
    num_salt = np.ceil(amount * row * col * salt_vs_pepper).astype(int)
    num_pepper = np.ceil(amount * row * col * (1.0 - salt_vs_pepper)).astype(int)
    
    # Add Salt (white pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in np_img.shape[:2]]
    np_img[coords[0], coords[1], :] = 255
    
    # Add Pepper (black pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in np_img.shape[:2]]
    np_img[coords[0], coords[1], :] = 0
    
    return Image.fromarray(np_img)

def add_blur(image, radius=1):
    """
    Applies Gaussian blur to an image.
    """
    return image.filter(ImageFilter.GaussianBlur(radius))

def shear_image(image, shear_factor=0.3):
    """
    Applies shear transformation to an image.
    """
    width, height = image.size
    m = shear_factor
    xshift = abs(m) * height
    new_width = width + int(round(xshift))
    image = image.transform(
        (new_width, height),
        Image.AFFINE,
        (1, m, -xshift if m > 0 else 0, 0, 1, 0),
        Image.BICUBIC,
    )
    return image

def rotate_image(image, angle=15):
    """
    Rotates the image by a given angle.
    """
    return image.rotate(angle, expand=1, fillcolor=(255, 255, 255))

def add_random_dots(image, dot_density=0.05):
    """
    Adds random dots to an image.
    """
    np_img = np.array(image)
    num_dots = int(dot_density * np_img.shape[0] * np_img.shape[1])
    for _ in range(num_dots):
        x = np.random.randint(0, np_img.shape[1])
        y = np.random.randint(0, np_img.shape[0])
        np_img[y, x] = [0, 0, 0]  # Black dot
    return Image.fromarray(np_img)

def add_random_lines(image, line_count=5):
    """
    Adds random lines to an image.
    """
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for _ in range(line_count):
        start = (np.random.randint(0, width), np.random.randint(0, height))
        end = (np.random.randint(0, width), np.random.randint(0, height))
        color = tuple(np.random.randint(0, 150, 3))
        draw.line([start, end], fill=color, width=2)
    return image

def add_combined_noise(image, sigma=25, amount=0.02, salt_vs_pepper=0.5):
    """
    Adds combined Gaussian and Salt-and-Pepper noise to an image.
    """
    noisy_img = add_gaussian_noise(image, mean=0, sigma=sigma)
    noisy_img = add_salt_pepper_noise(noisy_img, amount=amount, salt_vs_pepper=salt_vs_pepper)
    return noisy_img

def random_capitalization(word):
    """
    Randomly capitalizes individual letters in a word.
    """
    return ''.join([char.upper() if random.choice([True, False]) else char.lower() for char in word])

def add_captcha_noise(image):
    """
    Applies CAPTCHA-like noise and distortions to an image.
    """
    # Optionally, add random dots or lines (commented out)
    # image = add_random_dots(image, dot_density=0.02)
    # image = add_random_lines(image, line_count=8)
    image = shear_image(image, shear_factor=0.2)
    angle = random.uniform(-25, 25)
    image = rotate_image(image, angle=angle)
    image = add_blur(image, radius=1)
    return image

def generate_easy_set(words, output_dir, font_path, font_size=40, image_size=(400, 100), capitalization='lower', num_variations=5):
    """
    Generates the Easy Set of images with multiple variations per word.
    The labels for the Easy set are forced to be in lower case.
    """
    images_dir = os.path.join(output_dir, 'easy', 'images')
    os.makedirs(images_dir, exist_ok=True)
    labels = []

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file not found at {font_path}. Please ensure the font path is correct.")
        return

    for idx, word in enumerate(words):
        for variation in range(num_variations):
            # Create white background
            img = Image.new('RGB', image_size, color=(255, 255, 255))
            draw = ImageDraw.Draw(img)

            # Force labels to be lower case regardless of the provided parameter
            text = word.lower()

            # Calculate text size and position for centering using getbbox
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)

            # Draw text in black
            draw.text(position, text, fill=(0, 0, 0), font=font)

            # For variations > 0, apply additional noise/distortions
            if variation > 0:
                img = add_captcha_noise(img)

            filename = f'easy_{idx}_variation_{variation}.png'
            img.save(os.path.join(images_dir, filename))
            labels.append({'filename': filename, 'text': text})

    labels_csv = os.path.join(output_dir, 'easy', 'labels.csv')
    with open(labels_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)

    print(f"Easy Set generated with {len(labels)} images.")

def generate_hard_set(words, output_dir, font_paths, font_size=40, image_size=(400, 100), num_variations=5):
    """
    Generates the Hard Set of images with multiple variations per word.
    """
    images_dir = os.path.join(output_dir, 'hard', 'images')
    os.makedirs(images_dir, exist_ok=True)
    labels = []

    for idx, word in enumerate(words):
        for variation in range(num_variations):
            font_path = random.choice(font_paths)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                print(f"Font file not found at {font_path}. Skipping word: {word}")
                continue

            text = random_capitalization(word)
            img = Image.new('RGB', image_size, color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)
            text_color = tuple(np.random.randint(0, 150, 3))
            draw.text(position, text, fill=text_color, font=font)
            if variation > 0:
                img = add_captcha_noise(img)
            filename = f'hard_{idx}_variation_{variation}.png'
            img.save(os.path.join(images_dir, filename))
            labels.append({'filename': filename, 'text': word})  # label remains original word (lower case)

    labels_csv = os.path.join(output_dir, 'hard', 'labels.csv')
    with open(labels_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)

    print(f"Hard Set generated with {len(labels)} images.")

def generate_bonus_set(words, output_dir, font_paths, font_size=40, image_size=(400, 100), num_variations=5):
    """
    Generates the Bonus Set of images with multiple variations per word.
    """
    images_dir = os.path.join(output_dir, 'bonus', 'images')
    os.makedirs(images_dir, exist_ok=True)
    labels = []

    for idx, word in enumerate(words):
        for variation in range(num_variations):
            font_path = random.choice(font_paths)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                print(f"Font file not found at {font_path}. Skipping word: {word}")
                continue

            text = random_capitalization(word)
            bg_color_choice = random.choice(['green', 'red'])
            if bg_color_choice == 'green':
                bg_color = (0, 255, 0)
                render_text = text
                text_color = (0, 0, 0)
            else:
                bg_color = (255, 0, 0)
                render_text = text[::-1]
                text_color = (255, 255, 255)
            img = Image.new('RGB', image_size, color=bg_color)
            draw = ImageDraw.Draw(img)
            bbox = font.getbbox(render_text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)
            draw.text(position, render_text, fill=text_color, font=font)
            if variation > 0:
                img = add_captcha_noise(img)
            filename = f'bonus_{idx}_variation_{variation}.png'
            img.save(os.path.join(images_dir, filename))
            labels.append({'filename': filename, 'text': word})  # label remains original word (lower case)

    labels_csv = os.path.join(output_dir, 'bonus', 'labels.csv')
    with open(labels_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labels)

    print(f"Bonus Set generated with {len(labels)} images.")

def synthesize_dataset(words, output_dir, font_dir, font_size=40, image_size=(400, 100), capitalization='lower'):
    """
    Synthesizes the entire dataset by generating Easy, Hard, and Bonus sets.
    For the Easy Set, all images and labels will be in lower case.
    """
    font_files = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith('.ttf')]
    if not font_files:
        print(f"No .ttf font files found in {font_dir}. Please add font files and retry.")
        return

    print("Generating Easy Set...")
    generate_easy_set(
        words=words,
        output_dir=output_dir,
        font_path=font_files[0],
        font_size=font_size,
        image_size=image_size,
        capitalization=capitalization,  # 'lower' ensures lower case labels
        num_variations=50
    )

    print("Generating Hard Set...")
    generate_hard_set(
        words=words,
        output_dir=output_dir,
        font_paths=font_files,
        font_size=font_size,
        image_size=image_size,
        num_variations=50
    )

    print("Generating Bonus Set...")
    generate_bonus_set(
        words=words,
        output_dir=output_dir,
        font_paths=font_files,
        font_size=font_size,
        image_size=image_size,
        num_variations=50
    )

    print("Dataset synthesis complete.")

def test_noise_functions(font_path, sample_word, font_size=40, image_size=(400, 100)):
    """
    Generates a sample image with different noise types for testing.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Font file not found at {font_path}. Please ensure the font path is correct.")
        return

    img = Image.new('RGB', image_size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    text = sample_word.upper()
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)
    draw.text(position, text, fill=(0, 0, 0), font=font)
    img_gaussian = add_gaussian_noise(img, mean=0, sigma=50)
    img_salt_pepper = add_salt_pepper_noise(img, amount=0.05, salt_vs_pepper=0.5)
    img_combined = add_combined_noise(img, sigma=25, amount=0.02, salt_vs_pepper=0.5)
    img.show(title='Original')
    img_gaussian.show(title='Gaussian Noise')
    img_salt_pepper.show(title='Salt-and-Pepper Noise')
    img_combined.show(title='Combined Noise')

if __name__ == "__main__":
    # Define project directories
    project_dir = os.getcwd()
    fonts_dir = os.path.join(project_dir, 'fonts')
    dataset_dir = os.path.join(project_dir, 'dataset')

    if not os.path.isdir(fonts_dir):
        print(f"Fonts directory not found at {fonts_dir}. Please create it and add .ttf font files.")
    else:
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"Dataset directory created at {dataset_dir}.")

        # Uncomment to test noise functions on a sample word
        # test_word = 'Test'
        # test_font = os.path.join(fonts_dir, os.listdir(fonts_dir)[0])
        # test_noise_functions(font_path=test_font, sample_word=test_word, font_size=40, image_size=(400, 100))

        # IMPORTANT: Set capitalization to 'lower' so that the Easy Set labels are in lower case.
        synthesize_dataset(
            words=words,
            output_dir=dataset_dir,
            font_dir=fonts_dir,
            font_size=40,
            image_size=(400, 100),
            capitalization='lower'
        )
