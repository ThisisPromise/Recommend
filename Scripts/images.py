import pytesseract
from PIL import Image
import os
import io 


def create_dummy_image_with_text(file_path, text="Test OCR"):
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 150), color = (255, 255, 255))
        d = ImageDraw.Draw(img)
        try:
            font_path = "arial.ttf" # Windows
            if os.name == 'posix': # macOS, Linux
                if os.path.exists("/System/Library/Fonts/Supplemental/Arial.ttf"): # macOS
                    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
                elif os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"): # Linux
                     font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            font = ImageFont.truetype(font_path, 36)
        except IOError:
            font = ImageFont.load_default()
            print("Warning: Default font used as Arial/DejaVuSans not found. Text might not render ideally.")

        d.text((50,50), text, fill=(0,0,0), font=font)
        img.save(file_path)
        print(f"Dummy image created at: {file_path}")
        return True
    except Exception as e:
        print(f"Could not create dummy image (may need to manually provide one): {e}")
        print("Please ensure you have FreeType (Pillow dependency) and a font like Arial installed, or manually create 'sample_handwritten_query.png'.")
        return False

def extract_text_from_image(image_path):
    """
    Performs OCR on the given image file path and returns the extracted text.
    Handles Tesseract errors.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    try:
        img = Image.open(image_path)
        print(f"Processing image for OCR: {image_path}")

        text = pytesseract.image_to_string(img)
        return text
    except pytesseract.TesseractNotFoundError:
        print("\nERROR: Tesseract OCR engine not found.")
        print("Please ensure Tesseract is installed and its path is correctly set in pytesseract.pytesseract.tesseract_cmd if it's not in your system's PATH.")
        return None
    except Exception as e:
        print(f"\nAn error occurred during OCR: {e}")
        print("Ensure the image file is valid and readable.")
        return None

if __name__ == "__main__":
    dummy_image_path = 'sample_handwritten_query.png'
    test_text = "Hello OCR from direct run"

    if not os.path.exists(dummy_image_path):
        print(f"Image '{dummy_image_path}' not found. Creating a dummy one for demonstration.")
        create_dummy_image_with_text(dummy_image_path, test_text)

    extracted = extract_text_from_image(dummy_image_path)
    if extracted:
        print("\n--- Extracted Text from direct run ---")
        print(extracted)
        print("------------------------------------")
