import pandas as pd
import re
import os
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import requests

# --- Configuration ---
SAVE_DIR = "scraped_product_images"
os.makedirs(SAVE_DIR, exist_ok=True)


WEBDRIVER_PATH = None 


BROWSER = 'chrome'

SEARCH_BASE_URL = "https://www.google.com/search?tbm=isch&q=" 


try:
    df_cnn_train = pd.read_csv('CNN_Model_Train_Data.csv')
    df_cnn_train['StockCode'] = df_cnn_train['StockCode'].astype(str)

    df_original = pd.read_csv('dataset.csv')
    df_original['StockCode'] = df_original['StockCode'].astype(str)

    merged_original_df = pd.merge(
        df_cnn_train,
        df_original[['StockCode', 'Description']],
        on='StockCode',
        how='left'
    )
    merged_original_df.drop_duplicates(subset=['StockCode', 'Description'], inplace=True)

    def clean_description(description):
        if pd.isna(description):
            return None
        description = str(description)
        description = re.sub(r'^\$', '', description)
        description = description.strip()
        return description if description and description.lower() not in ['nan', 'damages'] else None

    merged_original_df['Cleaned_Description'] = merged_original_df['Description'].apply(clean_description)
    cleaned_products_for_scraping = merged_original_df.dropna(subset=['Cleaned_Description'])

    if cleaned_products_for_scraping.empty:
        print("No valid product descriptions found after cleaning. Exiting.")
        exit()

except FileNotFoundError as e:
    print(f"Error: One of the files not found. Error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading/cleaning: {e}")
    exit()

# --- Initialize WebDriver ---
driver = None
try:
    if BROWSER == 'chrome':
        service = ChromeService(executable_path=WEBDRIVER_PATH) if WEBDRIVER_PATH else ChromeService()
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless") # Commented out for visual debugging
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=service, options=options)
    elif BROWSER == 'firefox':
        service = FirefoxService(executable_path=WEBDRIVER_PATH) if WEBDRIVER_PATH else FirefoxService()
        options = webdriver.FirefoxOptions()
        # options.add_argument("--headless") # Commented out for visual debugging
        driver = webdriver.Firefox(service=service, options=options)
    else:
        print("Unsupported browser specified.")
        exit()
    
    driver.set_page_load_timeout(30) # Set page load timeout
    print(f"WebDriver initialized ({BROWSER} headless mode).")

except Exception as e:
    print(f"Error initializing WebDriver: {e}")
    print("Please ensure your browser and WebDriver are installed and correctly configured (check WEBDRIVER_PATH if used).")
    exit()

# --- Image Downloading Function ---
def download_image(product_data, download_folder, stock_code, description_clean, img_index):
    if BROWSER == 'chrome':
        service = ChromeService(executable_path=WEBDRIVER_PATH) if WEBDRIVER_PATH else ChromeService()
        options = webdriver.ChromeOptions()
        # options.add_argument("--headless") # Keep commented out for visual debugging
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=service, options=options)
    elif BROWSER == 'firefox':
        service = FirefoxService(executable_path=WEBDRIVER_PATH) if WEBDRIVER_PATH else FirefoxService()
        options = webdriver.FirefoxOptions()
        # options.add_argument("--headless") # Keep commented out for visual debugging
        driver = webdriver.Firefox(service=service, options=options)
    else:
        raise ValueError("Unsupported browser specified. Choose 'chrome' or 'firefox'.")

    try:
        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        # Wait for the search bar to be present
        try:
            search_bar = WebDriverWait(driver, 15).until( # Increased wait time
                EC.presence_of_element_located((By.NAME, "q"))
            )
            print("Search bar found.")
        except Exception as e:
            print(f"Error: Search bar not found or loaded in time: {e}")
            return # Exit if search bar not found

        for index, row in product_data.iterrows():
            stock_code = row['StockCode']
            description = row['Description']

            if not description or pd.isna(description):
                print(f"Skipping StockCode: {stock_code} due to missing description.")
                continue

            search_query = f"{description} product image" # Refine query for better results
            print(f"\nSearching for images for StockCode: {stock_code}, Description: '{description}'")

            try:
                search_bar.clear()
                search_bar.send_keys(search_query)
                search_bar.submit() # Press Enter
                time.sleep(3) # Give time for the new search results to load
            except Exception as e:
                print(f"Error performing search for '{search_query}': {e}")
                continue

            # Scroll down to load more images (optional, but good for more results)
            last_height = driver.execute_script("return document.body.scrollHeight")
            for _ in range(3): # Scroll down a few more times
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height
            print("Scrolled down page.")

            # Find clickable image elements (thumbnails)
            # NEW SELECTOR: Target the <a> tag inside the div with class "czzyk XOEbc"
            # This is based on your manual inspection: div.czzyk.XOEbc a
            clickable_thumbnail_elements = driver.find_elements(By.CSS_SELECTOR, "div.czzyk.XOEbc a")

            if not clickable_thumbnail_elements:
                print(f"No clickable thumbnail containers found for '{description}' with selector 'div.czzyk.XOEbc a'.")
                continue

            print(f"Found {len(clickable_thumbnail_elements)} clickable thumbnail containers for '{description}'.")
            
            # --- Attempt to click the first thumbnail and download the larger image ---
            downloaded_for_product = False
            for i, clickable_element in enumerate(clickable_thumbnail_elements):
                if i >= 1: # Only try for the first few thumbnails
                    break

                try:
                    print(f"Attempting to click clickable element {i+1} for '{description}'...")
                    

                    wait = WebDriverWait(driver, 10)
                    element_to_click = wait.until(EC.element_to_be_clickable(clickable_element))
                    
                    
                    driver.execute_script("arguments[0].click();", element_to_click)
                    print("JavaScript click executed.")

                    input("PAUSED: Check the browser window. Did the large image overlay appear? Press Enter in terminal to continue...")

                    large_image_element = WebDriverWait(driver, 15).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "img.n3VNCd"))
                    )
                    
                    large_image_url = large_image_element.get_attribute('src')
                    if not large_image_url:
                        large_image_url = large_image_element.get_attribute('data-src')

                    print(f"Extracted large image URL: {large_image_url}")

                    if large_image_url and large_image_url.startswith('http'):
                        filename = f"{stock_code}_{i+1}.jpg"
                        filepath = os.path.join(download_folder, filename)

                        if download_image(large_image_url, filepath):
                            print(f"Successfully downloaded: {filename}")
                            downloaded_for_product = True
                            break # Move to next product after successful download
                        else:
                            print(f"Failed to download large image from URL: {large_image_url}")
                    else:
                        print(f"Large image URL was invalid or not found: {large_image_url}")


                except Exception as e:
                    print(f"Error processing clickable element {i+1} for '{description}': {e}")
                    # If an error occurs here, ensure the overlay is closed to continue gracefully
                    try:
                        driver.back() # Try to go back
                        time.sleep(2)
                    except:
                        pass
                finally:
                    # After attempting to download, close the overlay/go back
                    try:
                        driver.back() # Go back to search results if it was a new page
                        time.sleep(2) # Give time to close overlay/go back
                    except:
                        print("Could not close overlay or go back, continuing anyway.")
                        pass

            if not downloaded_for_product:
                print(f"No large images successfully downloaded for StockCode: {stock_code}, Description: '{description}' after trying clicks.")

    except Exception as e:
        print(f"An unexpected error occurred during scraping process: {e}")
    finally:
        if driver:
            driver.quit()
            print("Browser closed.")


# --- Main Scraping Logic ---
print("\n--- Starting Image Scraping ---")
total_downloaded_images = 0

for index, row in cleaned_products_for_scraping.iterrows():
    stock_code = row['StockCode']
    cleaned_description = row['Cleaned_Description']
    search_query = cleaned_description # Use the cleaned description as search query

    print(f"\nSearching for images for StockCode: {stock_code}, Description: '{cleaned_description}'")
    search_url = f"{SEARCH_BASE_URL}{requests.utils.quote(search_query)}" # URL-encode the query

    try:
        driver.get(search_url)
        time.sleep(2) # Give page time to load

        # --- Scroll to load more images (adjust as needed for target site) ---
        # For Google Images, you might need to scroll multiple times
        scroll_count = 0
        while scroll_count < 3: # Scroll 3 times to load more content
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2) # Wait for new content to load
            scroll_count += 1
            # You might need to click a "Show more results" button if present
            # try:
            #     show_more_button = driver.find_element(By.CSS_SELECTOR, "your_show_more_button_selector")
            #     show_more_button.click()
            #     time.sleep(2)
            # except NoSuchElementException:
            #     pass # No more button or already clicked

        # --- Extract Image URLs ---
        # This selector is for Google Images. It's HIGHLY site-specific.
        # You'll need to inspect the target website's HTML to find the correct selector.
        # Common elements might be 'img.product-thumbnail' or 'img[alt="product image"]'
        image_elements = driver.find_elements(By.CSS_SELECTOR, "img.YQ4gaf") # This is a common Google Images selector

        if not image_elements:
            print(f"No image elements found for '{cleaned_description}'. Check selector or site.")
            continue

        images_for_product = 0
        for i, img_element in enumerate(image_elements[:5]): # Try to download top 5 images per product
            try:
                # Get the image source (src or data-src)
                img_src = img_element.get_attribute('src') or img_element.get_attribute('data-src')

                if img_src and (img_src.startswith('http') or img_src.startswith('data:image')):
                    if "data:image" in img_src: # Handle base64 encoded images (often large, sometimes skip)
                        print(f"Skipping base64 image for {cleaned_description}")
                        continue
                    if download_image(img_src, SAVE_DIR, stock_code, cleaned_description, i):
                        images_for_product += 1
                        total_downloaded_images += 1
                    time.sleep(0.5) # Polite delay between image downloads
            except Exception as img_e:
                print(f"Error processing image {i} for '{cleaned_description}': {img_e}")
                continue
        print(f"Downloaded {images_for_product} images for '{cleaned_description}'.")

    except TimeoutException:
        print(f"Page load timed out for '{cleaned_description}'. Skipping.")
    except Exception as e:
        print(f"An error occurred while scraping for '{cleaned_description}': {e}")
    
    time.sleep(3) # Polite delay between product searches

print(f"\n--- Web Scraping Completed ---")
print(f"Total images downloaded: {total_downloaded_images} to '{os.path.abspath(SAVE_DIR)}'")

# --- Cleanup ---
if driver:
    driver.quit()
    print("WebDriver closed.")

print("\n**Important Considerations for Future Scraping:**")
print("1.  **Site-Specific Selectors:** The `By.CSS_SELECTOR` used (`img.Q4LuWd` for Google Images) needs to be updated if you target a different website. You'll need to use your browser's 'Inspect Element' tool.")
print("2.  **`robots.txt`:** Always check the `robots.txt` file of the website you are scraping to ensure compliance with their policies.")
print("3.  **Rate Limiting:** Implement sufficient `time.sleep()` delays to avoid being blocked by the website.")
print("4.  **Error Handling:** Add more robust error handling for network issues, changing website structures, etc.")
print("5.  **Proxies/VPN:** For large-scale scraping, consider using proxies or VPNs.")
print("6.  **Terms of Service:** Always review the terms of service of any website you intend to scrape.")