import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def upload_file_measure_performance(url, file_path):
    """Upload a file and measure the performance."""
    file_size = os.path.getsize(file_path)  # Get file size in bytes
    start_time = time.time()  # Start time
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
    end_time = time.time()  # End time
    duration = end_time - start_time  # Duration of the upload in seconds

    print(f"File: {file_path}")
    print(f"Size: {file_size} bytes")
    print(f"Upload Duration: {duration:.2f} seconds")
    print(f"HTTP Status Code: {response.status_code}")

    return {
        "file_name": os.path.basename(file_path),
        "file_size_bytes": file_size,
        "upload_duration_seconds": duration,
        "http_status_code": response.status_code
    }

def upload_folder(url, folder_path, folder_type):
    """Upload all files in a folder and save the details."""
    details = []
    total_duration = 0
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            result = upload_file_measure_performance(url, file_path)
            result["folder_type"] = folder_type  # Add the folder type to the result (text or image)
            details.append(result)
            total_duration += result["upload_duration_seconds"]

    return details, total_duration

def compare_upload_performance(url, text_folder_path, image_folder_path, output_csv_path):
    """Compare the upload performance of text and image folders and save the results in a CSV file."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_text = executor.submit(upload_folder, url, text_folder_path, "text")
        future_image = executor.submit(upload_folder, url, image_folder_path, "image")

        text_details, text_duration = future_text.result()
        image_details, image_duration = future_image.result()

    # Combine all details into a single list
    all_details = text_details + image_details

    # Create a DataFrame to save the results
    df = pd.DataFrame(all_details)
    df = df[["file_name", "folder_type", "file_size_bytes", "upload_duration_seconds", "http_status_code"]]

    # Save the DataFrame as a CSV file
    df.to_csv(output_csv_path, index=False)

    print("\n=== Upload Performance Summary ===")
    print(f"Total Upload Time for Text Folder: {text_duration:.2f} seconds")
    print(f"Total Upload Time for Image Folder: {image_duration:.2f} seconds")

    if text_duration < image_duration:
        print("Text folder upload is faster than image folder upload.")
    else:
        print("Image folder upload is faster than text folder upload.")

# Example usage
if __name__ == "__main__":
    url = 'http://127.0.0.1:5000/upload'
    text_folder_path = "D:\\vene\\exp\\output\\des"    # 替换为包含文本的文件夹路径
    image_folder_path = "D:\\vene\\exp\\output\\image"  # 替换为包含图像的文件夹路径
    output_csv_path = "D:\\vene\\exp\\upload_results.csv"  # 输出 CSV 文件的路径

    compare_upload_performance(url, text_folder_path, image_folder_path, output_csv_path)



