import concurrent.futures
import time
import requests

# Define a function that simulates a blocking task
def fetch_url(url):
    try:
        print(f"Fetching {url}")
        response = requests.get(url)
        return f"{url}: {len(response.content)} bytes"
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"

# List of URLs to fetch
urls = [
    "https://example.com",
    # "https://httpbin.org/delay/2",
    "https://jsonplaceholder.typicode.com/posts",
    "https://jsonplaceholder.typicode.com/users"
]

# Infinite loop for continuously fetching URLs
def main():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        while True:
            # Schedule the fetch_url function for each URL
            future_to_url = {executor.submit(fetch_url, url): url for url in urls}

            # Process the results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    print(result)
                except Exception as e:
                    print(f"Error occurred while fetching {url}: {e}")

            # Delay to prevent overwhelming the server or API
            # time.sleep(5)  # Adjust delay as needed

if __name__ == "__main__":
    main()
