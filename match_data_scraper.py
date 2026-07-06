import asyncio
import json
import os
import random
import re
import sys
from playwright.async_api import async_playwright
from playwright_stealth import Stealth


def extract_json_by_brace_counting(html_content, target_key="matchCentreData"):
    start_search = html_content.find(target_key)
    if start_search == -1:
        return None

    start_brace = html_content.find("{", start_search)
    if start_brace == -1:
        return None

    brace_count = 0
    in_string = False
    is_escaped = False

    for i in range(start_brace, len(html_content)):
        char = html_content[i]

        if is_escaped:
            is_escaped = False
            continue

        if char == "\\":
            is_escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                if brace_count == 0:
                    return html_content[start_brace : i + 1]

    return None


async def get_match_data(url, scraped_ids, storage_dir="data/world_cup_2026"):
    match_id_search = re.search(r"/matches/(\d+)", url, re.IGNORECASE)
    if not match_id_search:
        print(f"[-] Could not parse match ID from URL: {url}")
        return False  # Return False so we don't trigger a sleep delay

    match_id = match_id_search.group(1)
    if match_id in scraped_ids:
        print(f"[+] Match {match_id} already scraped. Skipping.")
        return False  # Skip instantly, no delay needed

    print(f"[*] Initializing browser for Match ID: {match_id}...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, args=["--disable-blink-features=AutomationControlled"]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )

        await Stealth().apply_stealth_async(context)
        page = await context.new_page()

        try:
            print(f"[*] Navigating to {url}")
            response = await page.goto(url, wait_until="networkidle", timeout=60000)

            if response.status == 403 or "Just a moment..." in await page.content():
                print(
                    "[-] Blocked by Cloudflare or request forbidden. Aborting to protect IP."
                )
                sys.exit(
                    "[!] Critical: Cloudflare challenge detected. Increase delays or change proxies."
                )

            html_content = await page.content()
            print("[*] Parsing page source for event data via brace counting...")

            raw_json_str = extract_json_by_brace_counting(
                html_content, "matchCentreData"
            )

            if raw_json_str:
                match_data = json.loads(raw_json_str)

                home_team = (
                    match_data.get("home", {}).get("name", "Home").replace(" ", "_")
                )
                away_team = (
                    match_data.get("away", {}).get("name", "Away").replace(" ", "_")
                )
                filename = f"match_{match_id}_{home_team}_vs_{away_team}.json"
                filepath = os.path.join(storage_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(match_data, f, indent=4, ensure_ascii=False)

                print(f"[+] Successfully saved data to {filepath}")

                with open("scraped_ids.txt", "a") as track_file:
                    track_file.write(f"{match_id}\n")

            else:
                print(f"[-] Failed to extract JSON payload for ID {match_id}.")

        except Exception as e:
            print(f"[-] Error occurred while scraping {url}: {str(e)}")

        finally:
            await context.close()
            await browser.close()

    # Return True to indicate the browser was actually fired up and we hit their servers
    return True


async def main():
    storage_dir = "data/world_cup_2026"
    os.makedirs(storage_dir, exist_ok=True)

    scraped_ids = set()
    if os.path.exists("scraped_ids.txt"):
        with open("scraped_ids.txt", "r") as f:
            scraped_ids = set(line.strip() for line in f if line.strip())

    urls_to_scrape = []

    # Check if the user passed arguments in the terminal
    cli_args = sys.argv[1:]

    if cli_args:
        print("[*] Command line arguments detected. Overriding wc_match_urls.txt...")
        for arg in cli_args:
            # If the user just types the match ID, construct the generic URL
            if arg.isdigit():
                urls_to_scrape.append(f"https://www.whoscored.com/matches/{arg}/live")
            else:
                urls_to_scrape.append(arg)
    else:
        # Fallback to wc_match_urls.txt if no terminal arguments are provided
        url_file = "wc_match_urls.txt"
        if not os.path.exists(url_file):
            with open(url_file, "w") as f:
                f.write(
                    "# Paste your WhoScored Match Center URLs below, one per line\n"
                )
            print(
                f"[-] Created empty '{url_file}'. Please paste your URLs into it and run the script again."
            )
            return

        with open(url_file, "r") as f:
            for line in f:
                cleaned_line = line.strip()
                if cleaned_line and not cleaned_line.startswith("#"):
                    urls_to_scrape.append(cleaned_line)

    if not urls_to_scrape:
        print("[-] No valid URLs found to process. Execution stopped.")
        return

    print(f"[*] Found {len(urls_to_scrape)} target URLs to process.")

    for url in urls_to_scrape:
        # We capture the boolean return state here
        was_scraped = await get_match_data(url, scraped_ids, storage_dir)

        # Only sleep if we actually loaded a page
        if was_scraped:
            delay = random.uniform(8.0, 18.0)
            print(f"[*] Sleeping for {delay:.2f} seconds to protect rate limits...")
            await asyncio.sleep(delay)


if __name__ == "__main__":
    asyncio.run(main())
