import asyncio
import json
import os
import sys
import re
from playwright.async_api import async_playwright
from playwright_stealth import Stealth


def save_live_feed(data, filepath="./live_match_feed.json"):
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, filepath)
    except Exception as e:
        print(f"[-] Error writing atomic feed: {e}")


def extract_json_from_html(html_content):
    """Attempting standard extraction keys."""
    target_keys = ['"matchCentreData":', "matchCentreData:", "'matchCentreData':"]
    for key in target_keys:
        idx = html_content.find(key)
        if idx != -1:
            open_brace_idx = html_content.find("{", idx)
            if open_brace_idx != -1:
                try:
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(html_content[open_brace_idx:])
                    if isinstance(data, dict) and "events" in data:
                        return data
                except Exception:
                    continue
    return None


def run_dom_diagnostic(html_content):
    """
    Scans the raw HTML for variations of tracking variables, prints snippets
    to the terminal, and drops the raw payload to a local debug file.
    """
    print("\n[=] ================= RUNNING DOM DIAGNOSTIC ================= [=]")

    # 1. Dump full raw HTML safely to disk
    try:
        with open("debug_dump.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("[+] Successfully wrote raw page source to -> ./debug_dump.html")
    except Exception as e:
        print(f"[-] Failed to write debug file: {e}")

    # 2. Extract script tags and look for string matches
    script_tags = re.findall(r"<script[^>]*>(.*?)</script>", html_content, re.DOTALL)
    print(f"[*] Found {len(script_tags)} total <script> blocks embedded in this page.")

    potential_keywords = [
        "matchCentre",
        "matchData",
        "events",
        "opta",
        "initialData",
        "playerIdToNameMap",
    ]
    found_any_clue = False

    for idx, script in enumerate(script_tags):
        script_clean = script.strip()
        if not script_clean:
            continue

        for kw in potential_keywords:
            if kw.lower() in script_clean.lower():
                found_any_clue = True
                # Find the location of the keyword to extract a readable snippet
                kw_idx = script_clean.lower().find(kw.lower())
                start = max(0, kw_idx - 40)
                end = min(len(script_clean), kw_idx + 120)
                snippet = script_clean[start:end].replace("\n", " ").strip()
                print(f"  [!] Block #{idx} matched '{kw}': ... {snippet} ...")
                break

    if not found_any_clue:
        print(
            "[-] Warning: None of our core target tracking keywords were found anywhere inside the HTML script tags."
        )
    print("[=] ========================================================== [=]\n")


async def run_live_scraper(match_url):
    print(f"[*] Launching Live Opta Feed Scraper for: {match_url}")

    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir="./whoscored_chrome_profile",
            channel="chrome",
            headless=False,
            viewport={"width": 1920, "height": 1080},
            args=["--disable-blink-features=AutomationControlled"],
        )

        await Stealth().apply_stealth_async(context)
        page = context.pages[0] if context.pages else await context.new_page()

        # Network interceptor remains active in background
        async def handle_response(response):
            try:
                if response.status == 200 and "whoscored.com" in response.url:
                    try:
                        payload = await response.json()
                        if isinstance(payload, dict):
                            target_data = payload.get("matchCentreData", payload)
                            if "events" in target_data:
                                print(
                                    f"[!] NETWORK INTERCEPT: Caught active data packet ({len(target_data.get('events', []))} events)!"
                                )
                                save_live_feed(target_data)
                    except Exception:
                        pass
            except Exception:
                pass

        page.on("response", handle_response)

        try:
            print("[*] Navigating to match center...")
            await page.goto(match_url, wait_until="domcontentloaded", timeout=60000)

            # Give the dynamic JS an extra 3 seconds to spin up before our diagnostic check
            await asyncio.sleep(3.0)
            print("[+] Page settled! Starting monitoring loops...")

            last_event_count = -1
            poll_ticks = 0

            while True:
                poll_ticks += 1
                html_content = await page.content()

                # Trigger our diagnostic scan EXACTLY once on the first loop tick
                if poll_ticks == 1:
                    run_dom_diagnostic(html_content)

                # Standard extract attempt
                live_data = extract_json_from_html(html_content)

                if live_data and isinstance(live_data, dict):
                    events = live_data.get("events", [])
                    current_event_count = len(events)

                    if current_event_count != last_event_count:
                        print(
                            f"[+] Live Opta Sync: {current_event_count} events logged."
                        )
                        save_live_feed(live_data)
                        last_event_count = current_event_count

                elif poll_ticks % 5 == 0:
                    print(
                        f"[-] Loop tick {poll_ticks}: Script tag extraction returning None."
                    )

                await asyncio.sleep(2.0)

        except Exception as e:
            print(f"[-] Critical scraper error: {e}")
        finally:
            try:
                await context.close()
            except Exception:
                pass


if __name__ == "__main__":
    url = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "https://www.whoscored.com/matches/YOUR_MATCH_ID_HERE/live"
    )
    if url.isdigit():
        url = f"https://www.whoscored.com/matches/{url}/live"

    try:
        asyncio.run(run_live_scraper(url))
    except KeyboardInterrupt:
        print("\n[*] Live scraper shut down cleanly by operator.")
