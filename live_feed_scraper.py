import asyncio
import json
import os
import sys
from playwright.async_api import async_playwright
from playwright_stealth import Stealth


# Atomic save prevents main_live.py from reading a half-written file and crashing
def save_live_feed(data, filepath="./live_match_feed.json"):
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, filepath)
    except Exception as e:
        print(f"[-] Error writing atomic feed: {e}")


async def run_live_scraper(match_url):
    print(f"[*] Launching Live Opta Feed Scraper for: {match_url}")

    async with async_playwright() as p:
        # headless=False lets you watch the live WhoScored match center on your screen
        # while the script silently scrapes the data in the background!
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
        )
        await Stealth().apply_stealth_async(context)
        page = await context.new_page()

        # 1. NETWORK INTERCEPTOR: Catch live Opta data packets sent to the page in background
        async def handle_response(response):
            try:
                if "whoscored.com" in response.url and response.status == 200:
                    if any(
                        x in response.url.lower()
                        for x in ["live", "feed", "matchcentredata", "stat"]
                    ):
                        try:
                            payload = await response.json()
                            if isinstance(payload, dict) and (
                                "events" in payload or "matchCentreData" in payload
                            ):
                                target_data = payload.get("matchCentreData", payload)
                                save_live_feed(target_data)
                        except Exception:
                            pass
            except Exception:
                pass

        page.on("response", handle_response)

        try:
            print("[*] Navigating to match center...")
            response = await page.goto(
                match_url, wait_until="domcontentloaded", timeout=60000
            )

            # If Cloudflare pops up, headless=False lets you manually click the captcha!
            if response and (
                response.status == 403 or "Just a moment..." in await page.content()
            ):
                print(
                    "[!] Cloudflare challenge detected! Please solve the captcha in the open browser window."
                )
                await asyncio.sleep(20)

            print(
                "[+] Connected to Match Center! Starting real-time memory polling loop..."
            )
            last_event_count = -1

            # 2. CONTINUOUS MEMORY POLLING: Check browser RAM for updated Opta events
            while True:
                try:
                    live_data = await page.evaluate("""() => {
                        try {
                            if (typeof matchCentreData !== 'undefined') return matchCentreData;
                            if (window.matchCentreData) return window.matchCentreData;
                            return null;
                        } catch (e) {
                            return null;
                        }
                    }""")

                    if live_data and isinstance(live_data, dict):
                        events = live_data.get("events", [])
                        current_event_count = len(events)

                        # Only log to terminal when a new touch/event actually happens on the pitch
                        if current_event_count != last_event_count:
                            print(
                                f"[+] Live Opta Sync: {current_event_count} total events logged -> Updated live_match_feed.json"
                            )
                            save_live_feed(live_data)
                            last_event_count = current_event_count

                except Exception:
                    pass  # Suppress minor DOM evaluation glitches during page updates

                # Poll every 2 seconds for sub-second engine responsiveness
                await asyncio.sleep(2.0)

        except Exception as e:
            print(f"[-] Critical scraper error: {e}")
        finally:
            await context.close()
            await browser.close()


if __name__ == "__main__":
    # Supports passing the URL or Match ID directly via terminal argument, just like your historical script!
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
