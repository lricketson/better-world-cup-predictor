import asyncio
import json
import os
import sys
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


async def run_deep_xray(match_url):
    print(f"[*] Launching Deep Recursive RAM X-Ray for: {match_url}")

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

        # 1. STRICT WIRE LOGGER: Only print requests originating STRICTLY from WhoScored's own servers!
        async def handle_response(response):
            try:
                # startswith() completely prevents ad-tech tracking parameters from triggering the log!
                if response.url.startswith(
                    "https://www.whoscored.com/"
                ) and response.request.resource_type in ["xhr", "fetch"]:
                    if (
                        "localization" not in response.url
                        and "empty" not in response.url
                    ):
                        print(
                            f"[CLEAN WIRE] {response.request.method} {response.url[:120]} ({response.status})"
                        )
                        try:
                            # If WhoScored sends live JSON over the wire, catch it!
                            payload = await response.json()
                            if isinstance(payload, dict) and (
                                "events" in payload or "matchCentreData" in payload
                            ):
                                print(
                                    "[!] WIRE INTERCEPT: Caught live Opta JSON payload!"
                                )
                                save_live_feed(payload)
                        except Exception:
                            pass
            except Exception:
                pass

        page.on("response", handle_response)

        try:
            print("[*] Navigating to match center...")
            await page.goto(match_url, wait_until="domcontentloaded", timeout=60000)
            print("[+] Page loaded! Waiting 4 seconds for pitch rendering...")
            await asyncio.sleep(4.0)

            last_event_count = -1
            poll_ticks = 0

            print("[*] Engaging Recursive Memory X-Ray loop...")

            while True:
                poll_ticks += 1

                # 2. THE RECURSIVE RAM HUNTER
                # Crawls up to 6 layers deep into every RequireJS module looking for Opta coordinate arrays!
                ram_data = await page.evaluate("""() => {
                    const visited = new WeakSet();
                    
                    function searchObj(obj, path, depth) {
                        if (!obj || typeof obj !== 'object' || depth > 6) return null;
                        if (visited.has(obj)) return null;
                        visited.add(obj);

                        // Check if this specific object is an array of match events
                        if (Array.isArray(obj) && obj.length > 5) {
                            const first = obj[0];
                            if (first && typeof first === 'object') {
                                // Standard Opta fingerprint check
                                if (('x' in first && 'y' in first && ('minute' in first || 'period' in first)) ||
                                    ('eventId' in first || 'outcomeType' in first || 'isTouch' in first)) {
                                    return { path: path, events_array: obj };
                                }
                            }
                        }

                        // Recursively search dictionary properties
                        if (!Array.isArray(obj)) {
                            for (let k in obj) {
                                try {
                                    if (k === 'window' || k === 'document' || k === 'parent' || k === 'top') continue;
                                    const res = searchObj(obj[k], path + '.' + k, depth + 1);
                                    if (res) return res;
                                } catch(e) {}
                            }
                        }
                        return null;
                    }

                    // Search RequireJS Registry
                    if (window.require && window.require.s && window.require.s.contexts && window.require.s.contexts._) {
                        const defs = window.require.s.contexts._.defined;
                        for (let mod in defs) {
                            try {
                                const res = searchObj(defs[mod], "require('" + mod + "')", 0);
                                if (res) {
                                    return {
                                        source: res.path,
                                        events_count: res.events_array.length,
                                        full_module_data: defs[mod]
                                    };
                                }
                            } catch(e) {}
                        }
                    }
                    return null;
                }""")

                if ram_data and isinstance(ram_data, dict):
                    source_path = ram_data.get("source", "Unknown Path")
                    current_event_count = ram_data.get("events_count", 0)
                    full_payload = ram_data.get("full_module_data", {})

                    if current_event_count != last_event_count:
                        print(
                            f"\n[$$$] X-RAY TARGET LOCKED! Found data at: [{source_path}] [$$$]"
                        )
                        print(
                            f"[+] Syncing {current_event_count} live Opta events -> live_match_feed.json"
                        )
                        save_live_feed(full_payload)
                        last_event_count = current_event_count
                    elif poll_ticks % 10 == 0:
                        print(
                            f"[*] X-Ray Locked on [{source_path}] | Active Events: {current_event_count} | Listening..."
                        )

                elif poll_ticks % 5 == 0:
                    print(
                        "[-] Scanning browser RAM... (No Opta coordinate arrays found in RequireJS yet)"
                    )

                await asyncio.sleep(1.0)

        except Exception as e:
            print(f"[-] Error: {e}")
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
        asyncio.run(run_deep_xray(url))
    except KeyboardInterrupt:
        print("\n[*] X-Ray shut down cleanly.")
