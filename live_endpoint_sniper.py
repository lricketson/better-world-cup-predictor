import asyncio
import json
import os
import re
import sys
from playwright.async_api import async_playwright
from playwright_stealth import Stealth


def save_live_feed(data, filepath="./live_match_feed.json"):
    """
    Atomically writes clean, valid JSON to disk so LiveFeedHandler can read it without lock contention.
    """
    tmp_path = filepath + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, filepath)
    except Exception as e:
        print(f"[-] Error writing atomic feed: {e}")


def is_valid_opta_payload(payload):
    """
    Verifies that the JSON contains real Opta event structures with coordinates/qualifiers,
    rejecting ad-tech and lightweight UI clock timers.
    """
    if not payload:
        return False
    if isinstance(payload, dict):
        # Check standard WhoScored root containers
        for container_key in ["matchCentreData", "events", "matchData", "chalkboard"]:
            if container_key in payload:
                val = payload[container_key]
                if isinstance(val, dict) and "events" in val:
                    val = val["events"]
                if isinstance(val, list) and len(val) > 0:
                    first = val[0]
                    if isinstance(first, dict) and (
                        "eventId" in first
                        or "isTouch" in first
                        or ("x" in first and "y" in first)
                    ):
                        return True

        # Check if the dictionary itself is a WhoScored match bundle
        if any(
            k in payload
            for k in [
                "playerIdNameDictionary",
                "satisfiedEventsTypes",
                "periodMinuteLimits",
            ]
        ):
            return True

        # Check if payload is a single Opta event item
        if (
            ("eventId" in payload or "isTouch" in payload)
            and "x" in payload
            and "y" in payload
        ):
            return True

    elif isinstance(payload, list) and len(payload) > 0:
        first = payload[0]
        if isinstance(first, dict) and (
            "eventId" in first or "isTouch" in first or ("x" in first and "y" in first)
        ):
            return True

    return False


def extract_opta_from_html_python(html_content):
    """
    Python fallback: Uses regex lookaheads to locate Opta markers across the DOM source
    and balances braces with string syntax awareness.
    """
    markers = [
        r"matchCentreData\s*[:=]\s*(?=[\[\{])",
        r'require\.config\.params\[["\']args["\']\]\s*[:=]\s*(?=[\[\{])',
        r'("events"\s*:\s*(?=\[))',
        r'("playerIdNameDictionary"\s*:\s*(?=\{))',
        r'("satisfiedEventsTypes"\s*:\s*(?=\[))',
    ]
    for pattern in markers:
        for match in re.finditer(pattern, html_content):
            start_idx = match.end()
            if start_idx >= len(html_content):
                continue

            open_char = html_content[start_idx]
            if open_char not in ["{", "["]:
                for idx in range(start_idx, min(start_idx + 50, len(html_content))):
                    if html_content[idx] in ["{", "["]:
                        open_char = html_content[idx]
                        start_idx = idx
                        break
                else:
                    continue

            close_char = "}" if open_char == "{" else "]"
            count = 0
            in_string = False
            string_char = None
            is_escaped = False

            for i in range(start_idx, len(html_content)):
                char = html_content[i]
                if is_escaped:
                    is_escaped = False
                    continue
                if char == "\\":
                    is_escaped = True
                    continue
                if char in ['"', "'"]:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                    continue

                if not in_string:
                    if char == open_char:
                        count += 1
                    elif char == close_char:
                        count -= 1
                        if count == 0:
                            candidate_str = html_content[start_idx : i + 1]
                            try:
                                candidate_json = json.loads(candidate_str)
                                if is_valid_opta_payload(candidate_json):
                                    return candidate_json
                            except Exception:
                                pass
                            break
    return None


# 4-Layer Wildcard Hunter: Scans window objects, RequireJS modules, and <script> tags by schema
EXTRACT_MEMORY_JS = """() => {
    const visited = new WeakSet();
    
    function inspect(obj, depth = 0) {
        if (!obj || typeof obj !== 'object' || depth > 8) return null;
        if (visited.has(obj)) return null;
        visited.add(obj);

        if (Array.isArray(obj) && obj.length > 0) {
            const first = obj[0];
            if (first && typeof first === 'object' && ('eventId' in first || 'isTouch' in first || ('x' in first && 'y' in first))) {
                return { events: obj };
            }
        }

        if (!Array.isArray(obj)) {
            if ('events' in obj && Array.isArray(obj.events) && obj.events.length > 0) {
                const first = obj.events[0];
                if (first && typeof first === 'object' && ('eventId' in first || 'isTouch' in first || ('x' in first && 'y' in first))) {
                    return obj;
                }
            }
            if ('playerIdNameDictionary' in obj || 'periodMinuteLimits' in obj || 'satisfiedEventsTypes' in obj) {
                return obj;
            }
            for (let k in obj) {
                try {
                    if (['window', 'document', 'parent', 'top', 'location', 'history', 'localStorage', 'sessionStorage', 'console', 'performance'].includes(k)) continue;
                    const res = inspect(obj[k], depth + 1);
                    if (res) return res;
                } catch(e) {}
            }
        }
        return null;
    }

    // 1. Direct checks on standard WhoScored namespaces
    try {
        if (window.require && window.require.config && window.require.config.params && window.require.config.params.args) {
            const args = window.require.config.params.args;
            if (args.matchCentreData) return args.matchCentreData;
            const res = inspect(args, 0);
            if (res) return res;
        }
    } catch(e) {}

    try {
        if (window.matchCentreData) return window.matchCentreData;
        if (window.matchData) return window.matchData;
        if (window.chalkboardData) return window.chalkboardData;
    } catch(e) {}

    // 2. Search RequireJS defined module registry
    try {
        if (window.require && window.require.s && window.require.s.contexts && window.require.s.contexts._) {
            const defs = window.require.s.contexts._.defined;
            for (let mod in defs) {
                const found = inspect(defs[mod], 0);
                if (found) return found;
            }
        }
    } catch(e) {}

    // 3. Scan all <script> tags in DOM for Opta JSON strings
    try {
        const scripts = document.querySelectorAll('script');
        for (let i = 0; i < scripts.length; i++) {
            const text = scripts[i].textContent || scripts[i].innerText;
            if (text && (text.includes('eventId') || text.includes('satisfiedEventsTypes') || text.includes('playerIdNameDictionary') || text.includes('matchCentreData'))) {
                const keywords = ['matchCentreData', 'args', '"events":', 'playerIdNameDictionary', 'satisfiedEventsTypes'];
                for (let kw of keywords) {
                    let idx = text.indexOf(kw);
                    if (idx !== -1) {
                        let startBrace = text.indexOf('{', idx);
                        let startBracket = text.indexOf('[', idx);
                        let start = -1;
                        if (startBrace !== -1 && startBracket !== -1) start = Math.min(startBrace, startBracket);
                        else if (startBrace !== -1) start = startBrace;
                        else if (startBracket !== -1) start = startBracket;

                        if (start !== -1) {
                            const openChar = text[start];
                            const closeChar = openChar === '{' ? '}' : ']';
                            let count = 0;
                            let inStr = false;
                            let escape = false;
                            for (let j = start; j < text.length; j++) {
                                const c = text[j];
                                if (escape) { escape = false; continue; }
                                if (c === '\\\\') { escape = true; continue; }
                                if (c === '"' || c === "'") {
                                    if (!inStr) inStr = c;
                                    else if (inStr === c) inStr = false;
                                    continue;
                                }
                                if (!inStr) {
                                    if (c === openChar) count++;
                                    else if (c === closeChar) {
                                        count--;
                                        if (count === 0) {
                                            const jsonStr = text.substring(start, j + 1);
                                            try {
                                                const parsed = new Function("return " + jsonStr)();
                                                if (parsed) {
                                                    const valid = inspect(parsed, 0);
                                                    if (valid) return valid;
                                                }
                                            } catch(err) {}
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } catch(e) {}

    // 4. Global window scan fallback
    return inspect(window, 0);
}"""


async def trigger_ui_unpacking(page):
    """
    Simulates clicks on WhoScored match center tabs (Chalkboard, Live, Pitch)
    to force client-side lazy loading of Opta pitch coordinates.
    """
    print("[*] Engaging UI tab automation to force Opta event unpacking...")
    for tab_text in ["Chalkboard", "Live", "Pitch", "Match Centre", "Stats"]:
        try:
            elements = await page.locator(
                f"a:has-text('{tab_text}'), li:has-text('{tab_text}'), button:has-text('{tab_text}')"
            ).all()
            for el in elements:
                if await el.is_visible():
                    print(f"  -> Clicking UI navigation element: '{tab_text}'...")
                    await el.click(timeout=3000)
                    await asyncio.sleep(2.5)
                    break
        except Exception:
            continue


async def run_bulletproof_live_sniper(match_url):
    print(f"[*] Launching 4-Layer Wildcard Schema Dragnet for: {match_url}")

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

        # ---------------------------------------------------------------------
        # 1. BROAD NETWORK & WEBSOCKET DRAGNET (Inspects all WhoScored data)
        # ---------------------------------------------------------------------
        async def handle_response(response):
            try:
                if not any(
                    d in response.url.lower()
                    for d in ["whoscored.com", "opta", "perform"]
                ):
                    return

                # Check all XHR/Fetch responses from WhoScored servers
                if response.request.resource_type in ["xhr", "fetch"]:
                    try:
                        payload = await response.json()
                        if is_valid_opta_payload(payload):
                            print(
                                f"\n[$$$] OPTA NETWORK HIT! Clean JSON from: {response.url[:70]}..."
                            )
                            save_live_feed(payload)
                        return
                    except Exception:
                        pass

                    # V8 Array Sanitizer fallback for WhoScored JS literals
                    try:
                        text_payload = await response.text()
                        if any(
                            k in text_payload
                            for k in [
                                "eventId",
                                "satisfiedEventsTypes",
                                "matchCentreData",
                            ]
                        ):
                            clean_json = await page.evaluate(
                                """(rawText) => {
                                try {
                                    const evaluated = new Function("return " + rawText)();
                                    return JSON.parse(JSON.stringify(evaluated));
                                } catch (err) { return null; }
                            }""",
                                text_payload,
                            )
                            if clean_json and is_valid_opta_payload(clean_json):
                                print(
                                    f"\n[$$$] V8 SANITIZED OPTA HIT! Syncing to live_match_feed.json..."
                                )
                                save_live_feed(clean_json)
                    except Exception:
                        pass
            except Exception:
                pass

        def handle_websocket(ws):
            if not any(
                domain in ws.url.lower() for domain in ["whoscored", "opta", "perform"]
            ):
                return
            print(f"\n[~] Opta WebSocket Opened: {ws.url[:80]}...")

            def on_frame_received(payload):
                try:
                    text_data = (
                        payload if isinstance(payload, str) else payload.decode("utf-8")
                    )
                    if any(
                        k in text_data.lower()
                        for k in ["opta", "event", "typeid", "qualifiers"]
                    ):
                        try:
                            data = json.loads(text_data)
                            if is_valid_opta_payload(data):
                                print(f"\n[$$$] WEBSOCKET OPTA PULSE! Syncing...")
                                save_live_feed(data)
                        except json.JSONDecodeError:
                            for symbol in ["{", "["]:
                                idx = text_data.find(symbol)
                                if idx != -1:
                                    try:
                                        cleaned = json.loads(text_data[idx:])
                                        if is_valid_opta_payload(cleaned):
                                            print(
                                                f"\n[$$$] FRAMED WEBSOCKET OPTA PULSE! Syncing..."
                                            )
                                            save_live_feed(cleaned)
                                        break
                                    except Exception:
                                        continue
                except Exception:
                    pass

            ws.on("framereceived", on_frame_received)

        page.on("response", handle_response)
        page.on("websocket", handle_websocket)

        try:
            print("[*] Navigating to match center...")
            await page.goto(match_url, wait_until="networkidle", timeout=60000)
            print("[+] Page loaded! Waiting 5 seconds for client scripts to settle...")
            await asyncio.sleep(5.0)

            # ---------------------------------------------------------------------
            # 2. INSTANT BASELINE EXTRACTION (Chromium JS -> Python Regex -> UI Trigger)
            # ---------------------------------------------------------------------
            last_known_event_count = 0
            dom_data = await page.evaluate(EXTRACT_MEMORY_JS)

            if not dom_data:
                print(
                    "[*] JS Memory check empty. Engaging Python Regex DOM fallback..."
                )
                html_content = await page.content()
                dom_data = extract_opta_from_html_python(html_content)

            # If still empty, WhoScored is lazy-loading: trigger UI tabs
            if not dom_data:
                print(
                    "[-] Baseline still empty. Opta data is lazy-loaded! Triggering UI tabs..."
                )
                await trigger_ui_unpacking(page)
                dom_data = await page.evaluate(EXTRACT_MEMORY_JS)

            if dom_data and is_valid_opta_payload(dom_data):
                print(
                    f"[+] BASELINE SUCCESS! Extracted Opta match dictionary from browser."
                )
                payload_to_save = (
                    {"matchCentreData": dom_data} if "events" in dom_data else dom_data
                )
                save_live_feed(payload_to_save)
                events_list = (
                    dom_data.get("events", [])
                    if isinstance(dom_data, dict)
                    else dom_data.get("matchCentreData", {}).get("events", [])
                )
                last_known_event_count = len(events_list)
                print(
                    f"  -> Injected {last_known_event_count} historical Opta events into live_match_feed.json!"
                )
            else:
                print(
                    "[-] Warning: Initial extraction yielded 0 events. Relying on active memory polling..."
                )

            # ---------------------------------------------------------------------
            # 3. ACTIVE DOM MEMORY POLLING (Catches real-time background updates)
            # ---------------------------------------------------------------------
            print(
                "\n[*] Dragnet armed! Monitoring wire pulses and polling DOM memory for touch updates..."
            )
            ticks = 0
            while True:
                ticks += 1
                await asyncio.sleep(1.0)

                # Check browser RAM every 3 seconds for new pitch touches
                if ticks % 3 == 0:
                    try:
                        live_memory_data = await page.evaluate(EXTRACT_MEMORY_JS)
                        if live_memory_data and isinstance(live_memory_data, dict):
                            current_events = (
                                live_memory_data.get("events", [])
                                if "events" in live_memory_data
                                else live_memory_data.get("matchCentreData", {}).get(
                                    "events", []
                                )
                            )
                            current_count = len(current_events)

                            if current_count > last_known_event_count:
                                print(
                                    f"\n[$$$] DOM MEMORY DELTA DETECTED! Events jumped from {last_known_event_count} -> {current_count} [$$$]"
                                )
                                payload_to_save = (
                                    {"matchCentreData": live_memory_data}
                                    if "events" in live_memory_data
                                    else live_memory_data
                                )
                                save_live_feed(payload_to_save)
                                last_known_event_count = current_count
                                print(
                                    f"[+] Synced latest pitch coordinates to live_match_feed.json!"
                                )
                    except Exception:
                        pass

                if ticks % 30 == 0:
                    print(
                        f"[*] Active Monitoring | Tracking DOM Memory (Current Opta Events: {last_known_event_count})..."
                    )

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
        asyncio.run(run_bulletproof_live_sniper(url))
    except KeyboardInterrupt:
        print("\n[*] 4-Layer Wildcard Schema Dragnet shut down cleanly.")
