import asyncio
import json
import os
import time
import websockets
from typing import Dict, Any, List, Optional
from live_scraper import LiveEventScraper


class LiveFeedHandler:
    """
    Dual-layer ingestion bridge for Opta live event data.
    Listens to a live WebSocket feed with automated reconnection, while concurrently
    monitoring a local fallback directory for manual or scraped JSON updates.
    """

    def __init__(
        self,
        scraper: LiveEventScraper,
        websocket_url: Optional[str] = None,
        fallback_json_path: Optional[str] = "./live_match_feed.json",
    ):
        self.scraper = scraper
        self.ws_url = websocket_url
        self.fallback_path = fallback_json_path
        self.processed_event_ids = set()
        self.last_file_mtime = 0.0

    def _ingest_packet(self, packet: Dict[str, Any]) -> bool:
        """
        Validates event uniqueness and pushes touch events into the CTMC RAM ledgers.
        """
        event_id = packet.get("eventId") or packet.get("id")

        # Prevent double-counting if both WebSocket and File Poller see the same event
        if event_id and event_id in self.processed_event_ids:
            return False

        if event_id:
            self.processed_event_ids.add(event_id)

        # We only care about ball-in-play touch events (matching helpers.py logic)
        if not packet.get("isTouch", False):
            return False

        return self.scraper.process_event(packet)

    async def listen_websocket(self, auth_token: Optional[str] = None):
        """
        Primary asynchronous WebSocket listener with exponential backoff auto-reconnect.
        """
        if not self.ws_url:
            print("[!] No WebSocket URL provided. Skipping WebSocket listener.")
            return

        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        backoff = 1.0

        while True:
            try:
                print(f"[*] Connecting to Opta WebSocket: {self.ws_url}...")
                async with websockets.connect(self.ws_url, extra_headers=headers) as ws:
                    print(
                        "[+] WebSocket Connected! Listening for live World Cup events..."
                    )
                    backoff = 1.0  # Reset backoff on successful connection

                    async for message in ws:
                        payload = json.loads(message)

                        # Opta feeds can send single events or arrays of events
                        events = (
                            payload
                            if isinstance(payload, list)
                            else payload.get("events", [payload])
                        )

                        updated = False
                        for event in events:
                            if self._ingest_packet(event):
                                updated = True

                        if updated:
                            # Trigger your master pipeline here!
                            print(
                                f"[Live Feed] Clock: {self.scraper.current_clock//60:.0f}' | Score: {self.scraper.scoreboard.tolist()}"
                            )

            except (websockets.ConnectionClosed, Exception) as e:
                print(
                    f"[-] WebSocket Disconnected: {e}. Reconnecting in {backoff:.1f}s..."
                )
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    async def poll_fallback_file(self, poll_interval: float = 3.0):
        """
        Fail-Safe Layer: Continuously checks a local JSON file for modifications.
        Guarantees engine continuity even if internet sockets fail.
        """
        if not self.fallback_path:
            return

        print(f"[*] Fail-Safe File Poller active on: {self.fallback_path}")

        while True:
            try:
                if os.path.exists(self.fallback_path):
                    mtime = os.path.getmtime(self.fallback_path)

                    # If the file has been modified by an external scraper or manual drop
                    if mtime > self.last_file_mtime:
                        self.last_file_mtime = mtime
                        with open(self.fallback_path, mode="r", encoding="utf-8") as f:
                            match_data = json.load(f)

                        events = match_data.get("events", [])
                        updated = False
                        for event in events:
                            if self._ingest_packet(event):
                                updated = True

                        if updated:
                            print(
                                f"[File Fallback] Processed up to minute {self.scraper.current_clock//60:.0f}'"
                            )

            except Exception as e:
                print(f"[-] Error reading fallback file: {e}")

            await asyncio.sleep(poll_interval)

    async def run(self, auth_token: Optional[str] = None):
        """
        Master asynchronous loop running both WebSocket and File Poller concurrently.
        """
        await asyncio.gather(
            self.listen_websocket(auth_token), self.poll_fallback_file()
        )
