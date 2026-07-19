import re

print("[*] Hunting for exact Opta event structures in debug_dump.html...")

try:
    with open("debug_dump.html", "r", encoding="utf-8") as f:
        html = f.read()
except Exception as e:
    print(f"[-] Could not read file: {e}")
    exit()

# 1. Grab specifically Block #50 (or wherever initialMatchDataForScrappers lives)
scripts = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)
for idx, s in enumerate(scripts):
    if "initialMatchDataForScrappers" in s:
        print(
            f"\n[+] FOUND IT! 'initialMatchDataForScrappers' is strictly in Block #{idx}!"
        )
        print(f"[*] Total length of this data block: {len(s)} characters")

        with open("real_match_data.txt", "w", encoding="utf-8") as out:
            out.write(s)
        print("[+] Wrote complete Block to -> ./real_match_data.txt")
        break

# 2. Search the ENTIRE HTML for Opta's unique structural keys
opta_fingerprints = ["outcomeType", "isTouch", "eventId", "satisfiedEventsTypes"]
print("\n[*] Scanning entire HTML for Opta Event fingerprints...")

for fp in opta_fingerprints:
    matches = [m.start() for m in re.finditer(re.escape(fp), html, re.IGNORECASE)]
    print(f"  -> Keyword '{fp}': Found {len(matches)} occurrences in the DOM.")

    # If we found matches, let's print the immediate 200 characters around the very first match!
    if matches:
        first_idx = matches[0]
        start = max(0, first_idx - 60)
        end = min(len(html), first_idx + 140)
        snippet = html[start:end].replace("\n", " ").strip()
        print(f"     Preview: ... {snippet} ...\n")
