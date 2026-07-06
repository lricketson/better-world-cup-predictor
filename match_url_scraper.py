import asyncio
from playwright.async_api import async_playwright

async def get_match_urls():
    """
    Fetches match URLs for the World Cup from WhoScored.
    Iterates through each group stage page (Groups A-L) and the knockout stage.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()
        
        all_match_urls = set()
        output_file = "whoscored_match_urls.txt"
        
        # Selector for the "Previous Date" button based on actual HTML
        previous_button_selector = "#dayChangeBtn-prev"
        
        # 1. Fetch Group Stage Matches (Groups A-L: 23753 to 23764)
        group_stage_ids = range(23753, 23765)
        
        for stage_id in group_stage_ids:
            group_url = f"https://www.whoscored.com/regions/247/tournaments/36/seasons/10498/stages/{stage_id}/show/international-fifa-world-cup-2026"
            print(f"\nNavigating to Group URL: {group_url}")
            
            try:
                await page.goto(group_url, wait_until="domcontentloaded", timeout=60000)
                await page.wait_for_timeout(3000)
                
                group_urls = set()
                target_group_count = 6
                no_progress_count = 0
                
                while len(group_urls) < target_group_count:
                    current_count = len(group_urls)
                    
                    hrefs = await page.evaluate('''() => {
                        const links = Array.from(document.querySelectorAll('a'));
                        return links
                            .map(a => a.href)
                            .filter(href => href.includes('/matches/') && href.includes('/live/'));
                    }''')
                    
                    group_urls.update(hrefs)
                    new_count = len(group_urls)
                    print(f"  Extracted {new_count} unique match URLs so far.")
                    
                    if new_count >= target_group_count:
                        print(f"  Reached target of {target_group_count} matches for this group.")
                        break
                        
                    if new_count == current_count:
                        no_progress_count += 1
                        if no_progress_count >= 3:
                            print("  Warning: URL count not increasing. Breaking loop to prevent crawler trap.")
                            break
                    else:
                        no_progress_count = 0
                        
                    try:
                        prev_button = page.locator(previous_button_selector).first
                        if await prev_button.is_visible():
                            print("  Clicking 'Previous Date' button...")
                            await prev_button.click(force=True)
                            await page.wait_for_timeout(4000)
                        else:
                            print("  Warning: Previous button not visible. Breaking group loop.")
                            break
                    except Exception as e:
                        print(f"  Error clicking previous button: {e}")
                        break
                
                all_match_urls.update(group_urls)
                
            except Exception as e:
                print(f"An error occurred on group stage {stage_id}: {e}")

        # 2. Fetch Knockout Stage Matches (Stage ID: 23752)
        knockout_url = "https://www.whoscored.com/regions/247/tournaments/36/seasons/10498/stages/23752/show/international-fifa-world-cup-2026"
        print(f"\nNavigating to Knockout URL: {knockout_url}")
        
        try:
            await page.goto(knockout_url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)
            
            knockout_urls = set()
            target_knockout_count = 20
            no_progress_count = 0
            
            while len(knockout_urls) < target_knockout_count:
                current_count = len(knockout_urls)
                
                hrefs = await page.evaluate('''() => {
                    const links = Array.from(document.querySelectorAll('a'));
                    return links
                        .map(a => a.href)
                        .filter(href => href.includes('/matches/') && href.includes('/live/'));
                }''')
                
                knockout_urls.update(hrefs)
                new_count = len(knockout_urls)
                print(f"  Extracted {new_count} unique match URLs so far.")
                
                if new_count >= target_knockout_count:
                    print(f"  Reached target of {target_knockout_count} matches for knockouts.")
                    break
                    
                if new_count == current_count:
                    no_progress_count += 1
                    if no_progress_count >= 3:
                        print("  Warning: URL count not increasing. Breaking loop to prevent crawler trap.")
                        break
                else:
                    no_progress_count = 0
                    
                try:
                    prev_button = page.locator(previous_button_selector).first
                    if await prev_button.is_visible():
                        print("  Clicking 'Previous Date' button...")
                        await prev_button.click(force=True)
                        await page.wait_for_timeout(4000)
                    else:
                        print("  Warning: Previous button not visible. Breaking knockout loop.")
                        break
                except Exception as e:
                    print(f"  Error clicking previous button: {e}")
                    break
                    
            all_match_urls.update(knockout_urls)
            
        except Exception as e:
            print(f"An error occurred on knockout stage: {e}")

        # Save all URLs
        urls_list = list(all_match_urls)
        with open(output_file, "w", encoding="utf-8") as f:
            for url in urls_list:
                f.write(url + "\n")
                
        print(f"\nSuccessfully saved {len(urls_list)} URLs in total to {output_file}.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(get_match_urls())
