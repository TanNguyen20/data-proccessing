from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from playwright.async_api import async_playwright


async def extract_table_as_json(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url)

        # Get page content
        content = await page.content()
        await browser.close()

        # Parse HTML and find table
        soup = BeautifulSoup(content, 'html.parser')
        table = soup.find('table')
        if not table:
            return []

        # Extract header
        header_cells = table.find('tr').find_all(['th', 'td'])
        headers = [cell.get_text(strip=True) for cell in header_cells]

        # Extract data rows
        rows = table.find_all('tr')[1:]  # skip header
        json_data = []
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) == len(headers):
                row_data = {headers[i]: cells[i].get_text(strip=True) for i in range(len(headers))}
                json_data.append(row_data)

        return json_data