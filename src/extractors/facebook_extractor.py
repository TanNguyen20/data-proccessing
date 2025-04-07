from typing import Dict, List

import requests
from bs4 import BeautifulSoup

from ..config import USER_AGENT, REQUEST_TIMEOUT


class FacebookExtractor:
    """Extractor for Facebook data"""

    def __init__(self):
        self.headers = {
            "User-Agent": USER_AGENT
        }

    def extract_post_data(self, url: str) -> Dict:
        """Extract data from a Facebook post URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract post content
            content = self._extract_content(soup)

            # Extract metadata
            metadata = self._extract_metadata(soup)

            # Extract comments if available
            comments = self._extract_comments(soup)

            return {
                "url": url,
                "content": content,
                "metadata": metadata,
                "comments": comments
            }
        except Exception as e:
            raise Exception(f"Error extracting Facebook post data: {e}")

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract the main content of the post"""
        # This is a simplified implementation
        # Facebook's HTML structure may change, requiring updates to these selectors
        content_elements = soup.select('div[data-ad-preview="message"]')
        if content_elements:
            return content_elements[0].get_text(strip=True)

        # Fallback to other potential selectors
        content_elements = soup.select('div[data-ad-comet-preview="message"]')
        if content_elements:
            return content_elements[0].get_text(strip=True)

        return "Content not found"

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract metadata from the post"""
        metadata = {}

        # Extract author
        author_elements = soup.select('a[data-ad-preview="profile"]')
        if author_elements:
            metadata["author"] = author_elements[0].get_text(strip=True)

        # Extract timestamp
        timestamp_elements = soup.select('a[data-ad-preview="timestamp"]')
        if timestamp_elements:
            metadata["timestamp"] = timestamp_elements[0].get_text(strip=True)

        # Extract likes, shares, etc.
        reaction_elements = soup.select('span[data-ad-preview="reaction"]')
        if reaction_elements:
            metadata["reactions"] = reaction_elements[0].get_text(strip=True)

        return metadata

    def _extract_comments(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract comments from the post"""
        comments = []

        # This is a simplified implementation
        # Facebook's HTML structure may change, requiring updates to these selectors
        comment_elements = soup.select('div[data-ad-preview="comment"]')

        for comment_element in comment_elements:
            author_element = comment_element.select_one('a[data-ad-preview="profile"]')
            content_element = comment_element.select_one('div[data-ad-preview="message"]')

            if author_element and content_element:
                comments.append({
                    "author": author_element.get_text(strip=True),
                    "content": content_element.get_text(strip=True)
                })

        return comments

    def extract_page_data(self, page_url: str) -> Dict:
        """Extract data from a Facebook page URL"""
        try:
            response = requests.get(page_url, headers=self.headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract page name
            page_name = self._extract_page_name(soup)

            # Extract page description
            page_description = self._extract_page_description(soup)

            # Extract recent posts
            recent_posts = self._extract_recent_posts(soup)

            return {
                "url": page_url,
                "name": page_name,
                "description": page_description,
                "recent_posts": recent_posts
            }
        except Exception as e:
            raise Exception(f"Error extracting Facebook page data: {e}")

    def _extract_page_name(self, soup: BeautifulSoup) -> str:
        """Extract the name of the Facebook page"""
        # This is a simplified implementation
        name_elements = soup.select('h1[data-ad-preview="page_name"]')
        if name_elements:
            return name_elements[0].get_text(strip=True)

        return "Page name not found"

    def _extract_page_description(self, soup: BeautifulSoup) -> str:
        """Extract the description of the Facebook page"""
        # This is a simplified implementation
        description_elements = soup.select('div[data-ad-preview="page_description"]')
        if description_elements:
            return description_elements[0].get_text(strip=True)

        return "Page description not found"

    def _extract_recent_posts(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract recent posts from the Facebook page"""
        posts = []

        # This is a simplified implementation
        post_elements = soup.select('div[data-ad-preview="post"]')

        for post_element in post_elements[:5]:  # Limit to 5 recent posts
            content_element = post_element.select_one('div[data-ad-preview="message"]')
            timestamp_element = post_element.select_one('a[data-ad-preview="timestamp"]')

            if content_element:
                post_data = {
                    "content": content_element.get_text(strip=True)
                }

                if timestamp_element:
                    post_data["timestamp"] = timestamp_element.get_text(strip=True)

                posts.append(post_data)

        return posts
