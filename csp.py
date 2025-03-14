import asyncio
import aiohttp
import json
import time
import random
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import backoff
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sqlite3
import threading
import colorama
from colorama import Fore, Back, Style
import logging
from tqdm import tqdm
import sys

# Initialize colorama
colorama.init(autoreset=True)

# Create custom logger class with pretty formatting
class ColoredLogger:
    def __init__(self, name):
        self.name = name
        self.setup_file_logging()
        
    def setup_file_logging(self):
        # Create file handler for logging to file
        self.file_handler = logging.FileHandler('company_problems_scraper.log')
        self.file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        self.file_handler.setFormatter(formatter)
        
        # Configure root logger for file logging
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self.file_handler)

    def info(self, message):
        logging.info(f"{self.name} - {message}")
        print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {Fore.CYAN}[{self.name}]{Style.RESET_ALL} {message}")
    
    def warning(self, message):
        logging.warning(f"{self.name} - {message}")
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} {Fore.CYAN}[{self.name}]{Style.RESET_ALL} {message}")
    
    def error(self, message):
        logging.error(f"{self.name} - {message}")
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {Fore.CYAN}[{self.name}]{Style.RESET_ALL} {message}")
    
    def success(self, message):
        logging.info(f"{self.name} - {message}")
        print(f"{Fore.GREEN}[SUCCESS]{Style.RESET_ALL} {Fore.CYAN}[{self.name}]{Style.RESET_ALL} {message}")
    
    def debug(self, message):
        logging.debug(f"{self.name} - {message}")
        if logging.getLogger().level <= logging.DEBUG:
            print(f"{Fore.BLUE}[DEBUG]{Style.RESET_ALL} {Fore.CYAN}[{self.name}]{Style.RESET_ALL} {message}")

# Create logger
logger = ColoredLogger("Scraper")

# Constants and configuration
CONFIG = {
    'TIMEOUT': 30,
    'MAX_RETRIES': 5,
    'BASE_DELAY': 1,
    'MAX_DELAY': 60,
    'REQUESTS_PER_MINUTE': 20,
    'OUTPUT_DIR': 'output',
    'DATABASE_PATH': 'company_problems.db',
}

# Create output directory if it doesn't exist
os.makedirs(CONFIG['OUTPUT_DIR'], exist_ok=True)

class ProgressBar:
    def __init__(self, total, desc="Processing"):
        self.pbar = tqdm(total=total, desc=desc, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
        
    def update(self):
        self.pbar.update(1)
    
    def close(self):
        self.pbar.close()

class DatabaseManager:
    """Manages database operations for storing problem data"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.logger = ColoredLogger("Database")
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        self.logger.info(f"Initializing database at {self.db_path}")
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS problems (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    website TEXT,
                    company TEXT,
                    title TEXT,
                    url TEXT,
                    difficulty TEXT,
                    tags TEXT,
                    premium BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(website, title, company)
                )
            ''')
            conn.commit()
        self.logger.success("Database initialized successfully")

    def save_problem(self, website: str, company: str, title: str, url: str,
                     difficulty: Optional[str] = None, tags: Optional[List[str]] = None,
                     premium: bool = False):
        """Save a problem to the database"""
        tags_str = ','.join(tags) if tags else ''

        with self.lock, sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO problems
                    (website, company, title, url, difficulty, tags, premium)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (website, company, title, url, difficulty, tags_str, premium))
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                self.logger.error(f"Failed to save problem '{title}': {str(e)}")
                return None

    def get_companies_by_website(self, website: str) -> List[str]:
        """Get all companies for a given website"""
        self.logger.info(f"Retrieving companies for {website}")
        with self.lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT company FROM problems WHERE website = ?', (website,))
            companies = [row[0] for row in cursor.fetchall()]
            self.logger.info(f"Found {len(companies)} companies for {website}")
            return companies

    def get_problems_by_company(self, website: str, company: str) -> List[Dict]:
        """Get all problems for a given website and company"""
        self.logger.info(f"Retrieving problems for {company} on {website}")
        with self.lock, sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM problems
                WHERE website = ? AND company = ?
            ''', (website, company))
            problems = [dict(row) for row in cursor.fetchall()]
            self.logger.info(f"Found {len(problems)} problems for {company} on {website}")
            return problems

class SeleniumManager:
    """Manages Selenium webdriver instances for JavaScript-heavy sites"""

    def __init__(self):
        self.driver = None
        self.logger = ColoredLogger("Selenium")

    def get_driver(self):
        """Get a configured Chrome webdriver"""
        if self.driver is None:
            self.logger.info("Initializing Chrome webdriver")
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_argument(f'user-agent={UserAgent().random}')

            try:
                self.driver = webdriver.Chrome(options=options)
                self.logger.success("Chrome webdriver initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Chrome driver: {str(e)}")
                raise

        return self.driver

    def close(self):
        """Close the webdriver"""
        if self.driver:
            self.logger.info("Closing Chrome webdriver")
            try:
                self.driver.quit()
                self.logger.success("Chrome webdriver closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing driver: {str(e)}")
            finally:
                self.driver = None

class ProxyManager:
    """Manages and rotates proxies"""

    def __init__(self):
        self.proxies = []
        self.current_index = 0
        self.lock = threading.Lock()
        self.logger = ColoredLogger("Proxy")
        self._load_proxies()

    def _load_proxies(self):
        """Load proxies from file or service"""
        self.logger.info("Loading proxy configuration")
        # In a production environment, you might want to load these from a file or service
        # For demonstration purposes, we'll use a placeholder
        try:
            # Placeholder for proxy loading logic
            # You would typically load these from a file, API or database
            self.proxies = [
                None  # No proxy by default
                # 'http://user:pass@proxy1.example.com:8080',
                # 'http://user:pass@proxy2.example.com:8080',
            ]

            if not self.proxies:
                self.logger.warning("No proxies loaded, will use direct connections")
                self.proxies = [None]
            else:
                self.logger.success(f"Loaded {len(self.proxies)} proxies")
        except Exception as e:
            self.logger.error(f"Error loading proxies: {str(e)}")
            self.proxies = [None]

    def get_proxy(self):
        """Get the next proxy in rotation"""
        with self.lock:
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)
            if proxy:
                self.logger.debug(f"Using proxy: {proxy}")
            return proxy

class CompanyProblemsScraper:
    """Main scraper class to fetch programming problems from various websites"""

    def __init__(self):
        self.problems_map = {}
        self.session = None
        self.ua = UserAgent()
        self.selenium_manager = SeleniumManager()
        self.proxy_manager = ProxyManager()
        self.db_manager = DatabaseManager(CONFIG['DATABASE_PATH'])
        self.request_timestamps = []
        self.lock = threading.Lock()
        self.logger = ColoredLogger("Scraper")
        
        # Print welcome banner
        self._print_banner()
    
    def _print_banner(self):
        """Print a welcome banner for the scraper"""
        banner = f"""
{Fore.CYAN}╔═════════════════════════════════════════════════════════╗
{Fore.CYAN}║{Style.BRIGHT}{Fore.WHITE} COMPANY PROBLEMS SCRAPER                               {Fore.CYAN}║
{Fore.CYAN}╠═════════════════════════════════════════════════════════╣
{Fore.CYAN}║{Fore.WHITE} Fetches programming problems from:                      {Fore.CYAN}║
{Fore.CYAN}║{Fore.GREEN}  • LeetCode                                             {Fore.CYAN}║
{Fore.CYAN}║{Fore.GREEN}  • HackerRank                                           {Fore.CYAN}║
{Fore.CYAN}║{Fore.GREEN}  • InterviewBit                                         {Fore.CYAN}║
{Fore.CYAN}╚═════════════════════════════════════════════════════════╝
        """
        print(banner)
        self.logger.info("Starting company problems scraper")

    async def initialize(self):
        """Initialize the HTTP session"""
        if self.session is None:
            self.logger.info("Initializing HTTP session")
            self.session = aiohttp.ClientSession()
            self.logger.success("HTTP session initialized")

    async def close(self):
        """Close resources"""
        if self.session:
            self.logger.info("Closing HTTP session")
            await self.session.close()
            self.session = None
            self.logger.success("HTTP session closed")
        
        self.logger.info("Closing Selenium manager")
        self.selenium_manager.close()

    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid getting blocked"""
        now = time.time()

        with self.lock:
            # Remove timestamps older than 60 seconds
            self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]

            # If we've made too many requests in the last minute, delay
            if len(self.request_timestamps) >= CONFIG['REQUESTS_PER_MINUTE']:
                sleep_time = 60 - (now - self.request_timestamps[0]) + 1
                self.logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                for i in range(int(sleep_time)):
                    sys.stdout.write(f"\r{Fore.YELLOW}Waiting to respect rate limits: {sleep_time-i} seconds remaining...{Style.RESET_ALL}")
                    sys.stdout.flush()
                    time.sleep(1)
                sys.stdout.write("\r" + " " * 80 + "\r")  # Clear the line
                sys.stdout.flush()
                time.sleep(sleep_time - int(sleep_time))  # Sleep the remaining fraction of seconds

            # Add current timestamp
            self.request_timestamps.append(now)

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, aiohttp.ClientResponseError, asyncio.TimeoutError),
        max_tries=CONFIG['MAX_RETRIES'],
        factor=CONFIG['BASE_DELAY'],
        max_value=CONFIG['MAX_DELAY'],
        on_backoff=lambda details: print(f"{Fore.YELLOW}Retry attempt {details['tries']}/{CONFIG['MAX_RETRIES']} after {details['wait']:.2f}s delay...{Style.RESET_ALL}")
    )
    async def fetch_url(self, url: str, headers: Optional[Dict] = None) -> str:
        """Fetch a URL with retry logic"""
        if headers is None:
            headers = {'User-Agent': self.ua.random}

        self._enforce_rate_limit()
        proxy = self.proxy_manager.get_proxy()

        try:
            self.logger.info(f"Fetching URL: {url}")
            await self.initialize()
            async with self.session.get(url, headers=headers, proxy=proxy,
                                        timeout=CONFIG['TIMEOUT']) as response:
                response.raise_for_status()
                content = await response.text()
                self.logger.success(f"Successfully fetched URL: {url}")
                return content
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            raise

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=CONFIG['MAX_RETRIES'],
        factor=CONFIG['BASE_DELAY'],
        max_value=CONFIG['MAX_DELAY'],
        on_backoff=lambda details: print(f"{Fore.YELLOW}Selenium retry attempt {details['tries']}/{CONFIG['MAX_RETRIES']} after {details['wait']:.2f}s delay...{Style.RESET_ALL}")
    )
    def fetch_with_selenium(self, url: str, wait_for_selector: Optional[str] = None) -> str:
        """Fetch a JavaScript-heavy page using Selenium"""
        self._enforce_rate_limit()

        try:
            self.logger.info(f"Fetching URL with Selenium: {url}")
            driver = self.selenium_manager.get_driver()
            driver.get(url)

            if wait_for_selector:
                self.logger.info(f"Waiting for selector: {wait_for_selector}")
                WebDriverWait(driver, CONFIG['TIMEOUT']).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                )

            content = driver.page_source
            self.logger.success(f"Successfully fetched URL with Selenium: {url}")
            return content
        except Exception as e:
            self.logger.error(f"Selenium error fetching {url}: {str(e)}")
            # Try to refresh the driver
            self.selenium_manager.close()
            raise

    async def scrape_leetcode(self):
        """Scrape LeetCode for company-specific problems"""
        self.logger.info(f"{Fore.MAGENTA}Starting LeetCode scraper{Style.RESET_ALL}")
        website = "leetcode.com"

        try:
            # Fetch the companies list from LeetCode
            companies_url = "https://leetcode.com/company/"
            self.logger.info("Fetching list of companies from LeetCode")
            html = await self.fetch_url(companies_url)
            soup = BeautifulSoup(html, 'html.parser')

            companies = []
            company_elements = soup.select('.company-tag')
            for element in company_elements:
                company_name = element.text.strip()
                company_url = urljoin("https://leetcode.com", element.get('href'))
                companies.append((company_name, company_url))

            self.logger.success(f"Found {len(companies)} companies on LeetCode")
            
            # Create progress bar for companies
            companies_progress = ProgressBar(len(companies), f"{Fore.CYAN}Processing LeetCode Companies")

            # Process each company
            for company_name, company_url in companies:
                self.logger.info(f"Processing company: {Fore.YELLOW}{company_name}{Style.RESET_ALL}")

                try:
                    # Fetch company problems page
                    html = await self.fetch_url(company_url)
                    soup = BeautifulSoup(html, 'html.parser')

                    problems = []
                    problem_elements = soup.select('table.table tbody tr')
                    
                    # Create progress bar for problems
                    problems_progress = ProgressBar(len(problem_elements), f"  Problems for {company_name}")

                    for element in problem_elements:
                        try:
                            title_element = element.select_one('td.title-cell a')
                            title = title_element.text.strip()
                            problem_url = urljoin("https://leetcode.com", title_element.get('href'))

                            difficulty = element.select_one('td.difficulty-cell').text.strip()
                            premium = 'lock' in element.select_one('td.title-cell').get_text(strip=True)

                            # Save to database
                            self.db_manager.save_problem(
                                website=website,
                                company=company_name,
                                title=title,
                                url=problem_url,
                                difficulty=difficulty,
                                premium=premium
                            )

                            problems.append({
                                'title': title,
                                'url': problem_url,
                                'difficulty': difficulty,
                                'premium': premium
                            })
                            problems_progress.update()
                        except Exception as e:
                            self.logger.error(f"Error processing problem element: {str(e)}")
                    
                    problems_progress.close()

                    # Store in problems map
                    if website not in self.problems_map:
                        self.problems_map[website] = {}
                    self.problems_map[website][company_name] = problems

                    self.logger.success(f"Processed {Fore.GREEN}{len(problems)}{Style.RESET_ALL} problems for {Fore.YELLOW}{company_name}{Style.RESET_ALL}")

                    # Add a delay to avoid rate limiting
                    await asyncio.sleep(random.uniform(1, 3))
                    companies_progress.update()

                except Exception as e:
                    self.logger.error(f"Error processing company {company_name}: {str(e)}")
                    companies_progress.update()
            
            companies_progress.close()

        except Exception as e:
            self.logger.error(f"Error in LeetCode scraper: {str(e)}")

    async def scrape_hackerrank(self):
        """Scrape HackerRank for company-specific problems"""
        self.logger.info(f"{Fore.MAGENTA}Starting HackerRank scraper{Style.RESET_ALL}")
        website = "hackerrank.com"

        try:
            # HackerRank may require authentication for company problems
            # We'll use Selenium for this
            url = "https://www.hackerrank.com/work/tests"
            self.logger.info("Fetching list of companies from HackerRank")
            html = self.fetch_with_selenium(url, wait_for_selector='.company-card')
            soup = BeautifulSoup(html, 'html.parser')

            companies = []
            company_elements = soup.select('.company-card')
            for element in company_elements:
                company_name = element.select_one('.company-title').text.strip()
                company_url = urljoin("https://www.hackerrank.com", element.get('href'))
                companies.append((company_name, company_url))

            self.logger.success(f"Found {len(companies)} companies on HackerRank")
            
            # Create progress bar for companies
            companies_progress = ProgressBar(len(companies), f"{Fore.CYAN}Processing HackerRank Companies")

            # Process each company
            for company_name, company_url in companies:
                self.logger.info(f"Processing company: {Fore.YELLOW}{company_name}{Style.RESET_ALL}")

                try:
                    # Fetch company problems
                    html = self.fetch_with_selenium(company_url, wait_for_selector='.challenge-card')
                    soup = BeautifulSoup(html, 'html.parser')

                    problems = []
                    problem_elements = soup.select('.challenge-card')
                    
                    # Create progress bar for problems
                    problems_progress = ProgressBar(len(problem_elements), f"  Problems for {company_name}")

                    for element in problem_elements:
                        try:
                            title = element.select_one('.challenge-name').text.strip()
                            problem_url = urljoin("https://www.hackerrank.com", element.get('href'))
                            difficulty = element.select_one('.difficulty').text.strip()

                            # Save to database
                            self.db_manager.save_problem(
                                website=website,
                                company=company_name,
                                title=title,
                                url=problem_url,
                                difficulty=difficulty
                            )

                            problems.append({
                                'title': title,
                                'url': problem_url,
                                'difficulty': difficulty
                            })
                            problems_progress.update()
                        except Exception as e:
                            self.logger.error(f"Error processing problem element: {str(e)}")
                    
                    problems_progress.close()

                    # Store in problems map
                    if website not in self.problems_map:
                        self.problems_map[website] = {}
                    self.problems_map[website][company_name] = problems

                    self.logger.success(f"Processed {Fore.GREEN}{len(problems)}{Style.RESET_ALL} problems for {Fore.YELLOW}{company_name}{Style.RESET_ALL}")

                    # Add a delay to avoid rate limiting
                    await asyncio.sleep(random.uniform(1, 3))
                    companies_progress.update()

                except Exception as e:
                    self.logger.error(f"Error processing company {company_name}: {str(e)}")
                    companies_progress.update()
            
            companies_progress.close()

        except Exception as e:
            self.logger.error(f"Error in HackerRank scraper: {str(e)}")

    async def scrape_interviewbit(self):
        """Scrape InterviewBit for company-specific problems"""
        self.logger.info(f"{Fore.MAGENTA}Starting InterviewBit scraper{Style.RESET_ALL}")
        website = "interviewbit.com"

        try:
            # Fetch the companies list
            companies_url = "https://www.interviewbit.com/companies/"
            self.logger.info("Fetching list of companies from InterviewBit")
            html = await self.fetch_url(companies_url)
            soup = BeautifulSoup(html, 'html.parser')

            companies = []
            company_elements = soup.select('.company-card')
            for element in company_elements:
                company_name = element.select_one('.company-name').text.strip()
                company_url = urljoin("https://www.interviewbit.com", element.get('href'))
                companies.append((company_name, company_url))

            self.logger.success(f"Found {len(companies)} companies on InterviewBit")
            
            # Create progress bar for companies
            companies_progress = ProgressBar(len(companies), f"{Fore.CYAN}Processing InterviewBit Companies")

            # Process each company
            for company_name, company_url in companies:
                self.logger.info(f"Processing company: {Fore.YELLOW}{company_name}{Style.RESET_ALL}")

                try:
                    # Fetch company problems
                    html = await self.fetch_url(company_url)
                    soup = BeautifulSoup(html, 'html.parser')

                    problems = []
                    problem_elements = soup.select('.problem-card')
                    
                    # Create progress bar for problems
                    problems_progress = ProgressBar(len(problem_elements), f"  Problems for {company_name}")

                    for element in problem_elements:
                        try:
                            title = element.select_one('.problem-title').text.strip()
                            problem_url = urljoin("https://www.interviewbit.com", element.get('href'))
                            difficulty = element.select_one('.difficulty-label').text.strip()
                            tags = [tag.text.strip() for tag in element.select('.tag')]

                            # Save to database
                            self.db_manager.save_problem(
                                website=website,
                                company=company_name,
                                title=title,
                                url=problem_url,
                                difficulty=difficulty,
                                tags=tags
                            )

                            problems.append({
                                'title': title,
                                'url': problem_url,
                                'difficulty': difficulty,
                                'tags': tags
                            })
                            problems_progress.update()
                        except Exception as e:
                            self.logger.error(f"Error processing problem element: {str(e)}")
                    
                    problems_progress.close()

                    # Store in problems map
                    if website not in self.problems_map:
                        self.problems_map[website] = {}
                    self.problems_map[website][company_name] = problems

                    self.logger.success(f"Processed {Fore.GREEN}{len(problems)}{Style.RESET_ALL} problems for {Fore.YELLOW}{company_name}{Style.RESET_ALL}")

                    # Add a delay to avoid rate limiting
                    await asyncio.sleep(random.uniform(1, 3))
                    companies_progress.update()

                except Exception as e:
                    self.logger.error(f"Error processing company {company_name}: {str(e)}")
                    companies_progress.update()
            
            companies_progress.close()

        except Exception as e:
            self.logger.error(f"Error in InterviewBit scraper: {str(e)}")

    def save_to_json(self):
        """Save the problems map to a JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(CONFIG['OUTPUT_DIR'], f'company_problems_{timestamp}.json')

        try:
            self.logger.info(f"Saving results to JSON file: {filename}")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.problems_map, f, indent=2)
            self.logger.success(f"Successfully saved results to {Fore.GREEN}{filename}{Style.RESET_ALL}")
            
            # Print a summary of the data
            total_websites = len(self.problems_map)
            total_companies = sum(len(companies) for companies in self.problems_map.values())
            total_problems = sum(
                sum(len(problems) for problems in companies.values())
                for companies in self.problems_map.values()
            )
            
            summary = f"""
{Fore.CYAN}╔═══════════════════════════════════════╗
{Fore.CYAN}║{Style.BRIGHT}{Fore.WHITE} DATA SUMMARY                       {Fore.CYAN}║
{Fore.CYAN}╠═══════════════════════════════════════╣
{Fore.CYAN}║{Fore.WHITE} Websites scraped: {Fore.GREEN}{total_websites:<16}{Fore.CYAN}║
{Fore.CYAN}║{Fore.WHITE} Companies found: {Fore.GREEN}{total_companies:<16}{Fore.CYAN}║
{Fore.CYAN}║{Fore.WHITE} Problems collected: {Fore.GREEN}{total_problems:<13}{Fore.CYAN}║
{Fore.CYAN}╚═══════════════════════════════════════╝
            """
            print(summary)
            
            return filename
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            # Try to save to a fallback location
            try:
                fallback_filename = f'company_problems_{timestamp}.json'
                self.logger.warning(f"Trying fallback location: {fallback_filename}")
                with open(fallback_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.problems_map, f, indent=2)
                self.logger.success(f"Saved results to fallback location: {Fore.GREEN}{fallback_filename}{Style.RESET_ALL}")
                return fallback_filename
            except Exception as e2:
                self.logger.error(f"Failed to save results to fallback location: {str(e2)}")
                return None

    def export_from_database(self):
        """Export all data from the database to the problems map"""
        try:
            self.logger.info("Exporting data from database to memory")
            websites = ['leetcode.com', 'hackerrank.com', 'interviewbit.com']
            
            websites_progress = ProgressBar(len(websites), f"{Fore.CYAN}Exporting websites from database")
            
            for website in websites:
                companies = self.db_manager.get_companies_by_website(website)
                
                if website not in self.problems_map:
                    self.problems_map[website] = {}
                
                companies_progress = ProgressBar(len(companies), f"  Companies from {website}")
                
                for company in companies:
                    problems = self.db_manager.get_problems_by_company(website, company)
                    
                    # Convert database format to output format
                    formatted_problems = []
                    for problem in problems:
                        formatted_problem = {
                            'title': problem['title'],
                            'url': problem['url'],
                            'difficulty': problem['difficulty']
                        }
                        
                        if problem['tags']:
                            formatted_problem['tags'] = problem['tags'].split(',')
                            
                        if problem['premium']:
                            formatted_problem['premium'] = problem['premium']
                        
                        formatted_problems.append(formatted_problem)
                    
                    self.problems_map[website][company] = formatted_problems
                    companies_progress.update()
                
                companies_progress.close()
                websites_progress.update()
            
            websites_progress.close()
            self.logger.success("Successfully exported data from database")
        except Exception as e:
            self.logger.error(f"Error exporting from database: {str(e)}")

    async def run(self):
        """Run the scraper for all websites"""
        self.logger.info(f"{Fore.MAGENTA}Starting scraping process{Style.RESET_ALL}")
        
        try:
            # Run all scrapers concurrently
            self.logger.info("Launching scrapers for all websites")
            await asyncio.gather(
                self.scrape_leetcode(),
                self.scrape_hackerrank(),
                self.scrape_interviewbit()
            )
            
            # Export all data from database to ensure we have everything
            self.logger.info("Exporting all data from database")
            self.export_from_database()
            
            # Save results
            self.logger.info("Saving final results")
            self.save_to_json()
            
            self.logger.success(f"{Fore.GREEN}{Style.BRIGHT}Scraping completed successfully!{Style.RESET_ALL}")
            
        except Exception as e:
            self.logger.error(f"Error running scrapers: {str(e)}")
        finally:
            self.logger.info("Cleaning up resources")
            await self.close()

async def main():
    """Main function to run the scraper"""
    print(f"{Fore.CYAN}{Style.BRIGHT}Company Problems Scraper{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Starting up...{Style.RESET_ALL}")
    
    scraper = CompanyProblemsScraper()
    
    try:
        await scraper.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Scraper interrupted by user{Style.RESET_ALL}")
        logger.warning("Scraper interrupted by user")
        # Try to save partial results
        logger.info("Saving partial results")
        scraper.export_from_database()
        scraper.save_to_json()
    except Exception as e:
        logger.error(f"Fatal error in main function: {str(e)}")
        # Try to save partial results
        logger.info("Saving partial results after error")
        scraper.export_from_database()
        scraper.save_to_json()
    finally:
        await scraper.close()
        print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")