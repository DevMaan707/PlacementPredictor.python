# server.py
import os
import re
import json
import time
import random
import pandas as pd
import numpy as np
import requests
import ast
import pickle
import threading
import schedule
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("placement_predictor_server.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PlacementPredictorServer")

# Flask app setup
app = Flask(__name__)

class ProblemScraper:
    """Base class for scraping coding problems from various platforms"""

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def scrape_problems(self):
        """Method to be implemented by specific platform scrapers"""
        raise NotImplementedError("Subclasses must implement this method")

class LeetCodeScraper(ProblemScraper):
    """Scrape problems from LeetCode"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://leetcode.com"
        self.api_url = "https://leetcode.com/api/problems/all/"
        self.graphql_url = "https://leetcode.com/graphql"

    def scrape_problems(self, limit=500):
        """Fetch problems from LeetCode"""
        logger.info("Scraping problems from LeetCode...")

        try:
            # Get all problem slugs first
            response = requests.get(self.api_url, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if not data or "stat_status_pairs" not in data:
                logger.error("Failed to fetch LeetCode problems")
                return []

            problems = []
            processed = 0

            # Process problems with details
            for problem_data in data["stat_status_pairs"][:limit]:
                try:
                    stat = problem_data["stat"]
                    difficulty = problem_data["difficulty"]["level"]
                    slug = stat["question__title_slug"]

                    # Get detailed data for each problem
                    problem_details = self._get_problem_details(slug)
                    company_tags = self._get_company_tags(slug)

                    problem_info = {
                        "source": "LeetCode",
                        "id": f"LC_{stat['question_id']}",
                        "title": stat["question__title"],
                        "slug": slug,
                        "url": f"{self.base_url}/problems/{slug}/",
                        "difficulty": difficulty,
                        "acceptance_rate": round(stat["total_acs"] / stat["total_submitted"] * 100, 2) if stat["total_submitted"] > 0 else 0,
                        "description": problem_details.get("description", ""),
                        "examples": problem_details.get("examples", ""),
                        "constraints": problem_details.get("constraints", ""),
                        "companies": company_tags,
                        "topics": problem_details.get("tags", []),
                        "last_scraped": datetime.now().isoformat()
                    }

                    problems.append(problem_info)
                    processed += 1

                    # Log progress periodically
                    if processed % 20 == 0:
                        logger.info(f"Processed {processed} LeetCode problems")

                    # Be gentle with the API
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error processing LeetCode problem: {e}")
                    continue

            logger.info(f"Successfully scraped {len(problems)} LeetCode problems")
            return problems

        except Exception as e:
            logger.error(f"Error during LeetCode scraping: {e}")
            return []

    def _get_problem_details(self, slug):
        """Get detailed description and examples for a problem using GraphQL"""
        query = """
        query questionData(\$titleSlug: String!) {
          question(titleSlug: \$titleSlug) {
            content
            exampleTestcases
            topicTags {
              name
              slug
            }
            stats
            hints
          }
        }
        """

        variables = {"titleSlug": slug}

        try:
            response = requests.post(
                self.graphql_url,
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()

            if "data" in data and "question" in data["data"] and data["data"]["question"]:
                question_data = data["data"]["question"]

                # Extract content
                content = question_data.get("content", "")
                soup = BeautifulSoup(content, "html.parser")
                description = soup.get_text().strip()

                # Extract examples
                examples = question_data.get("exampleTestcases", "")

                # Extract topic tags
                tags = []
                if "topicTags" in question_data:
                    tags = [tag["name"] for tag in question_data["topicTags"]]

                return {
                    "description": description,
                    "examples": examples,
                    "tags": tags
                }

            return {
                "description": "",
                "examples": "",
                "tags": []
            }

        except Exception as e:
            logger.error(f"Error fetching problem details for {slug}: {e}")
            return {
                "description": "",
                "examples": "",
                "tags": []
            }

    def _get_company_tags(self, slug):
        """
        Try to get company tags (may require premium access)
        This uses patterns in problem discussions as a workaround
        """
        discussion_url = f"{self.base_url}/discuss/title/{slug}/"
        companies = []

        try:
            response = requests.get(discussion_url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Look for company mentions in discussion titles
            potential_companies = [
                "Google", "Amazon", "Facebook", "Microsoft", "Apple", "Twitter",
                "LinkedIn", "Uber", "Airbnb", "Netflix", "Adobe", "Bloomberg",
                "Oracle", "Salesforce", "IBM", "Intel", "Goldman Sachs", "JPMorgan",
                "Walmart", "ByteDance", "Snapchat"
            ]

            text = soup.get_text()

            for company in potential_companies:
                if company in text:
                    companies.append(company)

            return list(set(companies))

        except Exception as e:
            logger.error(f"Error getting company tags for {slug}: {e}")
            return []

class GeeksForGeeksScraper(ProblemScraper):
    """Scrape problems from GeeksForGeeks"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://practice.geeksforgeeks.org"
        self.api_url = "https://practice.geeksforgeeks.org/api/v1/problems-list/"

    def scrape_problems(self, limit=300):
        """Fetch problems from GeeksForGeeks"""
        logger.info("Scraping problems from GeeksForGeeks...")

        try:
            # Parameters for API
            params = {
                "category": "all",
                "difficulty": "all",
                "page": 1,
                "sortBy": "submissions",
                "limit": min(100, limit)  # API constraint
            }

            problems = []
            processed = 0
            page = 1

            while processed < limit:
                params["page"] = page

                response = requests.get(self.api_url, headers=self.headers, params=params)
                response.raise_for_status()
                data = response.json()

                if "results" not in data or not data["results"]:
                    break

                for problem_data in data["results"]:
                    try:
                        # Map difficulty level to numeric value similar to LeetCode
                        difficulty_map = {"school": 1, "basic": 1, "easy": 1, "medium": 2, "hard": 3}
                        difficulty = difficulty_map.get(problem_data.get("difficulty", "").lower(), 2)

                        slug = problem_data.get("slug", "")
                        problem_url = f"{self.base_url}/problem/{slug}"

                        # Get problem details
                        problem_details = self._get_problem_details(problem_url)

                        problem_info = {
                            "source": "GeeksForGeeks",
                            "id": f"GFG_{problem_data.get('id', processed)}",
                            "title": problem_data.get("title", ""),
                            "slug": slug,
                            "url": problem_url,
                            "difficulty": difficulty,
                            "acceptance_rate": float(problem_data.get("accuracy", "0").replace("%", "") or 0),
                            "description": problem_details.get("description", ""),
                            "examples": problem_details.get("examples", ""),
                            "constraints": problem_details.get("constraints", ""),
                            "companies": problem_details.get("companies", []),
                            "topics": [t.strip() for t in problem_data.get("tags", "").split(",") if t.strip()],
                            "last_scraped": datetime.now().isoformat()
                        }

                        problems.append(problem_info)
                        processed += 1

                        if processed >= limit:
                            break

                        # Log progress periodically
                        if processed % 20 == 0:
                            logger.info(f"Processed {processed} GeeksForGeeks problems")

                        # Be gentle with the API
                        time.sleep(0.5)

                    except Exception as e:
                        logger.error(f"Error processing GeeksForGeeks problem: {e}")
                        continue

                page += 1
                time.sleep(1)

            logger.info(f"Successfully scraped {len(problems)} GeeksForGeeks problems")
            return problems

        except Exception as e:
            logger.error(f"Error during GeeksForGeeks scraping: {e}")
            return []

    def _get_problem_details(self, problem_url):
        """Get detailed description and examples for a problem"""
        try:
            response = requests.get(problem_url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Get problem description
            problem_div = soup.select_one(".problem-statement")
            description = ""
            examples = ""
            constraints = ""
            companies = []

            if problem_div:
                # Extract description
                paragraphs = problem_div.select("p")
                if paragraphs:
                    description = "\n".join(p.get_text() for p in paragraphs)

                # Extract examples
                example_divs = problem_div.select(".example")
                if example_divs:
                    examples = "\n".join(div.get_text() for div in example_divs)

                # Extract constraints
                constraint_divs = soup.select(".problemNote")
                if constraint_divs:
                    constraints = "\n".join(div.get_text() for div in constraint_divs)

                # Try to find company information
                company_section = soup.select_one(".problems-asked-by-div")
                if company_section:
                    company_text = company_section.get_text()
                    potential_companies = [
                        "Google", "Amazon", "Facebook", "Microsoft", "Apple", "Twitter",
                        "LinkedIn", "Uber", "Airbnb", "Netflix", "Adobe", "Bloomberg",
                        "Oracle", "Salesforce", "IBM", "Intel", "Goldman Sachs", "JPMorgan",
                        "Walmart", "ByteDance", "Snapchat"
                    ]

                    for company in potential_companies:
                        if company in company_text:
                            companies.append(company)

            return {
                "description": description,
                "examples": examples,
                "constraints": constraints,
                "companies": companies
            }

        except Exception as e:
            logger.error(f"Error fetching GeeksForGeeks problem details: {e}")
            return {
                "description": "",
                "examples": "",
                "constraints": "",
                "companies": []
            }

class InterviewBitScraper(ProblemScraper):
    """Scrape problems from InterviewBit"""

    def __init__(self):
        super().__init__()
        self.base_url = "https://www.interviewbit.com"
        self.problems_url = "https://www.interviewbit.com/problems/"

    def scrape_problems(self, limit=200):
        """Fetch problems from InterviewBit"""
        logger.info("Scraping problems from InterviewBit...")

        try:
            # Get the main problems page to extract problem links
            response = requests.get(self.problems_url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Find links to all problem categories
            category_links = []
            for a in soup.select("a.topic-link"):
                href = a.get("href")
                if href and "/problems/topics/" in href:
                    category_links.append(self.base_url + href)

            # Process each category
            problems = []
            processed = 0

            with ThreadPoolExecutor(max_workers=3) as executor:
                category_results = list(executor.map(self._scrape_category, category_links))

            # Flatten results and take up to the limit
            for category_problems in category_results:
                problems.extend(category_problems)
                if len(problems) >= limit:
                    problems = problems[:limit]
                    break

            logger.info(f"Successfully scraped {len(problems)} InterviewBit problems")
            return problems

        except Exception as e:
            logger.error(f"Error during InterviewBit scraping: {e}")
            return []

    def _scrape_category(self, category_url):
        """Scrape problems from a category page"""
        category_problems = []
        try:
            response = requests.get(category_url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract category name
            category = category_url.split("/")[-1].replace("-", " ").title()

            # Find all problem links in this category
            problem_links = []
            for a in soup.select("a.problem-link"):
                href = a.get("href")
                if href and "/problems/" in href:
                    problem_links.append(self.base_url + href)

            # Process each problem
            for problem_url in problem_links[:25]:  # Limit per category
                try:
                    problem_details = self._get_problem_details(problem_url)
                    if not problem_details:
                        continue

                    # Extract slug from URL
                    slug = problem_url.split("/")[-1]

                    # Determine difficulty (InterviewBit doesn't always show it clearly)
                    difficulty_text = problem_details.get("difficulty_text", "").lower()
                    if "hard" in difficulty_text:
                        difficulty = 3
                    elif "easy" in difficulty_text:
                        difficulty = 1
                    else:
                        difficulty = 2  # Default to medium

                    problem_info = {
                        "source": "InterviewBit",
                        "id": f"IB_{len(category_problems)}_{slug}",
                        "title": problem_details.get("title", ""),
                        "slug": slug,
                        "url": problem_url,
                        "difficulty": difficulty,
                        "acceptance_rate": problem_details.get("acceptance_rate", 0),
                        "description": problem_details.get("description", ""),
                        "examples": problem_details.get("examples", ""),
                        "constraints": problem_details.get("constraints", ""),
                        "companies": problem_details.get("companies", []),
                        "topics": [category],
                        "last_scraped": datetime.now().isoformat()
                    }

                    category_problems.append(problem_info)

                    # Be gentle with the server
                    time.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error processing InterviewBit problem {problem_url}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping InterviewBit category {category_url}: {e}")

        return category_problems

    def _get_problem_details(self, problem_url):
        """Get detailed description and examples for a problem"""
        try:
            response = requests.get(problem_url, headers=self.headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Get problem title
            title_elem = soup.select_one("h1.problem-title")
            title = title_elem.get_text().strip() if title_elem else ""

            # Get problem description
            desc_elem = soup.select_one(".problem-description")
            description = desc_elem.get_text().strip() if desc_elem else ""

            # Get examples - usually part of the description in InterviewBit
            examples = ""
            if "Example" in description or "Input" in description:
                parts = description.split("Example")
                if len(parts) > 1:
                    examples = "Example" + parts[1]
                    description = parts[0].strip()

            # Get difficulty text if available
            difficulty_elem = soup.select_one(".difficulty")
            difficulty_text = difficulty_elem.get_text().strip() if difficulty_elem else ""

            # Try to extract company information
            companies = []
            company_section = soup.select_one(".company-tag-container")
            if company_section:
                company_elems = company_section.select(".company-tag")
                companies = [elem.get_text().strip() for elem in company_elems]

            # Extract constraints if available
            constraints = ""
            if "Constraints" in description:
                parts = description.split("Constraints")
                if len(parts) > 1:
                    constraints = "Constraints" + parts[1]
                    description = parts[0].strip()

            return {
                "title": title,
                "description": description,
                "examples": examples,
                "constraints": constraints,
                "companies": companies,
                "difficulty_text": difficulty_text,
                "acceptance_rate": 0  # InterviewBit doesn't show this
            }

        except Exception as e:
            logger.error(f"Error fetching InterviewBit problem details: {e}")
            return None

class ProblemDatabase:
    """Database to store and manage coding problems"""

    def __init__(self, db_file="problem_database.json"):
        self.db_file = db_file
        self.problems = []
        self.load()

    def load(self):
        """Load problems from database file"""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.problems = json.load(f)
                logger.info(f"Loaded {len(self.problems)} problems from database")
            else:
                logger.info("No existing database found, starting fresh")
                self.problems = []
        except Exception as e:
            logger.error(f"Error loading problem database: {e}")
            self.problems = []

    def save(self):
        """Save problems to database file"""
        try:
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.problems, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.problems)} problems to database")
        except Exception as e:
            logger.error(f"Error saving problem database: {e}")

    def update(self, new_problems):
        """Update database with new problems"""
        if not new_problems:
            logger.info("No new problems to update database with")
            return

        # Create lookup for existing problems
        existing_ids = {p["id"]: i for i, p in enumerate(self.problems)}

        added = 0
        updated = 0

        for problem in new_problems:
            if problem["id"] in existing_ids:
                # Update existing problem
                self.problems[existing_ids[problem["id"]]] = problem
                updated += 1
            else:
                # Add new problem
                self.problems.append(problem)
                added += 1

        logger.info(f"Database update complete: {added} added, {updated} updated")
        self.save()

    def get_problems(self, filters=None):
        """Get problems matching specified filters"""
        if not filters:
            return self.problems

        filtered_problems = self.problems

        if "difficulty" in filters:
            filtered_problems = [p for p in filtered_problems
                                if p.get("difficulty") == filters["difficulty"]]

        if "company" in filters:
            filtered_problems = [p for p in filtered_problems
                                if filters["company"] in p.get("companies", [])]

        if "topic" in filters:
            filtered_problems = [p for p in filtered_problems
                                if filters["topic"] in p.get("topics", [])]

        if "source" in filters:
            filtered_problems = [p for p in filtered_problems
                                if p.get("source") == filters["source"]]

        return filtered_problems

    def get_problem_by_id(self, problem_id):
        """Get a specific problem by ID"""
        for problem in self.problems:
            if problem.get("id") == problem_id:
                return problem
        return None

    def get_stats(self):
        """Get statistics about the database"""
        companies = {}
        topics = {}
        sources = {}
        difficulties = {1: 0, 2: 0, 3: 0}

        for problem in self.problems:
            # Count by company
            for company in problem.get("companies", []):
                companies[company] = companies.get(company, 0) + 1

            # Count by topic
            for topic in problem.get("topics", []):
                topics[topic] = topics.get(topic, 0) + 1

            # Count by source
            source = problem.get("source", "Unknown")
            sources[source] = sources.get(source, 0) + 1

            # Count by difficulty
            difficulty = problem.get("difficulty", 2)
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        return {
            "total_problems": len(self.problems),
            "companies": companies,
            "topics": topics,
            "sources": sources,
            "difficulties": difficulties
        }

class CodeAnalyzer:
    """Class to analyze code quality, complexity, and correctness"""

    def __init__(self):
        pass

    def analyze_time_complexity(self, code):
        """Analyze the time complexity of code"""
        code_lower = code.lower()

        # Look for explicit complexity comments
        complexity_patterns = {
            r'o$1$': 'O(1)',
            r'o$log n$': 'O(log n)',
            r'o$n$': 'O(n)',
            r'o$n log n$': 'O(n log n)',
            r'o$n\^2$': 'O(n²)',
            r'o$n\^3$': 'O(n³)',
            r'o$2\^n$': 'O(2ⁿ)',
            r'o$n!$': 'O(n!)'
        }

        for pattern, complexity in complexity_patterns.items():
            if re.search(pattern, code_lower):
                return complexity

        # Analyze code structure for complexity patterns
        if re.search(r'for.*for.*for.*for', code_lower):
            return 'O(n⁴)'
        elif re.search(r'for.*for.*for', code_lower):
            return 'O(n³)'
        elif re.search(r'for.*for', code_lower):
            return 'O(n²)'
        elif re.search(r'for.*mid.*=.*$.*\+.*$.*[/].*2', code_lower) or \
             re.search(r'while.*left.*<.*right', code_lower):
            return 'O(n log n)'
        elif re.search(r'for|while', code_lower):
            return 'O(n)'

        # Check for recursive patterns
        if 'def' in code_lower and re.search(r'return.*\w+$', code_lower):
            if 'memo' in code_lower or 'cache' in code_lower or '@lru_cache' in code_lower:
                return 'O(n)'
            elif re.search(r'return.*\w+$.*[+-].*1', code_lower):
                return 'O(2ⁿ)'

        # Default
        return 'O(n)'

    def analyze_space_complexity(self, code):
        """Estimate space complexity"""
        code_lower = code.lower()

        # Look for explicit space complexity comments
        complexity_patterns = {
            r'space.*o$1$': 'O(1)',
            r'space.*o$log n$': 'O(log n)',
            r'space.*o$n$': 'O(n)',
            r'space.*o$n\^2$': 'O(n²)'
        }

        for pattern, complexity in complexity_patterns.items():
            if re.search(pattern, code_lower):
                return complexity

        # Analyze code structure for space complexity
        if re.search(r'=\s*$$\s*$$', code_lower):  # 2D arrays/matrices
            return 'O(n²)'
        elif re.search(r'=\s*$$|=\s*dict$|=\s*set$|=\s*{', code_lower):
            return 'O(n)'

        # Check for recursion without memoization
        if 'def' in code_lower and re.search(r'return.*\w+$', code_lower):
            if 'memo' not in code_lower and 'cache' not in code_lower and '@lru_cache' not in code_lower:
                return 'O(n)'

        # Default
        return 'O(1)'

    def analyze_algorithm_pattern(self, code):
        """Identify common algorithm patterns in the code"""
        code_lower = code.lower()
        patterns = []

        # Algorithm pattern identification
        if re.search(r'memo|cache|@lru_cache|dp$$', code_lower):
            patterns.append('Dynamic Programming')

        if re.search(r'left.*=.*mid|right.*=.*mid', code_lower):
            patterns.append('Binary Search')

        if re.search(r'queue|deque', code_lower):
            if re.search(r'level|depth', code_lower):
                patterns.append('BFS')
            else:
                patterns.append('Queue-based Algorithm')

        if re.search(r'heapq|heappush|priority_queue', code_lower):
            patterns.append('Heap/Priority Queue')

        if re.search(r'def dfs|def depth', code_lower) or \
           (re.search(r'def .+$', code_lower) and re.search(r'return.*\w+$', code_lower)):
            patterns.append('DFS/Recursion')

        if 'sort' in code_lower:
            patterns.append('Sorting')

        if re.search(r'left.*right|start.*end|slow.*fast', code_lower):
            patterns.append('Two Pointers')

        if re.search(r'dict$|set$|map|counter|{}|$$$$', code_lower):
            patterns.append('Hash Table')

        if re.search(r'graph|adjacent|neighbor', code_lower):
            patterns.append('Graph Algorithm')

        if re.search(r'node|tree|root|left|right', code_lower):
            patterns.append('Tree Algorithm')

        if re.search(r'max$|min$|greedy', code_lower):
            patterns.append('Greedy')

        if len(patterns) == 0:
            patterns.append('Basic Implementation')

        return patterns

    def execute_code(self, code, problem, timeout=5):
        """
        Execute Python code against test cases
        Returns results and any errors
        """
        # Safety checks
        if not code.strip():
            return {
                "success": False,
                "error": "No code provided",
                "results": []
            }

        # Create a dictionary for test results
        results = []
        all_passed = True

        # Try to execute each test case
        for i, test_case in enumerate(problem.get("test_cases", [])):
            # Skip if no test cases
            if not test_case:
                continue

            # Extract input and expected output
            try:
                test_input = test_case.get("input", {})
                expected_output = test_case.get("output")

                # Prepare code for execution with test case
                # We'll use a safer approach by writing to temp file and executing
                test_code = f"""
import sys
import json
import traceback

# The user's solution code
{code}

# Test execution
try:
    # Parse input
    test_input = {test_input}

    # Call the solution function with appropriate arguments
    # We need to determine the main function name
    function_match = None
    import re
    for line in {repr(code)}.split('\\n'):
        if line.strip().startswith('def '):
            function_match = re.match(r'def\\s+(\\w+)\\s*$', line)
            if function_match:
                break

    if not function_match:
        print(json.dumps({{
            "status": "error",
            "message": "Could not find a function definition"
        }}))
        sys.exit(1)

    function_name = function_match.group(1)

    # Get the function from locals
    solution_func = locals()[function_name]

    # Call the function with the input arguments
    result = solution_func(**test_input)

    print(json.dumps({{
        "status": "success",
        "result": result
    }}))

except Exception as e:
    print(json.dumps({{
        "status": "error",
        "message": str(e),
        "traceback": traceback.format_exc()
    }}))
"""

                # Create temporary file for test
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
                    temp_filename = temp.name
                    temp.write(test_code.encode('utf-8'))

                try:
                    # Execute with timeout
                    import subprocess
                    proc = subprocess.Popen(
                        [sys.executable, temp_filename],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                    stdout, stderr = proc.communicate(timeout=timeout)
                    stdout = stdout.decode('utf-8', errors='ignore')
                    stderr = stderr.decode('utf-8', errors='ignore')

                    # Parse results
                    try:
                        output_data = json.loads(stdout)

                        if output_data.get("status") == "success":
                            actual_output = output_data.get("result")
                            is_correct = self._compare_outputs(actual_output, expected_output)

                            result = {
                                "test_case": i+1,
                                "passed": is_correct,
                                "input": str(test_input),
                                "expected": str(expected_output),
                                "actual": str(actual_output),
                                "error": None
                            }

                            if not is_correct:
                                all_passed = False

                        else:
                            all_passed = False
                            result = {
                                "test_case": i+1,
                                "passed": False,
                                "input": str(test_input),
                                "expected": str(expected_output),
                                "actual": None,
                                "error": output_data.get("message")
                            }

                    except json.JSONDecodeError:
                        all_passed = False
                        result = {
                            "test_case": i+1,
                            "passed": False,
                            "input": str(test_input),
                            "expected": str(expected_output),
                            "actual": None,
                            "error": f"Failed to parse output: {stdout[:100]}..."
                        }

                except subprocess.TimeoutExpired:
                    proc.kill()
                    all_passed = False
                    result = {
                        "test_case": i+1,
                        "passed": False,
                        "input": str(test_input),
                        "expected": str(expected_output),
                        "actual": None,
                        "error": f"Execution timed out after {timeout} seconds"
                    }

                finally:
                    # Clean up
                    try:
                        os.unlink(temp_filename)
                    except:
                        pass

            except Exception as e:
                all_passed = False
                result = {
                    "test_case": i+1,
                    "passed": False,
                    "input": str(test_input) if "test_input" in locals() else "Unknown",
                    "expected": str(expected_output) if "expected_output" in locals() else "Unknown",
                    "actual": None,
                    "error": f"Execution error: {str(e)}"
                }

            results.append(result)

        return {
            "success": all_passed,
            "error": None if all_passed else "Some test cases failed",
            "results": results
        }

    def _compare_outputs(self, actual, expected):
        """Compare actual output with expected output, handling different formats"""
        # Convert to string for flexible comparison
        str_actual = str(actual).strip()
        str_expected = str(expected).strip()

        # Direct equality check
        if actual == expected:
            return True

        # Handle list variations
        if isinstance(actual, list) and isinstance(expected, list):
            # Check if lists have the same elements, regardless of order
            if sorted(str(actual)) == sorted(str(expected)):
                return True

        # Handle string representations
        str_actual = str_actual.replace(" ", "").replace("'", '"')
        str_expected = str_expected.replace(" ", "").replace("'", '"')

        if str_actual == str_expected:
            return True

        # For numeric results, allow small differences
        try:
            float_actual = float(str_actual)
            float_expected = float(str_expected)
            if abs(float_actual - float_expected) < 1e-9:
                return True
        except:
            pass

        return False

    def analyze_code_quality(self, code):
        """Analyze code quality metrics"""
        lines = [line for line in code.split("\n")
                if line.strip() and not line.strip().startswith("#")]
        loc = len(lines)

        # Check variable naming
        camel_case = len(re.findall(r'\b[a-z][a-zA-Z0-9]*[A-Z]', code))
        snake_case = len(re.findall(r'\b[a-z][a-z0-9]*_[a-z0-9]', code))
        naming_consistency = "Mixed" if camel_case > 0 and snake_case > 0 else (
                            "Snake case" if snake_case > 0 else
                            "Camel case" if camel_case > 0 else
                            "Unknown")

        # Check for comments
        comments = len(re.findall(r'#.*\$|\""".*?\"""|\'\'\'.*?\'\'\'', code, re.MULTILINE | re.DOTALL))
        comment_ratio = comments / max(1, loc)

        # Check functions
        functions = len(re.findall(r'def\s+\w+\s*$', code))

        # Check for error handling
        has_error_handling = "try" in code and "except" in code

        # Calculate cyclomatic complexity (simplified)
        decision_points = len(re.findall(r'\bif\b|\bfor\b|\bwhile\b|\band\b|\bor\b', code))
        complexity_score = 1 + decision_points

        # Readability score
        readability = 5  # Start with neutral score

        # Good practices
        if re.search(r'def\s+[a-z_][a-z0-9_]*\s*$', code):  # Proper function naming
            readability += 1

        if re.search(r'"""', code) or re.search(r"'''", code):  # Docstrings
            readability += 1

        if re.search(r'^\s*#', code, re.MULTILINE):  # Comments
            readability += 1

        # Consistent indentation
        indents = re.findall(r'^\s+', code, re.MULTILINE)
        if indents and all(len(i) % 4 == 0 for i in indents):
            readability += 1

        # Bad practices
        if re.search(r'\b[a-z]{1}\b', code):  # Single-letter variables
            readability -= 1

        # Super long lines
        long_lines = [line for line in code.split('\n') if len(line.strip()) > 100]
        if long_lines:
            readability -= 1

        # Nested loops/conditionals
        if re.search(r'if.*if.*if|for.*for.*for', code):
            readability -= 1

        # Clamp to 1-10 range
        readability = max(1, min(10, readability))

        return {
            "lines_of_code": loc,
            "naming_convention": naming_consistency,
            "comment_ratio": comment_ratio,
            "function_count": functions,
            "has_error_handling": has_error_handling,
            "complexity_score": complexity_score,
            "readability": readability
        }


    def evaluate_solution(self, code, problem):
            """Complete evaluation of a solution"""
            # First analyze the code structure
            time_complexity = self.analyze_time_complexity(code)
            space_complexity = self.analyze_space_complexity(code)
            algorithm_patterns = self.analyze_algorithm_pattern(code)
            code_quality = self.analyze_code_quality(code)

            # Execute against test cases
            execution_results = self.execute_code(code, problem)

            # Score the solution (0-10)
            score = 0

            # Correctness is most important (0-5 points)
            if execution_results["success"]:
                score += 5
            elif execution_results["results"] and any(r["passed"] for r in execution_results["results"]):
                # Partial credit for some passing tests
                passing = sum(1 for r in execution_results["results"] if r["passed"])
                total = len(execution_results["results"])
                score += 5 * (passing / total)

            # Time complexity (0-2 points)
            optimal_time = problem.get("optimal_time_complexity", problem.get("time_complexity", "O(n)"))
            complexity_rank = {
                "O(1)": 1,
                "O(log n)": 2,
                "O(n)": 3,
                "O(n log n)": 4,
                "O(n²)": 5,
                "O(n³)": 6,
                "O(2ⁿ)": 7,
                "O(n!)": 8,
                "Unknown": 9
            }

            user_rank = complexity_rank.get(time_complexity, 9)
            optimal_rank = complexity_rank.get(optimal_time, 3)

            if user_rank <= optimal_rank:
                score += 2  # Optimal or better
            elif user_rank == optimal_rank + 1:
                score += 1  # One level worse

            # Code quality (0-3 points)
            score += min(3, code_quality["readability"] / 3)

            return {
                "time_complexity": time_complexity,
                "space_complexity": space_complexity,
                "algorithm_patterns": algorithm_patterns,
                "execution_results": execution_results,
                "code_quality": code_quality,
                "overall_score": round(score, 2),
                "max_score": 10
            }


class CompanyPredictor:
    """Predict suitable companies based on coding performance"""

    def __init__(self, model_file="company_predictor_model.pkl"):
        self.model_file = model_file
        self.model = None
        self.companies = []
        self.feature_scaler = None

    def load_model(self):
        """Load trained model if available"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.companies = model_data.get('companies', [])
                    self.feature_scaler = model_data.get('scaler')
                logger.info(f"Loaded company prediction model with {len(self.companies)} companies")
                return True
            except Exception as e:
                logger.error(f"Error loading prediction model: {e}")
        return False

    def save_model(self):
        """Save trained model"""
        if self.model and self.companies:
            try:
                with open(self.model_file, 'wb') as f:
                    pickle.dump({
                        'model': self.model,
                        'companies': self.companies,
                        'scaler': self.feature_scaler
                    }, f)
                logger.info("Saved company prediction model")
                return True
            except Exception as e:
                logger.error(f"Error saving prediction model: {e}")
        return False

    def train_model(self, problem_db):
        """Train prediction model using problem database"""
        logger.info("Training company prediction model...")

        # Gather company-specific patterns from real problems
        company_problems = {}
        all_companies = set()

        for problem in problem_db.problems:
            for company in problem.get("companies", []):
                if company not in company_problems:
                    company_problems[company] = []
                company_problems[company].append({
                    "difficulty": problem.get("difficulty", 2),
                    "topics": problem.get("topics", []),
                    "acceptance_rate": problem.get("acceptance_rate", 0)
                })
                all_companies.add(company)

        # Keep only companies with sufficient problems
        self.companies = [company for company, problems in company_problems.items() if len(problems) >= 3]

        if not self.companies:
            logger.warning("Not enough company data to train a model")
            return False

        logger.info(f"Training model with data from {len(self.companies)} companies")

        # Generate feature matrix
        X = []
        y = []

        # Feature extraction function
        def extract_features(student_performance):
            # Difficulty-based scores
            easy_score = sum(p["overall_score"] for p in student_performance if p["difficulty"] == 1) / max(1, sum(1 for p in student_performance if p["difficulty"] == 1))
            medium_score = sum(p["overall_score"] for p in student_performance if p["difficulty"] == 2) / max(1, sum(1 for p in student_performance if p["difficulty"] == 2))
            hard_score = sum(p["overall_score"] for p in student_performance if p["difficulty"] == 3) / max(1, sum(1 for p in student_performance if p["difficulty"] == 3))

            # Algorithm pattern scores
            algorithm_counts = {}
            for p in student_performance:
                for pattern in p.get("algorithm_patterns", []):
                    algorithm_counts[pattern] = algorithm_counts.get(pattern, 0) + 1

            # Time complexity score (lower is better)
            complexity_scores = []
            for p in student_performance:
                complexity = p.get("time_complexity", "Unknown")
                if complexity == "O(1)":
                    complexity_scores.append(1)
                elif complexity == "O(log n)":
                    complexity_scores.append(2)
                elif complexity == "O(n)":
                    complexity_scores.append(3)
                elif complexity == "O(n log n)":
                    complexity_scores.append(4)
                elif complexity == "O(n²)":
                    complexity_scores.append(5)
                else:
                    complexity_scores.append(6)

            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 4

            # Code quality
            avg_readability = sum(p.get("code_quality", {}).get("readability", 5) for p in student_performance) / len(student_performance)

            # Speed score
            correct_problems = [p for p in student_performance if p.get("execution_results", {}).get("success", False)]
            avg_time_ratio = 0
            if correct_problems:
                time_ratios = []
                for p in correct_problems:
                    if "time_taken" in p and "difficulty" in p:
                        # Normalize by difficulty: harder problems allow more time
                        expected_time = p["difficulty"] * 300  # 5 minutes for easy, 10 for medium, 15 for hard
                        ratio = min(1.0, expected_time / max(1, p["time_taken"]))
                        time_ratios.append(ratio)
                if time_ratios:
                    avg_time_ratio = sum(time_ratios) / len(time_ratios)

            # Return all features
            return [
                easy_score,
                medium_score,
                hard_score,
                avg_complexity,
                avg_readability,
                avg_time_ratio,
                algorithm_counts.get("Dynamic Programming", 0),
                algorithm_counts.get("Binary Search", 0),
                algorithm_counts.get("Graph Algorithm", 0),
                algorithm_counts.get("Tree Algorithm", 0)
            ]

        # Generate synthetic data for each company based on their problem patterns
        for company in self.companies:
            problems = company_problems[company]

            # Analyze company problem patterns
            avg_difficulty = sum(p["difficulty"] for p in problems) / len(problems)
            topic_distribution = {}
            for p in problems:
                for topic in p.get("topics", []):
                    topic_distribution[topic] = topic_distribution.get(topic, 0) + 1

            # Generate synthetic student performances that match this company's profile
            # Good matches
            for _ in range(20):
                synthetic_performance = []
                for _ in range(5):  # 5 problems per student
                    # Generate a problem with similar properties to this company's problems
                    problem_difficulty = round(min(3, max(1, random.gauss(avg_difficulty, 0.5))))

                    # Student performed well on this problem
                    synthetic_performance.append({
                        "difficulty": problem_difficulty,
                        "overall_score": random.uniform(7, 10),  # High score
                        "time_complexity": "O(n)" if random.random() < 0.7 else "O(n log n)",
                        "algorithm_patterns": random.sample(list(topic_distribution.keys()), k=min(3, len(topic_distribution))),
                        "execution_results": {"success": True},
                        "code_quality": {"readability": random.uniform(7, 10)},
                        "time_taken": problem_difficulty * 180 * random.uniform(0.7, 1.0)  # Reasonably fast
                    })

                X.append(extract_features(synthetic_performance))
                y.append(1)  # Good match

            # Poor matches
            for _ in range(10):
                synthetic_performance = []
                for _ in range(5):  # 5 problems per student
                    problem_difficulty = round(min(3, max(1, random.gauss(avg_difficulty, 0.5))))

                    # Student performed poorly on this problem
                    synthetic_performance.append({
                        "difficulty": problem_difficulty,
                        "overall_score": random.uniform(1, 5),  # Low score
                        "time_complexity": "O(n²)" if random.random() < 0.7 else "O(n³)",
                        "algorithm_patterns": random.sample(list(topic_distribution.keys()), k=min(2, len(topic_distribution))),
                        "execution_results": {"success": random.random() < 0.3},  # Mostly fail
                        "code_quality": {"readability": random.uniform(3, 6)},
                        "time_taken": problem_difficulty * 300 * random.uniform(1.2, 2.0)  # Slow
                    })

                X.append(extract_features(synthetic_performance))
                y.append(0)  # Poor match

        # Train model
        X = np.array(X)
        y = np.array(y)

        # Scale features
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)

        # Train a random forest model for each company
        self.model = {}
        for i, company in enumerate(self.companies):
            # Create binary labels for this company
            y_company = np.zeros_like(y)
            company_instances = i * 30  # 30 instances per company (20 good, 10 poor)
            y_company[company_instances:company_instances+20] = 1  # Good matches

            # Train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_scaled, y_company)
            self.model[company] = clf

        self.save_model()
        logger.info(f"Model training complete for {len(self.companies)} companies")
        return True

    def predict_companies(self, student_performance):
        """Predict suitable companies based on student performance"""
        if not self.model:
            if not self.load_model():
                logger.error("No trained model available for prediction")
                return []

        # Extract features
        features = []

        # Difficulty-based scores
        easy_score = sum(p["overall_score"] for p in student_performance if p["difficulty"] == 1) / max(1, sum(1 for p in student_performance if p["difficulty"] == 1))
        medium_score = sum(p["overall_score"] for p in student_performance if p["difficulty"] == 2) / max(1, sum(1 for p in student_performance if p["difficulty"] == 2))
        hard_score = sum(p["overall_score"] for p in student_performance if p["difficulty"] == 3) / max(1, sum(1 for p in student_performance if p["difficulty"] == 3))

        # Algorithm pattern scores
        algorithm_counts = {}
        for p in student_performance:
            for pattern in p.get("algorithm_patterns", []):
                algorithm_counts[pattern] = algorithm_counts.get(pattern, 0) + 1

        # Time complexity score
        complexity_scores = []
        for p in student_performance:
            complexity = p.get("time_complexity", "Unknown")
            if complexity == "O(1)":
                complexity_scores.append(1)
            elif complexity == "O(log n)":
                complexity_scores.append(2)
            elif complexity == "O(n)":
                complexity_scores.append(3)
            elif complexity == "O(n log n)":
                complexity_scores.append(4)
            elif complexity == "O(n²)":
                complexity_scores.append(5)
            else:
                complexity_scores.append(6)

        avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 4

        # Code quality
        avg_readability = sum(p.get("code_quality", {}).get("readability", 5) for p in student_performance) / len(student_performance)

        # Speed score
        correct_problems = [p for p in student_performance if p.get("execution_results", {}).get("success", False)]
        avg_time_ratio = 0
        if correct_problems:
            time_ratios = []
            for p in correct_problems:
                if "time_taken" in p and "difficulty" in p:
                    # Normalize by difficulty: harder problems allow more time
                    expected_time = p["difficulty"] * 300  # 5 minutes for easy, 10 for medium, 15 for hard
                    ratio = min(1.0, expected_time / max(1, p["time_taken"]))
                    time_ratios.append(ratio)
            if time_ratios:
                avg_time_ratio = sum(time_ratios) / len(time_ratios)

        features = [
            easy_score,
            medium_score,
            hard_score,
            avg_complexity,
            avg_readability,
            avg_time_ratio,
            algorithm_counts.get("Dynamic Programming", 0),
            algorithm_counts.get("Binary Search", 0),
            algorithm_counts.get("Graph Algorithm", 0),
            algorithm_counts.get("Tree Algorithm", 0)
        ]

        # Scale features
        if self.feature_scaler:
            features = self.feature_scaler.transform([features])[0]

        # Make predictions for each company
        predictions = []
        for company, clf in self.model.items():
            prob = clf.predict_proba([features])[0][1]  # Probability of positive class
            predictions.append({
                "company": company,
                "match_score": float(prob * 100)  # Convert to percentage
            })

        # Sort by match score
        predictions.sort(key=lambda x: x["match_score"], reverse=True)

        return predictions


class DataManager:
    """Class to manage data collection, storage, and model training"""

    def __init__(self):
        self.problem_db = ProblemDatabase()
        self.company_predictor = CompanyPredictor()
        self.scrapers = [
            LeetCodeScraper(),
            GeeksForGeeksScraper(),
            InterviewBitScraper()
        ]

    def initialize(self):
        """Initialize system - load database and models"""
        self.problem_db.load()
        self.company_predictor.load_model()

        # Check if we need to perform an initial scrape
        if len(self.problem_db.problems) < 100:
            logger.info("Initial database is small, performing first-time scraping...")
            self.update_problem_database()
            self.train_models()

        logger.info("DataManager initialized successfully")

    def update_problem_database(self):
        """Update problem database with fresh data from scrapers"""
        logger.info("Starting problem database update...")

        all_problems = []
        for scraper in self.scrapers:
            problems = scraper.scrape_problems(limit=200)
            logger.info(f"Scraped {len(problems)} problems from {scraper.__class__.__name__}")
            all_problems.extend(problems)

        if all_problems:
            self.problem_db.update(all_problems)
            logger.info(f"Added/updated {len(all_problems)} problems in database")
            return True
        else:
            logger.warning("No problems scraped, database not updated")
            return False

    def train_models(self):
        """Train prediction models using current data"""
        logger.info("Training models...")
        result = self.company_predictor.train_model(self.problem_db)
        if result:
            logger.info("Models trained successfully")
        else:
            logger.warning("Model training failed or was skipped")
        return result

    def start_scheduled_updates(self, interval_hours=24):
        """Start scheduled updates for data and models"""
        def update_job():
            logger.info(f"Running scheduled update (every {interval_hours} hours)")
            if self.update_problem_database():
                self.train_models()

        # Run immediately for first time
        update_job()

        # Schedule regular updates
        schedule.every(interval_hours).hours.do(update_job)

        # Start the background scheduler
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)

        # Run scheduler in background thread
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()

        logger.info(f"Scheduled updates started (every {interval_hours} hours)")


# Initialize components
data_manager = DataManager()
code_analyzer = CodeAnalyzer()

# Flask routes
@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to check if the server is running"""
    return jsonify({
        "status": "online",
        "database_size": len(data_manager.problem_db.problems) if data_manager.problem_db.problems else 0
    })

@app.route('/problems', methods=['GET'])
def get_problems():
    """Get problems with optional filtering"""
    difficulty = request.args.get('difficulty', type=int)
    company = request.args.get('company')
    topic = request.args.get('topic')
    source = request.args.get('source')
    limit = request.args.get('limit', 10, type=int)

    filters = {}
    if difficulty:
        filters["difficulty"] = difficulty
    if company:
        filters["company"] = company
    if topic:
        filters["topic"] = topic
    if source:
        filters["source"] = source

    problems = data_manager.problem_db.get_problems(filters)

    # Limit the number of returned problems
    if limit > 0 and limit < len(problems):
        problems = random.sample(problems, limit)

    return jsonify({
        "count": len(problems),
        "problems": problems
    })

@app.route('/problem/<problem_id>', methods=['GET'])
def get_problem(problem_id):
    """Get a specific problem by ID"""
    problem = data_manager.problem_db.get_problem_by_id(problem_id)
    if problem:
        return jsonify(problem)
    else:
        return jsonify({"error": "Problem not found"}), 404

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the database"""
    stats = data_manager.problem_db.get_stats()
    return jsonify(stats)

@app.route('/evaluate', methods=['POST'])
def evaluate_solution():
    """Evaluate a solution to a specific problem"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    code = data.get('code')
    problem_id = data.get('problem_id')
    time_taken = data.get('time_taken', 0)  # In seconds

    if not code:
        return jsonify({"error": "No code provided"}), 400

    if not problem_id:
        return jsonify({"error": "No problem_id provided"}), 400

    # Get the problem
    problem = data_manager.problem_db.get_problem_by_id(problem_id)
    if not problem:
        return jsonify({"error": f"Problem with ID {problem_id} not found"}), 404

    # Evaluate the solution
    evaluation = code_analyzer.evaluate_solution(code, problem)

    # Add time taken to the evaluation
    evaluation["time_taken"] = time_taken
    evaluation["difficulty"] = problem.get("difficulty", 2)

    return jsonify({
        "problem_id": problem_id,
        "evaluation": evaluation
    })

@app.route('/evaluate_custom', methods=['POST'])
def evaluate_custom_solution():
    """Evaluate a solution to a custom problem"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    code = data.get('code')
    problem = data.get('problem')
    time_taken = data.get('time_taken', 0)  # In seconds

    if not code:
        return jsonify({"error": "No code provided"}), 400

    if not problem:
        return jsonify({"error": "No problem definition provided"}), 400

    # Evaluate the solution
    evaluation = code_analyzer.evaluate_solution(code, problem)

    # Add time taken to the evaluation
    evaluation["time_taken"] = time_taken
    evaluation["difficulty"] = problem.get("difficulty", 2)

    return jsonify({
        "evaluation": evaluation
    })

@app.route('/predict', methods=['POST'])
def predict_companies():
    """Predict suitable companies based on performance on multiple problems"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    performance_data = data.get('performance', [])

    if not performance_data or len(performance_data) == 0:
        return jsonify({"error": "No performance data provided"}), 400

    predictions = data_manager.company_predictor.predict_companies(performance_data)

    # Get number of problems by difficulty
    difficulties = {1: 0, 2: 0, 3: 0}
    for p in performance_data:
        difficulty = p.get("difficulty", 2)
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

    # Calculate overall performance metrics
    avg_score = sum(p.get("overall_score", 0) for p in performance_data) / len(performance_data)

    return jsonify({
        "predictions": predictions,
        "performance_summary": {
            "problems_solved": len(performance_data),
            "difficulty_distribution": difficulties,
            "average_score": avg_score
        }
    })

@app.route('/update_database', methods=['POST'])
def trigger_update():
    """Manually trigger database update"""
    auth_key = request.json.get('auth_key', '')
    if auth_key != os.environ.get('ADMIN_KEY', 'admin_secret'):
        return jsonify({"error": "Unauthorized"}), 403

    # Trigger update in a background thread
    def update_job():
        data_manager.update_problem_database()
        data_manager.train_models()

    thread = threading.Thread(target=update_job)
    thread.daemon = True
    thread.start()

    return jsonify({"message": "Update job started"})

# Start the server
if __name__ == '__main__':
    # Initialize data manager
    data_manager.initialize()

    # Start scheduled updates (every 24 hours)
    data_manager.start_scheduled_updates()

    # Run Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)
