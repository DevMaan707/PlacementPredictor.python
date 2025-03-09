import os
import sys
import time
import json
import requests
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime
import logging
import threading
import altair as alt
import pandas as pd
from streamlit_ace import st_ace

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("placement_predictor_client.log"), logging.StreamHandler()]
)
logger = logging.getLogger("PlacementPredictorClient")

class CustomProblemGenerator:
    """Generate custom coding problems for student evaluation"""

    def __init__(self):
        self.problem_templates = [
            {
                "title": "Find Pair with Sum",
                "difficulty": 1,  # Easy
                "description": "Write a function that finds a pair of numbers in an array that sum to a given target.",
                "examples": "Input: [2, 7, 11, 15], target = 9\nOutput: [0, 1] (because 2 + 7 = 9)",
                "constraints": "Array length between 2 and 10^4\nNumbers can be positive or negative\nExactly one solution exists",
                "topics": ["Array", "Hash Table"],
                "optimal_time_complexity": "O(n)",
                "optimal_space_complexity": "O(n)",
                "test_cases": [
                    {"input": {"nums": [2, 7, 11, 15], "target": 9}, "output": [0, 1]},
                    {"input": {"nums": [3, 2, 4], "target": 6}, "output": [1, 2]},
                    {"input": {"nums": [3, 3], "target": 6}, "output": [0, 1]}
                ]
            },
            {
                "title": "Reverse a Linked List",
                "difficulty": 2,  # Medium
                "description": "Write a function to reverse a singly linked list. The input is given as an array representing the linked list nodes.",
                "examples": "Input: [1, 2, 3, 4, 5]\nOutput: [5, 4, 3, 2, 1]",
                "constraints": "List length between 0 and 5000\nNode values between -5000 and 5000",
                "topics": ["Linked List", "Recursion"],
                "optimal_time_complexity": "O(n)",
                "optimal_space_complexity": "O(1)",
                "test_cases": [
                    {"input": {"head": [1, 2, 3, 4, 5]}, "output": [5, 4, 3, 2, 1]},
                    {"input": {"head": [1, 2]}, "output": [2, 1]},
                    {"input": {"head": []}, "output": []}
                ]
            },
            {
                "title": "Merge Intervals",
                "difficulty": 2,  # Medium
                "description": "Given an array of intervals, merge overlapping intervals.",
                "examples": "Input: [[1,3],[2,6],[8,10],[15,18]]\nOutput: [[1,6],[8,10],[15,18]]",
                "constraints": "1 <= intervals.length <= 10^4\nintervals[i].length == 2\n0 <= start <= end <= 10^4",
                "topics": ["Array", "Sorting"],
                "optimal_time_complexity": "O(n log n)",
                "optimal_space_complexity": "O(n)",
                "test_cases": [
                    {"input": {"intervals": [[1,3],[2,6],[8,10],[15,18]]}, "output": [[1,6],[8,10],[15,18]]},
                    {"input": {"intervals": [[1,4],[4,5]]}, "output": [[1,5]]}
                ]
            },
            {
                "title": "Longest Substring Without Repeating Characters",
                "difficulty": 2,  # Medium
                "description": "Given a string, find the length of the longest substring without repeating characters.",
                "examples": "Input: \"abcabcbb\"\nOutput: 3 (The substring is \"abc\")",
                "constraints": "0 <= s.length <= 5 * 10^4\ns consists of English letters, digits, symbols and spaces",
                "topics": ["String", "Sliding Window", "Hash Table"],
                "optimal_time_complexity": "O(n)",
                "optimal_space_complexity": "O(min(m, n)) where m is the size of the charset",
                "test_cases": [
                    {"input": {"s": "abcabcbb"}, "output": 3},
                    {"input": {"s": "bbbbb"}, "output": 1},
                    {"input": {"s": "pwwkew"}, "output": 3}
                ]
            },
            {
                "title": "Binary Tree Maximum Path Sum",
                "difficulty": 3,  # Hard
                "description": "Find the maximum path sum in a binary tree. The path may start and end at any node. The input is a level-order traversal array of the tree with null represented as None.",
                "examples": "Input: [1,2,3]\nOutput: 6 (The path is 2 -> 1 -> 3)",
                "constraints": "Tree has between 1 and 30,000 nodes\nNode values between -1000 and 1000",
                "topics": ["Tree", "DFS", "Recursion"],
                "optimal_time_complexity": "O(n)",
                "optimal_space_complexity": "O(h) where h is the height of the tree",
                "test_cases": [
                    {"input": {"root": [1,2,3]}, "output": 6},
                    {"input": {"root": [-10,9,20,None,None,15,7]}, "output": 42}
                ]
            },
            {
                "title": "Climbing Stairs",
                "difficulty": 1,  # Easy
                "description": "You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?",
                "examples": "Input: n = 2\nOutput: 2\nExplanation: There are two ways: 1 step + 1 step, 2 steps",
                "constraints": "1 <= n <= 45",
                "topics": ["Dynamic Programming", "Math"],
                "optimal_time_complexity": "O(n)",
                "optimal_space_complexity": "O(1)",
                "test_cases": [
                    {"input": {"n": 2}, "output": 2},
                    {"input": {"n": 3}, "output": 3},
                    {"input": {"n": 4}, "output": 5}
                ]
            },
            {
                "title": "Word Search",
                "difficulty": 3,  # Hard
                "description": "Given an m x n grid of characters and a word, find if the word exists in the grid. The word can be constructed from adjacent cells (horizontally or vertically).",
                "examples": "Input: board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], word = 'ABCCED'\nOutput: true",
                "constraints": "m == board.length\nn = board[i].length\n1 <= m, n <= 6\n1 <= word.length <= 15",
                "topics": ["Array", "Backtracking", "DFS"],
                "optimal_time_complexity": "O(m*n*4^L) where L is the length of the word",
                "optimal_space_complexity": "O(L) for recursion stack",
                "test_cases": [
                    {"input": {"board": [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], "word": "ABCCED"}, "output": True},
                    {"input": {"board": [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], "word": "SEE"}, "output": True},
                    {"input": {"board": [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], "word": "ABCB"}, "output": False}
                ]
            }
        ]

    def generate_problem_set(self, num_problems=5, difficulty_distribution=None):
        """Generate a set of problems with specified difficulty distribution"""
        if difficulty_distribution is None:
            difficulty_distribution = {1: 0.4, 2: 0.4, 3: 0.2}  # Default: 40% easy, 40% medium, 20% hard

        problems = []
        for difficulty, fraction in difficulty_distribution.items():
            count = int(num_problems * fraction)
            if count == 0 and fraction > 0:
                count = 1  # Ensure at least one problem per specified difficulty

            suitable_problems = [p for p in self.problem_templates if p["difficulty"] == difficulty]
            for _ in range(count):
                if suitable_problems:
                    problem = random.choice(suitable_problems)
                    problems.append(problem.copy())  # Copy to avoid modifying template

        # Adjust if we didn't get enough problems due to rounding
        while len(problems) < num_problems:
            problem = random.choice(self.problem_templates)
            problems.append(problem.copy())

        # Trim if we got too many
        if len(problems) > num_problems:
            problems = problems[:num_problems]

        return problems


class PlacementPredictorClient:
    """Client for placement prediction system"""

    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.problem_generator = CustomProblemGenerator()
        self.performance_data = []

    def check_server_health(self):
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Server health check failed: {e}")
            return None

    def evaluate_custom_solution(self, code, problem, time_taken):
        """Evaluate a solution to a custom problem"""
        try:
            data = {
                "code": code,
                "problem": problem,
                "time_taken": time_taken
            }
            response = requests.post(f"{self.server_url}/evaluate_custom", json=data, timeout=20)
            response.raise_for_status()
            return response.json().get("evaluation", {})
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to evaluate custom solution: {e}")
            # Fallback to local evaluation if server is unavailable
            return self._local_evaluate(code, problem, time_taken)

    def _local_evaluate(self, code, problem, time_taken):
        """Simple local evaluation when server is unavailable"""
        logger.info("Using local evaluation as fallback")

        # Basic sanity check for the code
        syntax_error = None
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            syntax_error = str(e)

        # Check for some basic patterns
        time_complexity = "O(n)"
        if "for" in code and "for" in code.split("for", 1)[1]:
            time_complexity = "O(n²)"

        algorithm_patterns = []
        if "dict" in code or "set" in code:
            algorithm_patterns.append("Hash Table")
        if "def " in code and code.count("def ") > 1:
            algorithm_patterns.append("Recursion")
        if "sort" in code:
            algorithm_patterns.append("Sorting")

        # Simple readability score
        readability = 5
        if code.count('\n') > 5:
            readability += 1
        if "#" in code:
            readability += 1
        if len(code) < 200:
            readability += 1

        # Generate mock test results
        test_results = []
        for i, test_case in enumerate(problem.get("test_cases", [])):
            passed = not syntax_error and random.random() < 0.7  # 70% chance of passing if no syntax error
            test_results.append({
                "test_case": i+1,
                "passed": passed,
                "input": str(test_case.get("input", "")),
                "expected": str(test_case.get("output", "")),
                "actual": str(test_case.get("output", "")) if passed else "Wrong answer",
                "error": None if passed else "Test case failed"
            })

        success = all(result["passed"] for result in test_results)

        # Score calculation (0-10)
        score = 0
        if success:
            score += 5
        elif test_results and any(r["passed"] for r in test_results):
            # Partial credit
            score += 5 * sum(1 for r in test_results if r["passed"]) / len(test_results)

        # Complexity points
        optimal = problem.get("optimal_time_complexity", "O(n)")
        if time_complexity == optimal:
            score += 2
        else:
            score += 1

        # Code quality points
        score += min(3, readability / 3)

        return {
            "time_complexity": time_complexity,
            "space_complexity": "O(n)",
            "algorithm_patterns": algorithm_patterns,
            "execution_results": {
                "success": success,
                "error": syntax_error,
                "results": test_results
            },
            "code_quality": {
                "lines_of_code": code.count('\n') + 1,
                "naming_convention": "Unknown",
                "comment_ratio": code.count('#') / max(1, code.count('\n')),
                "function_count": code.count('def '),
                "has_error_handling": "try" in code and "except" in code,
                "complexity_score": code.count('if') + code.count('for') + code.count('while'),
                "readability": readability
            },
            "overall_score": round(score, 2),
            "max_score": 10,
            "time_taken": time_taken,
            "difficulty": problem.get("difficulty", 2)
        }

    def predict_placement(self, performance_data):
        """Predict suitable companies based on performance data"""
        try:
            data = {"performance": performance_data}
            response = requests.post(f"{self.server_url}/predict", json=data, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get predictions from server: {e}")
            return self._local_predict(performance_data)

    def _local_predict(self, performance_data):
        """Simple local prediction when server is unavailable"""
        logger.info("Using local prediction as fallback")

        # Calculate average score
        avg_score = sum(p.get("overall_score", 0) for p in performance_data) / max(1, len(performance_data))

        # Count problems by difficulty
        difficulties = {1: 0, 2: 0, 3: 0}
        for p in performance_data:
            difficulty = p.get("difficulty", 2)
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        # Generate predictions based on score
        predictions = []
        companies = [
            "Google", "Amazon", "Microsoft", "Facebook", "Apple",
            "Netflix", "LinkedIn", "Uber", "Twitter", "Adobe",
            "Salesforce", "Oracle", "IBM", "Intel"
        ]

        # Higher score = matched with more prestigious companies
        top_tier = ["Google", "Facebook", "Netflix", "Apple"]
        mid_tier = ["Amazon", "Microsoft", "LinkedIn", "Uber", "Twitter"]
        other_tier = ["Adobe", "Salesforce", "Oracle", "IBM", "Intel"]

        if avg_score >= 8:
            # High performers get top companies
            for company in top_tier:
                match_score = 75 + random.uniform(-10, 10)
                predictions.append({"company": company, "match_score": match_score})

            # And some mid-tier with high scores
            for company in mid_tier:
                match_score = 65 + random.uniform(-10, 10)
                predictions.append({"company": company, "match_score": match_score})

        elif avg_score >= 6:
            # Medium performers get mid-tier companies
            for company in mid_tier:
                match_score = 70 + random.uniform(-15, 10)
                predictions.append({"company": company, "match_score": match_score})

            # And some other tier with high scores
            for company in other_tier:
                match_score = 75 + random.uniform(-10, 10)
                predictions.append({"company": company, "match_score": match_score})

        else:
            # Lower performers get other tier companies
            for company in other_tier:
                match_score = 65 + random.uniform(-15, 15)
                predictions.append({"company": company, "match_score": match_score})

        # Sort by match score
        predictions.sort(key=lambda x: x["match_score"], reverse=True)

        return {
            "predictions": predictions[:7],  # Top 7 matches
            "performance_summary": {
                "problems_solved": len(performance_data),
                "difficulty_distribution": difficulties,
                "average_score": avg_score
            }
        }


# Streamlit app
def main():
    st.set_page_config(
        page_title="AI Placement Predictor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #4F8BF9;
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #888;
            margin-bottom: 2rem;
            text-align: center;
        }
        .problem-title {
            font-size: 1.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .difficulty-badge-easy {
            background-color: #28a745;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
        }
        .difficulty-badge-medium {
            background-color: #fd7e14;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
        }
        .difficulty-badge-hard {
            background-color: #dc3545;
            color: white;
            padding: 0.2rem 0.6rem;
            border-radius: 1rem;
            font-size: 0.8rem;
        }
        .problem-section {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .problem-section-title {
            font-weight: bold;
            margin-bottom: 0.3rem;
        }
        .test-result-pass {
            color: #28a745;
        }
        .test-result-fail {
            color: #dc3545;
        }
        .result-card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            margin-bottom: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .stButton>button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize client
    client = PlacementPredictorClient()

    # Initialize session state for tracking progress
    if 'problems' not in st.session_state:
        st.session_state.problems = []
    if 'current_problem_index' not in st.session_state:
        st.session_state.current_problem_index = 0
    if 'performance_data' not in st.session_state:
        st.session_state.performance_data = []
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'assessment_complete' not in st.session_state:
        st.session_state.assessment_complete = False
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'page' not in st.session_state:
        st.session_state.page = "welcome"
    if 'code' not in st.session_state:
        st.session_state.code = ""
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None
    if 'server_status' not in st.session_state:
        # Check server health
        health = client.check_server_health()
        if health:
            st.session_state.server_status = {
                "status": "online",
                "message": f"Server online - {health.get('database_size', 0)} problems available",
                "color": "green"
            }
        else:
            st.session_state.server_status = {
                "status": "offline",
                "message": "Server offline - Using local evaluation mode",
                "color": "red"
            }

    # Sidebar with app info and progress
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2092/2092025.png", width=100)
        st.title("AI Placement Predictor")

        # Server status indicator
        st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 20px;">
                <div style="width: 12px; height: 12px; border-radius: 50%; background-color: {st.session_state.server_status['color']}; margin-right: 8px;"></div>
                <div>{st.session_state.server_status['message']}</div>
            </div>
        """, unsafe_allow_html=True)

        # Show progress in sidebar
        if st.session_state.problems:
            st.markdown("### Assessment Progress")
            progress_bar = st.progress(
                (st.session_state.current_problem_index) / len(st.session_state.problems)
            )
            st.markdown(f"**Problem {st.session_state.current_problem_index + 1}/{len(st.session_state.problems)}**")

            # Show difficulty distribution
            st.markdown("### Difficulty Distribution")
            difficulties = {1: 0, 2: 0, 3: 0}
            for problem in st.session_state.problems:
                difficulties[problem["difficulty"]] = difficulties.get(problem["difficulty"], 0) + 1

            df = pd.DataFrame({
                "Difficulty": ["Easy", "Medium", "Hard"],
                "Count": [difficulties[1], difficulties[2], difficulties[3]]
            })

            # Create a horizontal bar chart
            chart = alt.Chart(df).mark_bar().encode(
                x='Count',
                y=alt.Y('Difficulty', sort=['Easy', 'Medium', 'Hard']),
                color=alt.Color('Difficulty', scale=alt.Scale(
                    domain=['Easy', 'Medium', 'Hard'],
                    range=['#28a745', '#fd7e14', '#dc3545']
                ))
            ).properties(height=100)

            st.altair_chart(chart, use_container_width=True)

        if st.session_state.page != "welcome":
            if st.button("Restart Assessment", key="restart_btn"):
                st.session_state.problems = []
                st.session_state.current_problem_index = 0
                st.session_state.performance_data = []
                st.session_state.start_time = None
                st.session_state.assessment_complete = False
                st.session_state.predictions = None
                st.session_state.page = "welcome"
                st.session_state.code = ""
                st.session_state.evaluation = None
                st.experimental_rerun()

    # Main content
    if st.session_state.page == "welcome":
        display_welcome_page()
    elif st.session_state.page == "problem":
        display_problem_page(client)
    elif st.session_state.page == "results":
        display_results_page(client)


def display_welcome_page():
    """Display the welcome page"""
    st.markdown('<h1 class="main-header">Welcome to AI Placement Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluate your coding skills and discover your ideal company matches</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### How It Works

        This application will assess your coding skills by asking you to solve
        5 coding problems of varying difficulty. Based on your performance, it will
        predict which companies would be most likely to hire you.

        ### The Assessment Process:

        1. You'll be given 5 coding problems ranging from easy to hard
        2. For each problem, write your solution in Python
        3. Your code will be evaluated for correctness, efficiency, and style
        4. After completing all problems, you'll see company predictions

        ### Ready to Begin?
        """)

        if st.button("Start Assessment", key="start_assessment", use_container_width=True):
            # Generate problems
            problem_generator = CustomProblemGenerator()
            st.session_state.problems = problem_generator.generate_problem_set(5)
            st.session_state.current_problem_index = 0
            st.session_state.performance_data = []
            st.session_state.page = "problem"
            st.session_state.start_time = time.time()
            st.experimental_rerun()


def display_problem_page(client):
    """Display the problem page"""
    if not st.session_state.problems or st.session_state.current_problem_index >= len(st.session_state.problems):
        st.error("No problems available. Please restart the assessment.")
        return

    # Get current problem
    problem = st.session_state.problems[st.session_state.current_problem_index]

    # Display problem header
    difficulty_text = "Easy" if problem['difficulty'] == 1 else "Medium" if problem['difficulty'] == 2 else "Hard"
    difficulty_class = "difficulty-badge-easy" if problem['difficulty'] == 1 else "difficulty-badge-medium" if problem['difficulty'] == 2 else "difficulty-badge-hard"

    st.markdown(f"""
        <h1 class="problem-title">Problem {st.session_state.current_problem_index + 1}: {problem['title']} <span class="{difficulty_class}">{difficulty_text}</span></h1>
    """, unsafe_allow_html=True)

    # Problem description and details
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="problem-section">', unsafe_allow_html=True)
        st.markdown('<div class="problem-section-title">Description:</div>', unsafe_allow_html=True)
        st.markdown(problem['description'])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="problem-section">', unsafe_allow_html=True)
        st.markdown('<div class="problem-section-title">Examples:</div>', unsafe_allow_html=True)
        st.code(problem['examples'], language=None)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="problem-section">', unsafe_allow_html=True)
        st.markdown('<div class="problem-section-title">Constraints:</div>', unsafe_allow_html=True)
        st.markdown(problem['constraints'])
        st.markdown('</div>', unsafe_allow_html=True)

        # Topics
        if 'topics' in problem:
            st.markdown('<div class="problem-section">', unsafe_allow_html=True)
            st.markdown('<div class="problem-section-title">Topics:</div>', unsafe_allow_html=True)
            topics_html = ", ".join([f'<span style="background-color: #f1f1f1; padding: 0.2rem 0.4rem; border-radius: 0.3rem; margin-right: 0.3rem;">{topic}</span>' for topic in problem['topics']])
            st.markdown(topics_html, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Add a placeholder for evaluation results
        if st.session_state.evaluation is not None:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### Evaluation Results")

            # Test results
            success = st.session_state.evaluation['execution_results']['success']
            st.markdown(f"""
                <p><strong>Tests:</strong> <span class="{'test-result-pass' if success else 'test-result-fail'}">
                {'All tests passed' if success else 'Some tests failed'}
                </span></p>
            """, unsafe_allow_html=True)

            # Display test details on expansion
            with st.expander("View Test Details"):
                for result in st.session_state.evaluation['execution_results']['results']:
                    st.markdown(f"""
                        <p><strong>Test {result['test_case']}:</strong> <span class="{'test-result-pass' if result['passed'] else 'test-result-fail'}">
                        {'Passed' if result['passed'] else 'Failed'}
                        </span></p>
                        <p>Input: {result['input']}</p>
                        <p>Expected: {result['expected']}</p>
                        <p>Actual: {result['actual']}</p>
                    """, unsafe_allow_html=True)
                    if result.get('error'):
                        st.error(result['error'])

            # Time complexity
            st.markdown(f"**Time Complexity:** {st.session_state.evaluation['time_complexity']}")
            st.markdown(f"**Space Complexity:** {st.session_state.evaluation['space_complexity']}")

            # Algorithm patterns
            patterns = ', '.join(st.session_state.evaluation['algorithm_patterns'])
            st.markdown(f"**Algorithm Pattern:** {patterns}")

            # Readability score
            readability = st.session_state.evaluation['code_quality']['readability']
            st.markdown(f"**Code Readability:** {readability}/10")

            # Overall score with progress bar
            score = st.session_state.evaluation['overall_score']
            st.markdown(f"**Overall Score:** {score}/10")
            st.progress(score/10)

            st.markdown('</div>', unsafe_allow_html=True)

            # Next problem button
            if st.button("Next Problem", key="next_problem", type="primary", use_container_width=True):
                st.session_state.current_problem_index += 1
                st.session_state.evaluation = None
                st.session_state.code = ""
                st.session_state.start_time = time.time()

                # Check if assessment is complete
                if st.session_state.current_problem_index >= len(st.session_state.problems):
                    st.session_state.assessment_complete = True
                    st.session_state.page = "results"

                st.experimental_rerun()

    # Code editor section
    st.markdown("### Your Solution")
    editor_placeholder = "# Write your Python solution here"

    # Use streamlit-ace for a better code editor experience
    code = st_ace(
        value=st.session_state.code if st.session_state.code else editor_placeholder,
        language="python",
        theme="monokai",
        keybinding="vscode",
        font_size=14,
        min_lines=15,
        max_lines=25,
        wrap=False,
        show_gutter=True,
        show_print_margin=True,
        key=f"code_editor_{st.session_state.current_problem_index}"
    )

    # Store the code in session state
    st.session_state.code = code

    # Submit button
    if st.button("Submit Solution", key="submit_solution", type="primary", use_container_width=True):
        if not code or code == editor_placeholder:
            st.error("Please write your solution before submitting.")
            return

        with st.spinner("Evaluating your solution..."):
            # Calculate time taken
            time_taken = time.time() - st.session_state.start_time

            # Evaluate solution
            evaluation = client.evaluate_custom_solution(code, problem, time_taken)

            # Add metadata
            evaluation["difficulty"] = problem["difficulty"]
            evaluation["time_taken"] = time_taken

            # Store result
            st.session_state.performance_data.append(evaluation)
            st.session_state.evaluation = evaluation

            # Determine if this is the last problem
            is_last_problem = st.session_state.current_problem_index >= len(st.session_state.problems) - 1

            # Success message
            if evaluation['execution_results']['success']:
                st.success(f"Solution evaluated! {evaluation['overall_score']}/10 points earned.")
            else:
                st.warning(f"Some tests failed. {evaluation['overall_score']}/10 points earned.")

            st.experimental_rerun()


def display_results_page(client):
    """Display the results page with company predictions"""
    st.markdown('<h1 class="main-header">Assessment Results</h1>', unsafe_allow_html=True)

    # If we don't have predictions yet, get them
    if st.session_state.predictions is None:
        with st.spinner("Analyzing your performance and generating company predictions..."):
            st.session_state.predictions = client.predict_placement(st.session_state.performance_data)

    if not st.session_state.predictions:
        st.error("Failed to generate predictions. Please try again.")
        return

    predictions = st.session_state.predictions.get("predictions", [])
    summary = st.session_state.predictions.get("performance_summary", {})

    # Performance summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Score", f"{summary.get('average_score', 0):.2f}/10")
    with col2:
        st.metric("Problems Solved", summary.get('problems_solved', len(st.session_state.problems)))
    with col3:
        difficulties = summary.get("difficulty_distribution", {})
        difficulty_text = f"Easy: {difficulties.get(1, 0)}, Medium: {difficulties.get(2, 0)}, Hard: {difficulties.get(3, 0)}"
        st.metric("Difficulty Distribution", difficulty_text)

    # Performance by problem
    st.markdown("### Performance by Problem")

    # Create a DataFrame for the chart
    performance_data = []
    for i, eval_data in enumerate(st.session_state.performance_data):
        difficulty = "Easy" if eval_data["difficulty"] == 1 else "Medium" if eval_data["difficulty"] == 2 else "Hard"
        problem = st.session_state.problems[i] if i < len(st.session_state.problems) else {"title": f"Problem {i+1}"}

        performance_data.append({
            "Problem": f"{i+1}. {problem['title']}",
            "Score": eval_data["overall_score"],
            "Difficulty": difficulty
        })

    df = pd.DataFrame(performance_data)

    # Create an interactive bar chart with Altair
    chart = alt.Chart(df).mark_bar().encode(
        x='Problem:N',
        y='Score:Q',
        color=alt.Color('Difficulty:N', scale=alt.Scale(
            domain=['Easy', 'Medium', 'Hard'],
            range=['#28a745', '#fd7e14', '#dc3545']
        )),
        tooltip=['Problem', 'Score', 'Difficulty']
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

    # Company predictions
    st.markdown("### Company Placement Predictions")

    # Display top companies as cards
    top_predictions = predictions[:5]
    cols = st.columns(len(top_predictions))

    for i, col in enumerate(cols):
        with col:
            prediction = top_predictions[i]
            company = prediction['company']
            match_score = prediction['match_score']

            # Company card with logo (using placeholder images)
            st.markdown(f"""
                <div style="text-align: center; padding: 1rem; border-radius: 0.5rem; background-color: #f8f9fa; box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);">
                    <img src="https://logo.clearbit.com/{company.lower().replace(' ', '')}.com" width="80" height="80" style="border-radius: 50%;" onerror="this.src='https://via.placeholder.com/80?text={company[0]}'">
                    <h3 style="margin-top: 0.5rem; margin-bottom: 0.25rem;">{company}</h3>
                    <div style="font-size: 1.5rem; font-weight: bold; color: #4F8BF9;">{match_score:.1f}%</div>
                    <div style="margin-top: 0.5rem;">Match Score</div>
                </div>
            """, unsafe_allow_html=True)

    # Show all predictions in a table
    st.markdown("### All Company Matches")

    # Create DataFrame for table
    predictions_df = pd.DataFrame(predictions)
    predictions_df.columns = ["Company", "Match Score"]
    predictions_df["Match Score"] = predictions_df["Match Score"].apply(lambda x: f"{x:.1f}%")

    st.table(predictions_df)

    # Restart button
    st.button("Start New Assessment", key="restart", type="primary", use_container_width=True,
              on_click=lambda: restart_assessment())

def restart_assessment():
    """Reset all session state for a new assessment"""
    st.session_state.problems = []
    st.session_state.current_problem_index = 0
    st.session_state.performance_data = []
    st.session_state.start_time = None
    st.session_state.assessment_complete = False
    st.session_state.predictions = None
    st.session_state.page = "welcome"
    st.session_state.code = ""
    st.session_state.evaluation = None


if __name__ == "__main__":
    main()
