import re
from typing import Dict, List, Tuple, Optional, Set
import sqlparse
from collections import defaultdict


class SQLNestingAnalyzer:
    def __init__(self, debug=False):
        """
        A class that analyzes the nesting structure of SQL queries.

        Args:
            debug: Whether to print debug information
        """
        self.debug = debug

    def analyze_query(self, query: str) -> Dict:
        """
        Analyze the nesting structure of a single SQL query.

        Returns:
            Dict: {
                'is_nested': bool,
                'height': int,
                'breadth': dict,  # {level: count}
                'max_breadth': int,
                'type': str  # 'non-nested' or 'nested'
            }
        """
        # Format the query for SQL parsing
        formatted_query = sqlparse.format(query, reindent=True, keyword_case="upper")

        # Analyze subquery structure
        structure = self._analyze_structure(formatted_query)

        if structure["height"] == 0:
            return {
                "is_nested": False,
                "height": 0,
                "breadth": {},
                "max_breadth": 0,
                "type": "non-nested",
            }

        max_breadth = max(structure["breadth"].values()) if structure["breadth"] else 0

        return {
            "is_nested": True,
            "height": structure["height"],
            "breadth": structure["breadth"],
            "max_breadth": max_breadth,
            "type": f'nested (height={structure["height"]}, max_breadth={max_breadth})',
        }

    def _find_matching_parenthesis(self, query: str, start: int) -> int:
        """
        Find the closing parenthesis matching the opening parenthesis at the given position.
        """
        count = 1
        i = start + 1
        while i < len(query) and count > 0:
            if query[i] == "(":
                count += 1
            elif query[i] == ")":
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    def _extract_all_subqueries(self, query: str) -> List[Dict]:
        """
        Extract all subqueries and store their positional information.
        More accurate version.
        """
        subqueries = []
        i = 0

        while i < len(query):
            if query[i] == "(":
                # Check if SELECT follows the parenthesis
                j = i + 1
                while j < len(query) and query[j] in " \n\t\r":
                    j += 1

                if j + 6 <= len(query) and query[j : j + 6] == "SELECT":
                    # Subquery found
                    end = self._find_matching_parenthesis(query, i)
                    if end != -1:
                        subqueries.append(
                            {
                                "start": i,
                                "end": end,
                                "select_start": j,
                                "content": query[i : end + 1],
                                "depth": 1,
                                "parent": None,
                                "children": [],
                            }
                        )
                        if self.debug:
                            print(
                                f"Found subquery at {i}-{end}: {query[i:min(i+50, end+1)]}"
                            )
            i += 1

        return subqueries

    def _analyze_structure(self, query: str) -> Dict:
        """
        Recursively analyze the overall structure of the query.
        """
        query_upper = query.upper()

        # Find all subqueries and analyze their structure
        subqueries = self._extract_all_subqueries(query_upper)

        if not subqueries:
            return {"height": 0, "breadth": {}}

        # Determine the depth and parent relationship of each subquery
        tree_structure = self._build_subquery_tree(subqueries, query_upper)

        return tree_structure

    def _build_subquery_tree(self, subqueries: List[Dict], full_query: str) -> Dict:
        """
        Build the tree structure of subqueries and compute height and breadth.
        Accurately determines parent-child relationships.
        """
        if not subqueries:
            return {"height": 0, "breadth": {}}

        # Sort subqueries by start position
        subqueries.sort(key=lambda x: (x["start"], -x["end"]))

        # Check containment relationships - find the direct parent of each subquery
        for i, child in enumerate(subqueries):
            best_parent = None
            best_parent_idx = None

            for j, potential_parent in enumerate(subqueries):
                if i != j:
                    # Check if child is completely contained within potential_parent
                    if (
                        potential_parent["start"] < child["start"]
                        and child["end"] < potential_parent["end"]
                    ):

                        # Find the closest parent (the one with the smallest range)
                        if best_parent is None or (
                            potential_parent["start"] > best_parent["start"]
                            and potential_parent["end"] < best_parent["end"]
                        ):
                            best_parent = potential_parent
                            best_parent_idx = j

            if best_parent_idx is not None:
                child["parent"] = best_parent_idx
                subqueries[best_parent_idx]["children"].append(i)
                if self.debug:
                    print(f"Subquery {i} is child of {best_parent_idx}")

        # Calculate depth
        def calculate_depth(sq_idx, current_depth=1):
            sq = subqueries[sq_idx]
            sq["depth"] = current_depth
            max_depth = current_depth

            for child_idx in sq["children"]:
                child_depth = calculate_depth(child_idx, current_depth + 1)
                max_depth = max(max_depth, child_depth)

            return max_depth

        # Find root subqueries (those without a parent)
        roots = [i for i, sq in enumerate(subqueries) if sq["parent"] is None]

        if self.debug:
            print(f"Root subqueries: {roots}")

        max_height = 0
        for root_idx in roots:
            height = calculate_depth(root_idx)
            max_height = max(max_height, height)

        # Count the number of subqueries at each level
        breadth = defaultdict(int)
        for sq in subqueries:
            breadth[sq["depth"]] += 1

        if self.debug:
            print(f"Breadth by level: {dict(breadth)}")
            print(f"Max height: {max_height}")

        return {"height": max_height, "breadth": dict(breadth)}

    def analyze_queries(self, queries: List[str]) -> List[Dict]:
        """
        Analyze multiple SQL queries.
        """
        results = []
        for i, query in enumerate(queries):
            try:
                result = self.analyze_query(query)
                result["query_index"] = i
                result["query"] = query[:100] + "..." if len(query) > 100 else query
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "query_index": i,
                        "query": query[:100] + "..." if len(query) > 100 else query,
                        "error": str(e),
                        "is_nested": None,
                        "height": None,
                        "breadth": None,
                        "type": "error",
                    }
                )

        return results

    def print_analysis(self, results: List[Dict]):
        """
        Pretty-print the analysis results.
        """
        print("=" * 80)
        print("SQL Query Nesting Analysis Results")
        print("=" * 80)

        for result in results:
            print(f"\nQuery #{result['query_index'] + 1}:")
            print(f"  Preview: {result['query']}")

            if "error" in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Type: {result['type']}")
                print(f"  Is Nested: {result['is_nested']}")

                if result["is_nested"]:
                    print(f"  Height: {result['height']}")
                    print(f"  Max Breadth: {result['max_breadth']}")
                    print(f"  Breadth by Level: {result['breadth']}")

        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)

        valid_results = [r for r in results if "error" not in r]
        non_nested = sum(1 for r in valid_results if not r["is_nested"])
        nested = sum(1 for r in valid_results if r["is_nested"])

        print(f"Total Queries: {len(results)}")
        print(f"Non-nested Queries: {non_nested}")
        print(f"Nested Queries: {nested}")

        if nested > 0:
            max_height = max(
                (r["height"] for r in valid_results if r["is_nested"]), default=0
            )
            max_breadth = max(
                (r["max_breadth"] for r in valid_results if r["is_nested"]), default=0
            )
            print(f"Maximum Height: {max_height}")
            print(f"Maximum Breadth: {max_breadth}")

    def visualize_structure(self, query: str):
        """
        Visualize and print the query structure (improved version).
        """
        print("\n" + "=" * 80)
        print("Query Structure Visualization")
        print("=" * 80)

        query_upper = query.upper()
        subqueries = self._extract_all_subqueries(query_upper)

        if not subqueries:
            print("No nested subqueries found.")
            return

        tree = self._build_subquery_tree(subqueries, query_upper)

        print("\nDetailed Subquery Information:")
        print("-" * 40)

        for i, sq in enumerate(subqueries):
            print(f"\nSubquery #{i}:")
            print(f"  Position: {sq['start']}-{sq['end']}")
            print(f"  Depth: {sq['depth']}")
            print(f"  Parent: {sq['parent']}")
            print(f"  Children: {sq['children']}")
            # Content preview
            content_preview = sq["content"][:60].replace("\n", " ").replace("  ", " ")
            if len(sq["content"]) > 60:
                content_preview += "..."
            print(f"  Content: {content_preview}")

        print("\n" + "-" * 40)
        print("Subquery Tree Structure:")
        print("-" * 40)

        # Print tree structure
        def print_tree(sq_idx, indent=0, prefix="", is_last=True):
            if sq_idx == -1:  # Main query
                print("Main Query")
                # Print root subqueries
                roots = [i for i, sq in enumerate(subqueries) if sq["parent"] is None]
                for i, root in enumerate(roots):
                    is_last_root = i == len(roots) - 1
                    print_tree(root, indent + 2, "", is_last_root)
            else:
                sq = subqueries[sq_idx]
                # Extract a portion of the subquery content
                content = (
                    sq["content"][1:50] if len(sq["content"]) > 1 else sq["content"]
                )
                content = " ".join(content.split())  # Normalize whitespace
                if len(sq["content"]) > 50:
                    content += "..."

                connector = "└── " if is_last else "├── "
                print(f"{prefix}{connector}[#{sq_idx}, Level {sq['depth']}]: {content}")

                # Print child subqueries
                children = sq["children"]
                for i, child_idx in enumerate(children):
                    is_last_child = i == len(children) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(child_idx, indent + 2, new_prefix, is_last_child)

        print_tree(-1)  # Start from the main query

        print("\n" + "-" * 40)
        print(f"Analysis Summary:")
        print(f"  Total Height: {tree['height']}")
        print(f"  Breadth by Level: {tree['breadth']}")
        print(
            f"  Max Breadth: {max(tree['breadth'].values()) if tree['breadth'] else 0}"
        )


# Usage example
if __name__ == "__main__":
    # Problem query - linear nesting structure (Height=3, 1 per level)
    problem_query = """SELECT MAX(salary) AS highest_paid_player_salary FROM nba_player_affiliation 
    WHERE salary >= (
        SELECT MAX(T2.salary) FROM nba_player_affiliation AS T2 WHERE T2.player_name IN (
            SELECT T3.player_name FROM nba_player_affiliation AS T3 WHERE T3.salary > (
                SELECT MAX(T4.salary) FROM nba_player_affiliation AS T4 
                WHERE T4.salary > 1000000 AND T4.season = '2006-07'
            ) 
            AND T3.salary > 20000000 AND T3.team_name = 'Lakers' AND T3.team_name = T2.team_name
        ) 
        AND T2.team_name = nba_player_affiliation.team_name
    ) 
    AND team_name = 'Lakers' AND season = '2013-14'"""

    # Create analyzer in debug mode
    analyzer = SQLNestingAnalyzer(debug=True)

    print("=" * 80)
    print("Analyzing Linear Nested Query (Expected: Height=3, Max_Breadth=1)")
    print("=" * 80)

    # Detailed analysis
    result = analyzer.analyze_query(problem_query)
    print(f"\nAnalysis Result:")
    print(f"  Height: {result['height']}")
    print(f"  Breadth by Level: {result['breadth']}")
    print(f"  Max Breadth: {result['max_breadth']}")
    print(f"  Type: {result['type']}")

    # Structure visualization
    analyzer.visualize_structure(problem_query)

    # Compare with other test queries
    print("\n" + "=" * 80)
    print("Comparison with Other Query Types")
    print("=" * 80)

    test_queries = [
        # 1. Linear nested (Height=3, Breadth=1 each level)
        problem_query,
        # 2. Parallel subqueries (Height=1, Breadth=2)
        """SELECT * FROM orders 
           WHERE user_id IN (SELECT id FROM users WHERE country = 'USA')
           AND product_id IN (SELECT id FROM products WHERE price > 100)""",
        # 3. Mixed structure (Height=2, varying breadth)
        """SELECT * FROM orders WHERE product_id IN (
               SELECT id FROM products 
               WHERE category_id IN (SELECT id FROM categories WHERE active = 1)
               AND supplier_id IN (SELECT id FROM suppliers WHERE country = 'US')
           )""",
    ]

    # Turn off debug mode and print results only
    analyzer_clean = SQLNestingAnalyzer(debug=False)
    results = analyzer_clean.analyze_queries(test_queries)

    print("\nComparison Results:")
    print("-" * 40)
    for i, r in enumerate(results):
        if "error" not in r:
            print(f"Query #{i+1}:")
            print(
                f"  Height: {r['height']}, Breadth: {r['breadth']}, Max Breadth: {r['max_breadth']}"
            )
