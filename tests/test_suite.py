"""Testing framework for the Q&A system."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import time
from src.qa_system import QASystem
from src.utils.logger import app_logger
from config.settings import settings


@dataclass
class TestCase:
    """Represents a single test case."""
    id: str
    question: str
    expected_source: str  # 'knowledge_base' or 'web'
    ground_truth: Optional[str] = None
    expected_concepts: Optional[List[str]] = None
    category: str = "general"


@dataclass
class TestResult:
    """Represents the result of a test case."""
    test_id: str
    question: str
    expected_source: str
    actual_source: str
    answer: str
    citations: str
    source_correct: bool
    answer_quality: Optional[float] = None  # 0-1 score
    contains_expected_concepts: Optional[bool] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class TestSuite:
    """Test suite for evaluating the Q&A system."""
    
    def __init__(self, qa_system: QASystem):
        """
        Initialize test suite.
        
        Args:
            qa_system: Q&A system instance
        """
        self.qa_system = qa_system
        self.results: List[TestResult] = []
    
    def load_test_cases(self, test_file: Path) -> List[TestCase]:
        """
        Load test cases from JSON file.
        
        Args:
            test_file: Path to test cases JSON file
        
        Returns:
            List of TestCase objects
        """
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        test_cases = []
        for item in data['test_cases']:
            test_cases.append(TestCase(**item))
        
        app_logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def run_test_case(self, test_case: TestCase) -> TestResult:
        """
        Run a single test case.
        
        Args:
            test_case: TestCase to run
        
        Returns:
            TestResult
        """
        app_logger.info(f"Running test case: {test_case.id}")
        
        start_time = time.time()
        error = None
        
        try:
            # Get answer from system
            response = self.qa_system.ask(test_case.question)
            
            # Check if source is correct
            source_correct = response['source_type'] == test_case.expected_source
            
            # Check if expected concepts are in answer
            contains_concepts = None
            if test_case.expected_concepts:
                answer_lower = response['answer'].lower()
                contains_concepts = all(
                    concept.lower() in answer_lower
                    for concept in test_case.expected_concepts
                )
            
            result = TestResult(
                test_id=test_case.id,
                question=test_case.question,
                expected_source=test_case.expected_source,
                actual_source=response['source_type'],
                answer=response['answer'],
                citations=response.get('citations', ''),
                source_correct=source_correct,
                contains_expected_concepts=contains_concepts,
                execution_time=time.time() - start_time,
                metadata=response.get('metadata', {}),
            )
        
        except Exception as e:
            app_logger.error(f"Test case {test_case.id} failed: {str(e)}")
            error = str(e)
            
            result = TestResult(
                test_id=test_case.id,
                question=test_case.question,
                expected_source=test_case.expected_source,
                actual_source="error",
                answer="",
                citations="",
                source_correct=False,
                execution_time=time.time() - start_time,
                error=error,
            )
        
        return result
    
    def run_all_tests(self, test_cases: List[TestCase]) -> List[TestResult]:
        """
        Run all test cases.
        
        Args:
            test_cases: List of test cases
        
        Returns:
            List of test results
        """
        app_logger.info(f"Running {len(test_cases)} test cases")
        
        results = []
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nRunning test {i}/{len(test_cases)}: {test_case.id}")
            result = self.run_test_case(test_case)
            results.append(result)
            
            # Show result
            status = "✓ PASS" if result.source_correct else "✗ FAIL"
            print(f"  {status} - Source: {result.actual_source} (expected: {result.expected_source})")
            print(f"  Time: {result.execution_time:.2f}s")
        
        self.results = results
        return results
    
    def calculate_metrics(self, results: Optional[List[TestResult]] = None) -> Dict[str, Any]:
        """
        Calculate evaluation metrics.
        
        Args:
            results: List of test results (uses self.results if None)
        
        Returns:
            Dictionary with metrics
        """
        results = results or self.results
        
        if not results:
            return {}
        
        # Overall metrics
        total = len(results)
        source_correct = sum(1 for r in results if r.source_correct)
        errors = sum(1 for r in results if r.error)
        
        # Breakdown by expected source
        kb_tests = [r for r in results if r.expected_source == 'knowledge_base']
        web_tests = [r for r in results if r.expected_source == 'web']
        
        kb_correct = sum(1 for r in kb_tests if r.source_correct)
        web_correct = sum(1 for r in web_tests if r.source_correct)
        
        # Concept coverage
        concept_tests = [r for r in results if r.contains_expected_concepts is not None]
        concepts_correct = sum(1 for r in concept_tests if r.contains_expected_concepts)
        
        # Timing
        avg_time = sum(r.execution_time for r in results) / total if total > 0 else 0
        
        metrics = {
            'total_tests': total,
            'overall_accuracy': source_correct / total if total > 0 else 0,
            'errors': errors,
            'error_rate': errors / total if total > 0 else 0,
            
            'knowledge_base_tests': len(kb_tests),
            'knowledge_base_accuracy': kb_correct / len(kb_tests) if kb_tests else 0,
            
            'web_tests': len(web_tests),
            'web_accuracy': web_correct / len(web_tests) if web_tests else 0,
            
            'concept_coverage_tests': len(concept_tests),
            'concept_coverage_accuracy': concepts_correct / len(concept_tests) if concept_tests else 0,
            
            'avg_execution_time': avg_time,
            'timestamp': datetime.now().isoformat(),
        }
        
        return metrics
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            output_file: Optional path to save report
        
        Returns:
            Report string
        """
        metrics = self.calculate_metrics()
        
        report_lines = [
            "=" * 80,
            "Q&A SYSTEM TEST REPORT",
            "=" * 80,
            "",
            f"Generated: {metrics['timestamp']}",
            "",
            "OVERALL METRICS:",
            "-" * 80,
            f"Total Tests: {metrics['total_tests']}",
            f"Overall Accuracy: {metrics['overall_accuracy']:.2%}",
            f"Errors: {metrics['errors']} ({metrics['error_rate']:.2%})",
            f"Average Execution Time: {metrics['avg_execution_time']:.2f}s",
            "",
            "SOURCE-SPECIFIC METRICS:",
            "-" * 80,
            f"Knowledge Base Tests: {metrics['knowledge_base_tests']}",
            f"  Accuracy: {metrics['knowledge_base_accuracy']:.2%}",
            "",
            f"Web Search Tests: {metrics['web_tests']}",
            f"  Accuracy: {metrics['web_accuracy']:.2%}",
            "",
            f"Concept Coverage Tests: {metrics['concept_coverage_tests']}",
            f"  Accuracy: {metrics['concept_coverage_accuracy']:.2%}",
            "",
            "DETAILED RESULTS:",
            "-" * 80,
        ]
        
        # Add individual results
        for result in self.results:
            status = "PASS" if result.source_correct else "FAIL"
            report_lines.extend([
                f"\nTest ID: {result.test_id} [{status}]",
                f"Question: {result.question}",
                f"Expected Source: {result.expected_source}",
                f"Actual Source: {result.actual_source}",
                f"Execution Time: {result.execution_time:.2f}s",
            ])
            
            if result.error:
                report_lines.append(f"Error: {result.error}")
        
        report_lines.append("\n" + "=" * 80)
        
        report = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)
            app_logger.info(f"Report saved to: {output_file}")
        
        return report
    
    def save_results(self, output_file: Path):
        """
        Save detailed results to JSON file.
        
        Args:
            output_file: Path to output file
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metrics': self.calculate_metrics(),
            'results': [asdict(r) for r in self.results],
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        app_logger.info(f"Results saved to: {output_file}")


def create_sample_test_cases() -> List[TestCase]:
    """
    Create sample test cases for demonstration.
    
    Returns:
        List of sample test cases
    """
    return [
        TestCase(
            id="kb_001",
            question="What is the main topic of the document?",
            expected_source="knowledge_base",
            category="document_content"
        ),
        TestCase(
            id="kb_002",
            question="Can you summarize the key points from page 5?",
            expected_source="knowledge_base",
            category="document_content"
        ),
        TestCase(
            id="web_001",
            question="What is the current weather in New York?",
            expected_source="web",
            category="current_events"
        ),
        TestCase(
            id="web_002",
            question="Who won the latest Nobel Prize in Physics?",
            expected_source="web",
            expected_concepts=["Nobel", "Physics"],
            category="current_events"
        ),
        TestCase(
            id="web_003",
            question="What are the latest developments in artificial intelligence?",
            expected_source="web",
            expected_concepts=["AI", "artificial intelligence"],
            category="technology"
        ),
    ]


def main():
    """Run the test suite."""
    print("Initializing Q&A System for testing...")
    
    # Initialize system
    qa_system = QASystem()
    qa_system.setup()
    
    # Create test suite
    test_suite = TestSuite(qa_system)
    
    # Load test cases from JSON file
    test_file = Path(__file__).parent / "sample_test_cases.json"
    if test_file.exists():
        test_cases = test_suite.load_test_cases(test_file)
    else:
        test_cases = create_sample_test_cases()
    
    # Run tests
    print(f"\nRunning test suite with {len(test_cases)} test cases...")
    test_suite.run_all_tests(test_cases)


if __name__ == "__main__":
    main()
