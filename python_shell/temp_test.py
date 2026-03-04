import sys
from pathlib import Path
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cortex.agent import CortexAgent

def main():
    """
    Manual test script to verify the agent's query functionality.
    """
    # Use the test_repo included in the project for a consistent test environment
    test_repo_path = project_root / "test_repo"
    
    if not test_repo_path.exists():
        print(f"Error: Test repository not found at {test_repo_path}")
        sys.exit(1)

    print(f"Initializing agent on repository: {test_repo_path}...")
    
    # Create and initialize agent
    agent = CortexAgent(test_repo_path)
    agent.initialize()

    # --- Test Query ---
    # This query should trigger the context assembly process
    test_query = "how is the fizz component implemented?"
    
    print(f"\n--- Running Test Query ---\nQuery: '{test_query}'")
    
    # Execute the query
    response = agent.query(test_query)
    
    # Print the response, which should include the assembled context
    print("\n--- Agent Response ---")
    print(response)
    print("\n--- Verification ---")
    print("Please manually verify that the 'Assembled Context' section above contains:")
    print("1. An 'ARCHITECTURE_MAP'.")
    print("2. 'FILE (anchor):' sections for files likely related to 'fizz'.")
    print("3. 'DEPENDENCIES (skeletons):' sections for files related to the anchors.")
    print("4. A final 'LLM Response Placeholder' with the assembled context.")

if __name__ == "__main__":
    main()
