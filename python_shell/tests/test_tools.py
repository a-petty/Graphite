import pytest
from pathlib import Path
import tempfile
from cortex.agent import CortexAgent

def test_reflexive_loop_rejects_invalid_code():
    """
    Verifies that the ToolExecutor's write_file method refuses to save
    a file with syntax errors, meeting the 'Reflexive Sensory Loop' criteria.
    """
    # 1. SETUP: Create an isolated agent and project directory
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        # We don't need to initialize the agent for this test, as we are
        # testing the tool executor directly.
        agent = CortexAgent(project_root=temp_dir)
        
        # Directly access the agent's tool executor
        tools = agent.tools
        
        # 2. DEFINE INVALID CODE
        invalid_python_code = "a = 1\nTHIS IS NOT VALID PYTHON\nb = 2"
        file_to_write = "bad_code.py"

        # 3. ACTION: Attempt to write the syntactically invalid file
        result = tools.write_file(file_to_write, invalid_python_code)

        # 4. VERIFICATION
        
        # Criterion 1: Prevents saving invalid code
        assert not (temp_dir / file_to_write).exists(), "The file with invalid syntax should NOT have been created."
            
        # Criterion 3: Agent receives a clear error message
        assert not result['success'], "The tool call should report failure."
        assert result['status'] == 'refused_by_parser', "The failure status must indicate it was refused by the parser."
        assert "Syntax Error" in result['error'], "The error message should explicitly mention a syntax error."
