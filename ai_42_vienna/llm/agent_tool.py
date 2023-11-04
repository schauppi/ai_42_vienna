from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from typing import Optional

class WriteTool(BaseTool):
    """
    A tool that writes the query to a file
    """
    name = "custom_search"
    description = "useful for when you need to answer questions about current events"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """
        Run the tool

        Args:
            query: the query to run
            run_manager: the run manager
        
        Returns:
            str: message
        """

        with open("ai_42_vienna/llm/output/answer.txt", "w") as f:
            f.write(query)
            
            return "Successfully written to file"