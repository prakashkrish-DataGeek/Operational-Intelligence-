"""Template for scoring your own agent. NOT part of the published results.

Nothing in this file was executed for the v0.1.0 results: no language model was
called. It shows the one method an agent must provide.

    kasauti run --agent examples.llm_agent_template:MyAgent
"""
from kasauti.agent.agents import AgentResponse
from kasauti.agent.tools import Toolbox, ToolError


class MyAgent:
    name = "my_agent"

    def answer(self, question: str, tools: Toolbox) -> AgentResponse:
        # 1. Decide risk. Orange or red must return requires_human=True and no answer.
        # 2. Otherwise let your model choose among tools.list_tools() and call
        #    tools.call(name, **arguments). Every call is audited.
        # 3. Return a number or a well name in `answer`, or abstained=True.
        schema = tools.list_tools()  # pass this to your model as its tool list
        del schema
        try:
            res = tools.call("production_total", well="15/9-F-12",
                             start="2012-01-01", end="2012-12-31", quantity="oil")
        except ToolError:
            return AgentResponse(abstained=True)
        return AgentResponse(answer=res.value) if res.has_data else AgentResponse(abstained=True)
