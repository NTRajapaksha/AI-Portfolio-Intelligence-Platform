"""
LangGraph agent orchestration
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from datetime import datetime
import time
from config import config
from tools import ALL_TOOLS, DATA_STATE, reset_state

# Agent State Definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

class PortfolioAnalysisAgent:
    """Main agent class with multiple execution modes"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            api_key=config.GOOGLE_API_KEY
        )
        self.llm_with_tools = self.llm.bind_tools(ALL_TOOLS)
        self.app = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        def agent_node(state: AgentState):
            return {"messages": [self.llm_with_tools.invoke(state["messages"])]}
        
        tool_node = ToolNode(ALL_TOOLS)
        
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    def run_autonomous(self, query: str) -> dict:
        """Run fully autonomous agent with System Prompt guidance"""
        reset_state()
        
        current_date = datetime.now().strftime("%B %d, %Y")
        system_msg = SystemMessage(content=f"""
        You are a Senior Portfolio Manager. Today is {current_date}.
        
        Your Goal: {query}
        
        STRICT INSTRUCTIONS:
        1. Use 'fetch_portfolio_data' first.
        2. Then use 'calculate_risk_metrics'.
        3. Then forecast each stock.
        4. Finally, write an Executive Summary.
        """)
        
        inputs = {"messages": [system_msg, HumanMessage(content=query)]}
        result = {"steps": [], "final_response": "", "data_state": {}}
        
        try:
            # Recursion limit prevents infinite loops
            for event in self.app.stream(inputs, {"recursion_limit": 15}):
                
                # --- NEW: RATE LIMIT PROTECTION ---
                # Pause for 5 seconds between steps to respect Free Tier limits
                time.sleep(5) 
                # ----------------------------------

                for key, value in event.items():
                    messages = value.get("messages", [])
                    if messages:
                        last_msg = messages[-1]
                        content = getattr(last_msg, "content", "")
                        
                        step_name = key.capitalize()
                        if key == "tools":
                            step_name = "Tool Output"
                        
                        result["steps"].append({
                            "step": step_name,
                            "result": str(content)[:300] + "..." if content else "[Data Processed]"
                        })
            
            final_state = self.app.invoke(inputs)
            result["final_response"] = final_state["messages"][-1].content
            result["data_state"] = DATA_STATE
            
        except Exception as e:
            result["final_response"] = f"⚠️ Autonomous Agent Error: {str(e)}"
            result["steps"].append({"step": "Critical Error", "result": str(e)})
            
        return result
    
    # Manual & Hybrid modes (Unchanged)
    def run_manual(self, tickers: str, forecast_days: int = 60, include_sentiment: bool = True) -> dict:
        reset_state()
        from tools import fetch_portfolio_data, calculate_risk_metrics, ensemble_forecast, analyze_sentiment, compare_portfolio
        
        results = {"steps": [], "final_response": "", "data_state": {}}
        
        # Execution Pipeline
        steps = [
            ("Fetching Data", fetch_portfolio_data, {"tickers": tickers}),
            ("Risk Analysis", calculate_risk_metrics, {"tickers": tickers}),
        ]
        
        # Add Forecasts
        for t in tickers.split(','):
            steps.append((f"Forecasting {t.strip()}", ensemble_forecast, {"ticker": t.strip(), "days": forecast_days}))
            
        # Add Sentiment
        if include_sentiment and config.ENABLE_SENTIMENT:
            for t in tickers.split(','):
                steps.append((f"Sentiment {t.strip()}", analyze_sentiment, {"ticker": t.strip()}))
                
        steps.append(("Portfolio Ranking", compare_portfolio, {}))
        
        # Run Steps
        for name, func, args in steps:
            res = func.invoke(args)
            results["steps"].append({"step": name, "result": str(res)})
            
        results["final_response"] = results["steps"][-1]["result"]
        results["data_state"] = DATA_STATE
        return results

    def run_hybrid(self, tickers: str, forecast_days: int = 60, include_sentiment: bool = True) -> dict:
        manual_results = self.run_manual(tickers, forecast_days, include_sentiment)
        current_date = datetime.now().strftime("%B %d, %Y")
        
        synthesis_prompt = f"""
        You are a Senior Financial Analyst. Date: {current_date}.
        
        Based on the data below, provide an Executive Summary with:
        1. Top Pick
        2. Key Risks
        3. Outlook
        
        DATA:
        {chr(10).join([f"{s['step']}: {s['result']}" for s in manual_results['steps']])}
        """
        synthesis = self.llm.invoke(synthesis_prompt)
        manual_results["final_response"] = synthesis.content
        manual_results["synthesis_mode"] = "hybrid"
        return manual_results

agent = PortfolioAnalysisAgent()