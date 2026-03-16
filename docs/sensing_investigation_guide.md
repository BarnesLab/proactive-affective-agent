# **Proactive Affective Agent for JITAI via Multi-Modal Sensing & LLM Agency**

Moving Beyond Reactive Inference: An Autonomous Agent Framework for Emotion Intervention

## **1\. Motivation: The Shift from Inference to Agency**

Our work in CALLM (CHI '26) established that LLMs can infer affect and intervention opportunities from ultra-brief digital diaries with high accuracy. However, this paradigm is still **reactive**, as it requires users to initiate the process by writing a diary.

Inspired by next-generation agentic frameworks like **Clawdbot** and **Manus**, we propose a **Proactive Affective Agent**. Instead of waiting for a prompt or a diary entry, the agent operates in a continuous "Sense-Think-Act" loop. It monitors raw **smartphone sensing signals** in the background and autonomously decides when to trigger a reasoning cycle, when to retrieve historical context, and when to deliver a proactive intervention.

## **2\. Agentic Insights: Proactive Reasoning & Deep Retrieval**

The recent success of autonomous agents enables three critical shifts for our research:

* **Proactive Ticker:** The agent runs on a 5-minute autonomous cycle, treating the continuous stream of sensing data (GPS, activity, screen-on time series, diaries) as its environmental observation.  
* **Agentic Retrieval (Deep RAG):** When the agent detects an anomaly in digital behavior, it doesn't just look for "similar text." It autonomously queries the CALLM diary database: *"Find historical states where this user exhibited similar sedentary patterns but negative affect."*  
* **Self-Correction & Memory Updates:** Unlike static models, the agent updates its internal "User Memory" after every observation, allowing it to learn the user's shifting "Receptivity" thresholds over time.

## **3\. System Architecture**

### **A. Core Components**

1. **Signal Ingestor:** Continuously processes 15-minute (window size to be tested) windows of passive sensing data.  
2. **Reasoning Brain (LLM Agent):** Evaluates whether current patterns warrant a deeper investigation into the user's affective state.  
3. **Dynamic Memory Retriever:** Autonomously retrieves similarity-aligned peer cases and the user's personal diary history.  
4. **Action Executor:** Delivers the intervention if both **Subjective Desire** and **Objective Availability** (Opportunity) are confirmed.

### **B. Technical Workflow** 

### **C. Implementation Pseudo-code**

Python  
class ProactiveAffectiveAgent:  
    def \_\_init\_\_(self, user\_id):  
        self.brain \= CALLM\_Engine(model="gemini-1.5-pro")  
        self.memory \= VectorStore(user\_id) \# Includes historical diaries

    def run\_autonomous\_cycle(self):  
        \# 1\. Sense: Autonomous ingestion of sensing streams  
        current\_signals \= mobile\_api.fetch\_signals(window=5)  
          
        \# 2\. Reason: Deep retrieval & multi-step inference  
        if self.brain.detect\_potential\_need(current\_signals):  
            \# Agentic RAG: Decide what context is needed to validate the shift  
            relevant\_history \= self.memory.retrieve\_context(current\_signals.features)  
              
            \# 3\. Decision: Balance Desire vs. Availability (Clawdbot-style)  
            decision \= self.brain.reason\_opportunity(  
                current=current\_signals,  
                history=relevant\_history  
            )  
              
            if decision.deliver\_intervention:  
                jitai\_service.trigger(decision.payload)  
                  
            \# 4\. Final Step: Update long-term memory and go to sleep  
            self.memory.update\_state\_trajectory(current\_signals, decision)  
            self.standby()

\# Orchestrator  
scheduler.add\_job(agent.run\_autonomous\_cycle, 'interval', minutes=5)

## **4\. Evaluation Strategy: Retrospective Validation via CALLM Dataset**

To validate the agent's decision-making accuracy before deployment, we will perform a **Retrospective Evaluation** using the existing CALLM dataset:

* **Data Alignment:** We will align the 24,183 diary entries with the corresponding **mobile sensing data (unused in CALLM CHI’26 work)** collected during the same period.  
* **Blind Simulation:** We will feed the agent only the sensing data and observe if its proactive "Intervention Decisions" match the "Subjective Desire" and "Availability" labels actually reported by the participants in their diaries.  
* **Metric:** We will measure the "Pre-emptive & Post-emptive Accuracy"—the agent's ability to predict a state of need *before* and *right after* the user actually submits a diary entry.

